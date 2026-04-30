use crate::config::{Config, ModelsListMode, ResolvedRoute};
use crate::error::{ProxyError, ProxyResult};
use crate::metrics;
use crate::models::{anthropic, openai};
use crate::translate::{pipeline, stream};
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
    Extension, Json,
};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub async fn proxy_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    Json(req): Json<anthropic::AnthropicRequest>,
) -> ProxyResult<Response> {
    let is_streaming = req.stream.unwrap_or(false);
    let start = Instant::now();
    let msg_count = req.messages.len();
    let requested_model = req.model.clone();

    tracing::info!(
        model = requested_model.as_str(),
        stream = is_streaming,
        messages = msg_count,
        "Received request"
    );
    metrics::request_started(is_streaming);

    if config.verbose {
        tracing::trace!(
            "Incoming Anthropic request: {}",
            serde_json::to_string_pretty(&req).unwrap_or_default()
        );
    }

    let route = config
        .resolve_model(&requested_model)
        .map_err(|e| ProxyError::Config(e.to_string()))?;

    let policy = pipeline::TranslationPolicy {
        reasoning_model: route
            .reasoning_model
            .clone()
            .or_else(|| config.reasoning_model.clone()),
        completion_model: route
            .completion_model
            .clone()
            .or_else(|| config.completion_model.clone()),
        model_map: config.model_map.clone(),
        ignore_terms: config.system_prompt_ignore_terms.clone(),
        merge_system_messages: config.merge_system_messages,
        merge_user_messages: config.merge_user_messages,
    };

    let mut req_with_target = req;
    req_with_target.model = route.target.clone();

    let openai_req = pipeline::translate_request(req_with_target, &policy)?;

    if config.verbose {
        tracing::trace!(
            "Transformed OpenAI request: {}",
            serde_json::to_string_pretty(&openai_req).unwrap_or_default()
        );
    }

    let result = if route.is_legacy {
        if is_streaming {
            handle_legacy_streaming(config, client, openai_req, start).await
        } else {
            handle_legacy_non_streaming(config, client, openai_req, start).await
        }
    } else if is_streaming {
        handle_routed_streaming(config, client, openai_req, start, route, requested_model).await
    } else {
        handle_routed_non_streaming(config, client, openai_req, start, route, requested_model).await
    };

    let status = match &result {
        Ok(resp) => resp.status().as_u16(),
        Err(_) => 500,
    };
    metrics::request_finished(start, status, is_streaming);

    result
}

pub async fn list_models_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
) -> ProxyResult<Response> {
    match config.models_list_mode {
        ModelsListMode::Static => {
            let raw_models = config.static_models_list();
            let data: Vec<anthropic::ModelInfo> = raw_models
                .into_iter()
                .map(|(id, display_name)| anthropic::ModelInfo {
                    id,
                    display_name,
                    created_at: "1970-01-01T00:00:00Z".to_string(),
                    model_type: "model".to_string(),
                })
                .collect();

            let first_id = data.first().map(|m| m.id.clone());
            let last_id = data.last().map(|m| m.id.clone());

            Ok(Json(anthropic::ModelsListResponse {
                data,
                first_id,
                has_more: false,
                last_id,
            })
            .into_response())
        }
        ModelsListMode::Upstream => list_models_from_upstream(&config, &client).await,
        ModelsListMode::Merge => {
            let mut config_models = config.static_models_list();
            let config_ids: std::collections::HashSet<String> =
                config_models.iter().map(|(id, _)| id.clone()).collect();
            let config_bare_targets: std::collections::HashSet<String> =
                config_models.iter().map(|(_, d)| d.clone()).collect();

            let upstream_resp = list_models_from_upstream_raw(&config, &client).await?;

            for model in &upstream_resp.data {
                if config_ids.contains(&model.id) || config_bare_targets.contains(&model.id) {
                    continue;
                }
                config_models.push((model.id.clone(), model.id.clone()));
            }

            config_models.sort_by(|a, b| a.0.cmp(&b.0));

            let data: Vec<anthropic::ModelInfo> = config_models
                .into_iter()
                .map(|(id, display_name)| anthropic::ModelInfo {
                    id,
                    display_name,
                    created_at: "1970-01-01T00:00:00Z".to_string(),
                    model_type: "model".to_string(),
                })
                .collect();

            let first_id = data.first().map(|m| m.id.clone());
            let last_id = data.last().map(|m| m.id.clone());

            Ok(Json(anthropic::ModelsListResponse {
                data,
                first_id,
                has_more: false,
                last_id,
            })
            .into_response())
        }
    }
}

async fn list_models_from_upstream(config: &Config, client: &Client) -> ProxyResult<Response> {
    let urls = config.models_urls();
    let mut last_err = None;

    for url in &urls {
        tracing::debug!("Fetching models from {}", url);

        let mut req_builder = client.get(url).timeout(Duration::from_secs(60));
        if let Some(api_key) = &config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        match req_builder.send().await {
            Ok(response) if response.status().is_success() => {
                let openai_resp: openai::ModelsListResponse = response.json().await?;
                let anthropic_resp = pipeline::translate_models_list(openai_resp);
                return Ok(Json(anthropic_resp).into_response());
            }
            Ok(response) => {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
                if is_retriable_status(status.as_u16()) {
                    last_err = Some(format!("Upstream returned {}: {}", status, error_text));
                    continue;
                }
                return Err(ProxyError::Upstream(format!(
                    "Upstream returned {}: {}",
                    status, error_text
                )));
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                last_err = Some(format!("HTTP error: {}", err));
                continue;
            }
        }
    }

    Err(ProxyError::Upstream(
        last_err.unwrap_or_else(|| "All upstreams failed".to_string()),
    ))
}

async fn list_models_from_upstream_raw(
    config: &Config,
    client: &Client,
) -> ProxyResult<openai::ModelsListResponse> {
    let urls = config.models_urls();
    let mut last_err = None;

    for url in &urls {
        let mut req_builder = client.get(url).timeout(Duration::from_secs(60));
        if let Some(api_key) = &config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        match req_builder.send().await {
            Ok(response) if response.status().is_success() => {
                return Ok(response.json().await?);
            }
            Ok(response) => {
                let status = response.status();
                if is_retriable_status(status.as_u16()) {
                    last_err = Some(format!("Upstream returned {}", status));
                    continue;
                }
                return Err(ProxyError::Upstream(format!(
                    "Upstream returned {}",
                    status
                )));
            }
            Err(err) => {
                last_err = Some(format!("HTTP error: {}", err));
                continue;
            }
        }
    }

    Err(ProxyError::Upstream(
        last_err.unwrap_or_else(|| "All upstreams failed".to_string()),
    ))
}

fn is_retriable_status(status: u16) -> bool {
    matches!(status, 429 | 500..=599)
}

async fn handle_routed_non_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
    request_start: Instant,
    route: ResolvedRoute,
    requested_model: String,
) -> ProxyResult<Response> {
    let model = openai_req.model.clone();

    let mut urls_to_try = vec![(route.base_url.clone(), route.api_key.clone())];

    if route.allow_failover {
        let failovers = config.failover_upstreams(&requested_model, &route.upstream_name);
        for fo in &failovers {
            urls_to_try.push((fo.base_url.clone(), fo.api_key.clone()));
        }
    }

    let mut last_err = None;

    for (url, api_key) in &urls_to_try {
        tracing::debug!(
            "Sending non-streaming request to {} (model: {})",
            url,
            model
        );

        let mut req_builder = client
            .post(url)
            .json(&openai_req)
            .timeout(Duration::from_secs(300));

        if let Some(key) = api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let upstream_start = Instant::now();
        let response = match req_builder.send().await {
            Ok(resp) => {
                metrics::upstream_latency(
                    upstream_start.elapsed().as_secs_f64(),
                    "chat_completions",
                );
                resp
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                metrics::upstream_error("chat_completions");
                last_err = Some(ProxyError::Http(err));
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
            metrics::upstream_error("chat_completions");

            if is_retriable_status(status.as_u16()) {
                last_err = Some(ProxyError::Upstream(format!(
                    "Upstream returned {}: {}",
                    status, error_text
                )));
                continue;
            }
            return Err(ProxyError::Upstream(format!(
                "Upstream returned {}: {}",
                status, error_text
            )));
        }

        let openai_resp: openai::OpenAIResponse = response.json().await?;
        let prompt_tokens = openai_resp.usage.prompt_tokens;
        let completion_tokens = openai_resp.usage.completion_tokens;

        tracing::info!(
            model = model.as_str(),
            ttfb_ms = request_start.elapsed().as_millis(),
            "First token received"
        );
        metrics::tokens(prompt_tokens, completion_tokens, &model);

        if config.verbose {
            tracing::trace!(
                "Received OpenAI response: {}",
                serde_json::to_string_pretty(&openai_resp).unwrap_or_default()
            );
        }

        let anthropic_resp = pipeline::translate_response(openai_resp, &model)?;

        tracing::info!(
            model = model.as_str(),
            total_ms = request_start.elapsed().as_millis(),
            prompt_tokens = prompt_tokens,
            completion_tokens = completion_tokens,
            "Request completed"
        );

        return Ok(Json(anthropic_resp).into_response());
    }

    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("All upstreams failed".to_string())))
}

async fn handle_routed_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
    request_start: Instant,
    route: ResolvedRoute,
    requested_model: String,
) -> ProxyResult<Response> {
    let model = openai_req.model.clone();

    let mut urls_to_try = vec![(route.base_url.clone(), route.api_key.clone())];

    if route.allow_failover {
        let failovers = config.failover_upstreams(&requested_model, &route.upstream_name);
        for fo in &failovers {
            urls_to_try.push((fo.base_url.clone(), fo.api_key.clone()));
        }
    }

    let mut last_err = None;

    for (url, api_key) in &urls_to_try {
        tracing::debug!("Sending streaming request to {} (model: {})", url, model);

        let mut req_builder = client
            .post(url)
            .json(&openai_req)
            .timeout(Duration::from_secs(300));

        if let Some(key) = api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let upstream_start = Instant::now();
        let response = match req_builder.send().await {
            Ok(resp) => {
                metrics::upstream_latency(
                    upstream_start.elapsed().as_secs_f64(),
                    "chat_completions",
                );
                resp
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                metrics::upstream_error("chat_completions");
                last_err = Some(ProxyError::Http(err));
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
            metrics::upstream_error("chat_completions");

            if is_retriable_status(status.as_u16()) {
                last_err = Some(ProxyError::Upstream(format!(
                    "Upstream returned {}: {}",
                    status, error_text
                )));
                continue;
            }
            return Err(ProxyError::Upstream(format!(
                "Upstream returned {}: {}",
                status, error_text
            )));
        }

        let upstream = response.bytes_stream();
        let sse_stream = create_sse_stream(upstream, model.clone(), request_start);

        let mut headers = HeaderMap::new();
        headers.insert(
            "Content-Type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        headers.insert("Connection", HeaderValue::from_static("keep-alive"));

        return Ok((headers, Body::from_stream(sse_stream)).into_response());
    }

    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("All upstreams failed".to_string())))
}

async fn handle_legacy_non_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
    request_start: Instant,
) -> ProxyResult<Response> {
    let model = openai_req.model.clone();
    let urls = config.chat_completions_urls();
    let mut last_err = None;

    for url in &urls {
        tracing::debug!(
            "Sending non-streaming request to {} (model: {})",
            url,
            model
        );

        let mut req_builder = client
            .post(url)
            .json(&openai_req)
            .timeout(Duration::from_secs(300));

        if let Some(api_key) = &config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let upstream_start = Instant::now();
        let response = match req_builder.send().await {
            Ok(resp) => {
                metrics::upstream_latency(
                    upstream_start.elapsed().as_secs_f64(),
                    "chat_completions",
                );
                resp
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                metrics::upstream_error("chat_completions");
                last_err = Some(ProxyError::Http(err));
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
            metrics::upstream_error("chat_completions");

            if is_retriable_status(status.as_u16()) {
                last_err = Some(ProxyError::Upstream(format!(
                    "Upstream returned {}: {}",
                    status, error_text
                )));
                continue;
            }
            return Err(ProxyError::Upstream(format!(
                "Upstream returned {}: {}",
                status, error_text
            )));
        }

        let openai_resp: openai::OpenAIResponse = response.json().await?;

        let prompt_tokens = openai_resp.usage.prompt_tokens;
        let completion_tokens = openai_resp.usage.completion_tokens;

        tracing::info!(
            model = model.as_str(),
            ttfb_ms = request_start.elapsed().as_millis(),
            "First token received"
        );

        metrics::tokens(prompt_tokens, completion_tokens, &model);

        if config.verbose {
            tracing::trace!(
                "Received OpenAI response: {}",
                serde_json::to_string_pretty(&openai_resp).unwrap_or_default()
            );
        }

        let anthropic_resp = pipeline::translate_response(openai_resp, &model)?;

        tracing::info!(
            model = model.as_str(),
            total_ms = request_start.elapsed().as_millis(),
            prompt_tokens = prompt_tokens,
            completion_tokens = completion_tokens,
            "Request completed"
        );

        if config.verbose {
            tracing::trace!(
                "Transformed Anthropic response: {}",
                serde_json::to_string_pretty(&anthropic_resp).unwrap_or_default()
            );
        }

        return Ok(Json(anthropic_resp).into_response());
    }

    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("All upstreams failed".to_string())))
}

async fn handle_legacy_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
    request_start: Instant,
) -> ProxyResult<Response> {
    let model = openai_req.model.clone();
    let urls = config.chat_completions_urls();
    let mut last_err = None;

    for url in &urls {
        tracing::debug!("Sending streaming request to {} (model: {})", url, model);

        let mut req_builder = client
            .post(url)
            .json(&openai_req)
            .timeout(Duration::from_secs(300));

        if let Some(api_key) = &config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let upstream_start = Instant::now();
        let response = match req_builder.send().await {
            Ok(resp) => {
                metrics::upstream_latency(
                    upstream_start.elapsed().as_secs_f64(),
                    "chat_completions",
                );
                resp
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                metrics::upstream_error("chat_completions");
                last_err = Some(ProxyError::Http(err));
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
            metrics::upstream_error("chat_completions");

            if is_retriable_status(status.as_u16()) {
                last_err = Some(ProxyError::Upstream(format!(
                    "Upstream returned {}: {}",
                    status, error_text
                )));
                continue;
            }
            return Err(ProxyError::Upstream(format!(
                "Upstream returned {}: {}",
                status, error_text
            )));
        }

        let upstream = response.bytes_stream();
        let sse_stream = create_sse_stream(upstream, model.clone(), request_start);

        let mut headers = HeaderMap::new();
        headers.insert(
            "Content-Type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        headers.insert("Connection", HeaderValue::from_static("keep-alive"));

        return Ok((headers, Body::from_stream(sse_stream)).into_response());
    }

    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("All upstreams failed".to_string())))
}

fn serialize_event(event: &anthropic::StreamEvent) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event.event_type(),
        serde_json::to_string(event).unwrap_or_default()
    )
}

fn create_sse_stream(
    upstream: impl Stream<Item = Result<Bytes, impl std::fmt::Display + Send + 'static>>
        + Send
        + 'static,
    model: String,
    request_start: Instant,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut state = stream::initial_state(model.clone());
        let mut first_token_logged = false;

        tokio::pin!(upstream);

        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);

                    while let Some(pos) = buffer.find("\n\n") {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        if line.trim().is_empty() {
                            continue;
                        }

                        for l in line.lines() {
                            if let Some(data) = l.strip_prefix("data: ") {
                                if data.trim() == "[DONE]" {
                                    for event in stream::translate_done(&mut state) {
                                        yield Ok(Bytes::from(serialize_event(&event)));
                                    }
                                    tracing::info!(
                                        model = model.as_str(),
                                        total_ms = request_start.elapsed().as_millis(),
                                        "Stream completed"
                                    );
                                    continue;
                                }

                                if let Ok(chunk) = serde_json::from_str::<openai::StreamChunk>(data) {
                                    let has_content = chunk.choices.iter().any(|c| {
                                        c.delta.content.as_ref().is_some_and(|s| !s.is_empty())
                                            || c.delta.reasoning.as_ref().is_some_and(|s| !s.is_empty())
                                            || c.delta.tool_calls.as_ref().is_some_and(|t| !t.is_empty())
                                    });

                                    if has_content && !first_token_logged {
                                        tracing::info!(
                                            model = model.as_str(),
                                            ttfb_ms = request_start.elapsed().as_millis(),
                                            "First token received"
                                        );
                                        first_token_logged = true;
                                    }

                                    for event in stream::translate_chunk(&mut state, &chunk) {
                                        yield Ok(Bytes::from(serialize_event(&event)));
                                    }
                                } else {
                                    tracing::debug!("Ignoring unrecognized upstream stream chunk: {}", data);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    tracing::info!(
                        model = model.as_str(),
                        total_ms = request_start.elapsed().as_millis(),
                        "Stream ended with error"
                    );
                    for event in stream::translate_error(format!("Stream error: {}", e)) {
                        yield Ok(Bytes::from(serialize_event(&event)));
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::create_sse_stream;
    use bytes::Bytes;
    use futures::stream::{self, StreamExt};
    use serde_json::{json, Value};
    use std::fmt;
    use std::time::Instant;

    #[derive(Debug)]
    struct TestError;
    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test error")
        }
    }

    fn openai_chunk(
        id: &str,
        model: &str,
        content: Option<&str>,
        finish_reason: Option<&str>,
    ) -> String {
        let mut delta = json!({});
        if let Some(c) = content {
            delta["content"] = json!(c);
        }
        let mut choice = json!({ "index": 0, "delta": delta });
        if let Some(fr) = finish_reason {
            choice["finish_reason"] = json!(fr);
        }
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [choice],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_chunk_with_reasoning(id: &str, model: &str, reasoning: &str) -> String {
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning": reasoning } }],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_chunk_with_tool_call(
        id: &str,
        model: &str,
        tool_id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
        finish_reason: Option<&str>,
    ) -> String {
        let mut tc = json!({ "index": 0 });
        if let Some(tid) = tool_id {
            tc["id"] = json!(tid);
            tc["type"] = json!("function");
        }
        let mut func = json!({});
        if let Some(n) = name {
            func["name"] = json!(n);
        }
        if let Some(a) = args {
            func["arguments"] = json!(a);
        }
        if !func.as_object().unwrap().is_empty() {
            tc["function"] = func;
        }
        let mut choice = json!({ "index": 0, "delta": { "tool_calls": [tc] } });
        if let Some(fr) = finish_reason {
            choice["finish_reason"] = json!(fr);
        }
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [choice],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_done() -> String {
        "data: [DONE]\n\n".to_string()
    }

    fn make_stream(
        chunks: Vec<String>,
    ) -> impl futures::Stream<Item = Result<Bytes, TestError>> + Send + 'static {
        stream::iter(chunks.into_iter().map(|c| Ok(Bytes::from(c))))
    }

    async fn collect_events(chunks: Vec<String>, model: &str) -> Vec<Value> {
        let s = make_stream(chunks);
        let sse = create_sse_stream(s, model.to_string(), Instant::now());
        tokio::pin!(sse);

        let mut events = Vec::new();
        while let Some(Ok(bytes)) = sse.next().await {
            let text = String::from_utf8_lossy(&bytes);
            for segment in text.split("\n\n").filter(|s| !s.is_empty()) {
                if let Some(data_line) = segment.lines().find(|l| l.starts_with("data: ")) {
                    let json_str = data_line.strip_prefix("data: ").unwrap();
                    if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                        events.push(v);
                    }
                }
            }
        }
        events
    }

    #[tokio::test]
    async fn text_stream_produces_message_start_content_block_and_stop() {
        let chunks = vec![
            openai_chunk("chatcmpl-1", "gpt-4o", Some("Hello"), None),
            openai_chunk("chatcmpl-1", "gpt-4o", Some(" world"), None),
            openai_chunk("chatcmpl-1", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;

        assert_eq!(events[0]["type"], "message_start");
        assert_eq!(events[0]["message"]["id"], "chatcmpl-1");
        assert_eq!(events[0]["message"]["model"], "gpt-4o");
        assert_eq!(events[0]["message"]["role"], "assistant");

        assert_eq!(events[1]["type"], "content_block_start");
        assert_eq!(events[1]["content_block"]["type"], "text");

        assert_eq!(events[2]["type"], "content_block_delta");
        assert_eq!(events[2]["delta"]["type"], "text_delta");
        assert_eq!(events[2]["delta"]["text"], "Hello");

        assert_eq!(events[3]["type"], "content_block_delta");
        assert_eq!(events[3]["delta"]["text"], " world");

        assert_eq!(events[4]["type"], "content_block_stop");

        assert_eq!(events[5]["type"], "message_delta");
        assert_eq!(events[5]["delta"]["stop_reason"], "end_turn");

        assert_eq!(events[6]["type"], "message_stop");
    }

    #[tokio::test]
    async fn thinking_then_text_produces_two_content_blocks() {
        let chunks = vec![
            openai_chunk_with_reasoning("chatcmpl-2", "gpt-4o", "Let me think..."),
            openai_chunk_with_reasoning("chatcmpl-2", "gpt-4o", " more thinking"),
            openai_chunk("chatcmpl-2", "gpt-4o", Some("The answer is 42"), None),
            openai_chunk("chatcmpl-2", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;

        assert_eq!(events[0]["type"], "message_start");
        assert_eq!(events[1]["type"], "content_block_start");
        assert_eq!(events[1]["content_block"]["type"], "thinking");
        assert_eq!(events[1]["index"], 0);
        assert_eq!(events[4]["type"], "content_block_stop");
        assert_eq!(events[4]["index"], 0);
        assert_eq!(events[5]["type"], "content_block_start");
        assert_eq!(events[5]["content_block"]["type"], "text");
        assert_eq!(events[5]["index"], 1);
    }

    #[tokio::test]
    async fn tool_call_stream_produces_tool_use_block() {
        let chunks = vec![
            openai_chunk_with_tool_call(
                "chatcmpl-3",
                "gpt-4o",
                Some("call_abc"),
                Some("read_file"),
                None,
                None,
            ),
            openai_chunk_with_tool_call(
                "chatcmpl-3",
                "gpt-4o",
                None,
                None,
                Some("{\"path\":"),
                None,
            ),
            openai_chunk_with_tool_call(
                "chatcmpl-3",
                "gpt-4o",
                None,
                None,
                Some("\"/tmp\"}"),
                None,
            ),
            openai_chunk("chatcmpl-3", "gpt-4o", None, Some("tool_calls")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;
        assert_eq!(events[1]["content_block"]["type"], "tool_use");
        assert_eq!(events[1]["content_block"]["id"], "call_abc");
        assert_eq!(events[5]["delta"]["stop_reason"], "tool_use");
    }

    #[tokio::test]
    async fn done_without_finish_reason_still_produces_message_stop() {
        let chunks = vec![
            openai_chunk("chatcmpl-4", "gpt-4o", Some("hi"), None),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        assert_eq!(events.last().unwrap()["type"], "message_stop");
    }

    #[tokio::test]
    async fn fallback_model_used_when_upstream_omits_model() {
        let chunk = json!({
            "choices": [{ "index": 0, "delta": { "content": "hey" } }],
        });
        let chunks = vec![
            format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()),
            openai_chunk("id", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];
        let events = collect_events(chunks, "my-fallback-model").await;
        assert_eq!(events[0]["message"]["model"], "my-fallback-model");
    }

    #[tokio::test]
    async fn empty_content_chunks_are_not_emitted() {
        let chunks = vec![
            openai_chunk("chatcmpl-5", "gpt-4o", Some(""), None),
            openai_chunk("chatcmpl-5", "gpt-4o", Some("hello"), None),
            openai_chunk("chatcmpl-5", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        let text_deltas: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
            .collect();
        assert_eq!(text_deltas.len(), 1);
        assert_eq!(text_deltas[0]["delta"]["text"], "hello");
    }

    #[tokio::test]
    async fn stream_error_produces_error_event_and_stops() {
        let items: Vec<Result<Bytes, TestError>> = vec![
            Ok(Bytes::from(openai_chunk(
                "chatcmpl-6",
                "gpt-4o",
                Some("start"),
                None,
            ))),
            Err(TestError),
        ];
        let s = stream::iter(items);
        let sse = create_sse_stream(s, "fallback".to_string(), Instant::now());
        tokio::pin!(sse);

        let mut events = Vec::new();
        while let Some(Ok(bytes)) = sse.next().await {
            let text = String::from_utf8_lossy(&bytes);
            for segment in text.split("\n\n").filter(|s| !s.is_empty()) {
                if let Some(data_line) = segment.lines().find(|l| l.starts_with("data: ")) {
                    let json_str = data_line.strip_prefix("data: ").unwrap();
                    if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                        events.push(v);
                    }
                }
            }
        }
        let error_events: Vec<_> = events.iter().filter(|e| e["type"] == "error").collect();
        assert_eq!(error_events.len(), 1);
    }

    #[tokio::test]
    async fn chunked_delivery_handles_split_sse_frames() {
        let full_chunk = openai_chunk("chatcmpl-7", "gpt-4o", Some("split"), None);
        let mid = full_chunk.len() / 2;
        let part1 = full_chunk[..mid].to_string();
        let part2 = format!(
            "{}{}{}",
            &full_chunk[mid..],
            openai_chunk("chatcmpl-7", "gpt-4o", None, Some("stop")),
            openai_done()
        );
        let events = collect_events(vec![part1, part2], "fallback").await;
        let text_deltas: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
            .collect();
        assert_eq!(text_deltas.len(), 1);
        assert_eq!(text_deltas[0]["delta"]["text"], "split");
    }

    #[tokio::test]
    async fn text_then_tool_call_produces_two_blocks() {
        let chunks = vec![
            openai_chunk("chatcmpl-8", "gpt-4o", Some("I'll read that file."), None),
            openai_chunk_with_tool_call(
                "chatcmpl-8",
                "gpt-4o",
                Some("call_xyz"),
                Some("read_file"),
                None,
                None,
            ),
            openai_chunk_with_tool_call(
                "chatcmpl-8",
                "gpt-4o",
                None,
                None,
                Some("{\"path\":\"/etc\"}"),
                None,
            ),
            openai_chunk("chatcmpl-8", "gpt-4o", None, Some("tool_calls")),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        let block_starts: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "content_block_start")
            .collect();
        assert_eq!(block_starts.len(), 2);
        assert_eq!(block_starts[0]["content_block"]["type"], "text");
        assert_eq!(block_starts[1]["content_block"]["type"], "tool_use");
    }
}
