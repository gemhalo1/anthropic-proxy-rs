# Multi-Upstream Model Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add JSON config file support for per-model routing across multiple upstreams, with namespaced model IDs, configurable failover, and a controllable `/v1/models` endpoint.

**Architecture:** Introduce `UpstreamConfig`, `ModelConfig`, and `ModelsListMode` structs in `config.rs`. The `Config` struct gains an `upstreams` map (replacing the flat `upstream_urls` + `api_key` + `model_map` combo when a config file is present). A new `resolve_model()` method on `Config` returns a `ResolvedRoute` (upstream name, target model, base_url, api_key, allow_failover). `proxy.rs` uses `resolve_model()` to route requests to specific upstreams with per-upstream auth, and `list_models_handler` gains three modes via `models_list_mode`.

**Tech Stack:** Rust, Axum, serde/serde_json, reqwest, tracing

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/config.rs` | JSON config file loading, new structs (`UpstreamConfig`, `ModelConfig`, `ModelsListMode`, `ResolvedRoute`), `resolve_model()` routing method, env var coexistence merge logic |
| `src/cli.rs` | Add `--config-file` CLI arg |
| `src/main.rs` | Wire `--config-file` into config loading, pass resolved route info to proxy |
| `src/proxy.rs` | Use `resolve_model()` for per-upstream routing in `proxy_handler`, update `list_models_handler` for static/upstream/merge modes |
| `src/translate/pipeline.rs` | No changes needed — `TranslationPolicy` stays the same, model substitution happens in `proxy.rs` before calling `translate_request` |

---

### Task 1: Config structs and JSON file loading

**Files:**
- Modify: `src/config.rs`

- [ ] **Step 1: Write the failing test for config file loading**

Add to `src/config.rs` under `#[cfg(test)] mod tests`:

```rust
use std::io::Write;

#[test]
fn load_config_from_json_file() {
    let json = serde_json::json!({
        "port": 8080,
        "debug": true,
        "models_list_mode": "static",
        "upstreams": {
            "openai": {
                "base_url": "https://api.openai.com",
                "api_key": "sk-test123",
                "models": {
                    "gpt-4.1": {},
                    "o1-pro": { "target": "o1-pro" }
                }
            }
        }
    });

    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(serde_json::to_string_pretty(&json).unwrap().as_bytes()).unwrap();

    let config = Config::from_json_file(tmp.path()).unwrap();
    assert_eq!(config.port, 8080);
    assert!(config.debug);
    assert_eq!(config.models_list_mode, ModelsListMode::Static);
    assert_eq!(config.upstreams.len(), 1);

    let upstream = &config.upstreams["openai"];
    assert_eq!(upstream.base_url, "https://api.openai.com");
    assert_eq!(upstream.api_key.as_deref(), Some("sk-test123"));
    assert_eq!(upstream.models.len(), 2);

    let gpt_model = &upstream.models["gpt-4.1"];
    assert_eq!(gpt_model.target.as_deref(), Some("gpt-4.1"));
    assert!(!gpt_model.allow_failover);

    let o1_model = &upstream.models["o1-pro"];
    assert_eq!(o1_model.target.as_deref(), Some("o1-pro"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test load_config_from_json_file`
Expected: FAIL — `Config::from_json_file`, `ModelsListMode`, `UpstreamConfig`, `ModelConfig` not defined

- [ ] **Step 3: Add new structs to `src/config.rs`**

Add these structs before the `Config` struct definition:

```rust
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelsListMode {
    Static,
    Upstream,
    Merge,
}

impl Default for ModelsListMode {
    fn default() -> Self {
        ModelsListMode::Static
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamConfig {
    pub base_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    pub models: BTreeMap<String, ModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default)]
    pub allow_failover: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_model: Option<String>,
}
```

Add `upstreams` and `models_list_mode` fields to `Config`:

```rust
pub struct Config {
    pub port: u16,
    pub upstream_urls: Vec<String>,
    pub api_key: Option<String>,
    pub model_map: BTreeMap<String, String>,
    pub system_prompt_ignore_terms: Vec<String>,
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,
    pub merge_system_messages: bool,
    pub merge_user_messages: bool,
    pub debug: bool,
    pub verbose: bool,
    pub models_list_mode: ModelsListMode,
    pub upstreams: BTreeMap<String, UpstreamConfig>,
}
```

Update `Default` impl to include the new fields:

```rust
impl Default for Config {
    fn default() -> Self {
        Self {
            port: 3000,
            upstream_urls: vec!["http://localhost:11434".to_string()],
            api_key: None,
            model_map: BTreeMap::new(),
            system_prompt_ignore_terms: Vec::new(),
            reasoning_model: None,
            completion_model: None,
            merge_system_messages: false,
            merge_user_messages: false,
            debug: false,
            verbose: false,
            models_list_mode: ModelsListMode::Static,
            upstreams: BTreeMap::new(),
        }
    }
}
```

Add `from_json_file` method and `FileConfig` deserialization struct:

```rust
#[derive(Debug, Deserialize)]
struct FileConfig {
    port: Option<u16>,
    debug: Option<bool>,
    verbose: Option<bool>,
    merge_system_messages: Option<bool>,
    merge_user_messages: Option<bool>,
    system_prompt_ignore_terms: Option<Vec<String>>,
    reasoning_model: Option<String>,
    completion_model: Option<String>,
    models_list_mode: Option<ModelsListMode>,
    upstreams: BTreeMap<String, UpstreamConfig>,
}

impl Config {
    pub fn from_json_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let file_config: FileConfig = serde_json::from_str(&content)?;

        let mut config = Config::default();

        config.port = file_config.port.unwrap_or(3000);
        config.debug = file_config.debug.unwrap_or(false);
        config.verbose = file_config.verbose.unwrap_or(false);
        config.merge_system_messages = file_config.merge_system_messages.unwrap_or(false);
        config.merge_user_messages = file_config.merge_user_messages.unwrap_or(false);
        config.reasoning_model = file_config.reasoning_model;
        config.completion_model = file_config.completion_model;
        config.models_list_mode = file_config.models_list_mode.unwrap_or_default();
        config.upstreams = file_config.upstreams;

        if let Some(terms) = file_config.system_prompt_ignore_terms {
            config.system_prompt_ignore_terms = terms;
            Self::dedupe_ignore_terms(&mut config.system_prompt_ignore_terms);
        }

        // Validate upstream URLs
        for (name, upstream) in &config.upstreams {
            Self::resolve_chat_completions_url(&upstream.base_url)
                .map_err(|e| anyhow::anyhow!("Upstream '{}': {}", name, e))?;
        }

        // Set upstream_urls from config file upstreams (for legacy compatibility)
        config.upstream_urls = config
            .upstreams
            .values()
            .map(|u| u.base_url.clone())
            .collect();

        // Use first upstream's api_key as global if no global key set
        if config.api_key.is_none() {
            config.api_key = config
                .upstreams
                .values()
                .next()
                .and_then(|u| u.api_key.clone());
        }

        // Build model_map from upstream model configs where target differs from key
        for (upstream_name, upstream) in &config.upstreams {
            for (bare_name, model_conf) in &upstream.models {
                let namespaced_id = format!("{}/{}", upstream_name, bare_name);
                let target = model_conf.target.as_deref().unwrap_or(bare_name);
                if target != bare_name {
                    config.model_map.insert(namespaced_id, target.to_string());
                }
            }
        }

        Ok(config)
    }
}
```

Add `tempfile` to `Cargo.toml` dev-dependencies:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test load_config_from_json_file`
Expected: PASS

- [ ] **Step 5: Write tests for resolve_model**

Add to `#[cfg(test)] mod tests`:

```rust
#[test]
fn resolve_model_prefix_match() {
    let config = Config {
        upstreams: BTreeMap::from([
            ("openai".to_string(), UpstreamConfig {
                base_url: "https://api.openai.com".to_string(),
                api_key: Some("sk-test".to_string()),
                models: BTreeMap::from([
                    ("gpt-4.1".to_string(), ModelConfig {
                        target: Some("gpt-4.1".to_string()),
                        allow_failover: false,
                        reasoning_model: None,
                        completion_model: None,
                    }),
                ]),
            }),
        ]),
        ..Default::default()
    };

    let route = config.resolve_model("openai/gpt-4.1").unwrap();
    assert_eq!(route.upstream_name, "openai");
    assert_eq!(route.target, "gpt-4.1");
    assert_eq!(route.base_url, "https://api.openai.com/v1/chat/completions");
    assert_eq!(route.api_key.as_deref(), Some("sk-test"));
    assert!(!route.allow_failover);
}

#[test]
fn resolve_model_bare_name_match() {
    let config = Config {
        upstreams: BTreeMap::from([
            ("openai".to_string(), UpstreamConfig {
                base_url: "https://api.openai.com".to_string(),
                api_key: None,
                models: BTreeMap::from([
                    ("gpt-4.1".to_string(), ModelConfig {
                        target: Some("gpt-4.1".to_string()),
                        allow_failover: false,
                        reasoning_model: None,
                        completion_model: None,
                    }),
                ]),
            }),
        ]),
        ..Default::default()
    };

    let route = config.resolve_model("gpt-4.1").unwrap();
    assert_eq!(route.upstream_name, "openai");
    assert_eq!(route.target, "gpt-4.1");
}

#[test]
fn resolve_model_no_match_returns_legacy() {
    let config = Config {
        upstream_urls: vec!["https://api.openai.com".to_string()],
        api_key: Some("sk-global".to_string()),
        ..Default::default()
    };

    let route = config.resolve_model("unknown-model").unwrap();
    assert!(route.is_legacy);
    assert_eq!(route.target, "unknown-model");
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test resolve_model`
Expected: FAIL — `resolve_model` and `ResolvedRoute` not defined

- [ ] **Step 7: Implement `ResolvedRoute` and `resolve_model()`**

Add `ResolvedRoute` struct:

```rust
#[derive(Debug, Clone)]
pub struct ResolvedRoute {
    pub upstream_name: String,
    pub target: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub allow_failover: bool,
    pub reasoning_model: Option<String>,
    pub completion_model: Option<String>,
    pub is_legacy: bool,
    pub legacy_urls: Vec<String>,
    pub legacy_api_key: Option<String>,
}
```

Add `resolve_model` method to `Config`:

```rust
pub fn resolve_model(&self, model: &str) -> Result<ResolvedRoute> {
    // Step 1: Prefix match — split on first /
    if let Some((prefix, remainder)) = model.split_once('/') {
        if let Some(upstream) = self.upstreams.get(prefix) {
            if let Some(model_conf) = upstream.models.get(remainder) {
                let target = model_conf.target.as_deref().unwrap_or(remainder);
                let chat_url = Self::resolve_chat_completions_url(&upstream.base_url)?;
                let api_key = upstream.api_key.clone().or_else(|| self.api_key.clone());

                return Ok(ResolvedRoute {
                    upstream_name: prefix.to_string(),
                    target: target.to_string(),
                    base_url: chat_url,
                    api_key,
                    allow_failover: model_conf.allow_failover,
                    reasoning_model: model_conf.reasoning_model.clone(),
                    completion_model: model_conf.completion_model.clone(),
                    is_legacy: false,
                    legacy_urls: Vec::new(),
                    legacy_api_key: None,
                });
            }
        }
    }

    // Step 2: Bare name match — iterate upstreams in alphabetical order
    for (upstream_name, upstream) in &self.upstreams {
        if let Some(model_conf) = upstream.models.get(model) {
            let target = model_conf.target.as_deref().unwrap_or(model);
            let chat_url = Self::resolve_chat_completions_url(&upstream.base_url)?;
            let api_key = upstream.api_key.clone().or_else(|| self.api_key.clone());

            return Ok(ResolvedRoute {
                upstream_name: upstream_name.clone(),
                target: target.to_string(),
                base_url: chat_url,
                api_key,
                allow_failover: model_conf.allow_failover,
                reasoning_model: model_conf.reasoning_model.clone(),
                completion_model: model_conf.completion_model.clone(),
                is_legacy: false,
                legacy_urls: Vec::new(),
                legacy_api_key: None,
            });
        }
    }

    // Step 3: No match found — fall back to legacy behavior
    Ok(ResolvedRoute {
        upstream_name: String::new(),
        target: model.to_string(),
        base_url: String::new(),
        api_key: None,
        allow_failover: false,
        reasoning_model: None,
        completion_model: None,
        is_legacy: true,
        legacy_urls: self.chat_completions_urls(),
        legacy_api_key: self.api_key.clone(),
    })
}
```

Add `failover_upstreams` method:

```rust
pub fn failover_upstreams(&self, model: &str, exclude_upstream: &str) -> Vec<ResolvedRoute> {
    let mut alternatives = Vec::new();

    let original_target = self.resolve_model(model)
        .ok()
        .map(|r| r.target)
        .unwrap_or(model.to_string());

    for (upstream_name, upstream) in &self.upstreams {
        if upstream_name == exclude_upstream {
            continue;
        }

        for (bare_name, model_conf) in &upstream.models {
            let target = model_conf.target.as_deref().unwrap_or(bare_name);
            if bare_name == model || target == original_target {
                if let Ok(chat_url) = Self::resolve_chat_completions_url(&upstream.base_url) {
                    let api_key = upstream.api_key.clone().or_else(|| self.api_key.clone());
                    alternatives.push(ResolvedRoute {
                        upstream_name: upstream_name.clone(),
                        target: target.to_string(),
                        base_url: chat_url,
                        api_key,
                        allow_failover: model_conf.allow_failover,
                        reasoning_model: model_conf.reasoning_model.clone(),
                        completion_model: model_conf.completion_model.clone(),
                        is_legacy: false,
                        legacy_urls: Vec::new(),
                        legacy_api_key: None,
                    });
                }
                break;
            }
        }
    }

    alternatives
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cargo test resolve_model`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/config.rs Cargo.toml
git commit -m "feat: add config file structs, JSON loading, and model routing"
```

---

### Task 2: CLI arg for config file path

**Files:**
- Modify: `src/cli.rs`

- [ ] **Step 1: Add `--config-file` arg to `Cli` struct**

In `src/cli.rs`, add after the `pub config: Option<PathBuf>` field:

```rust
/// Path to JSON config file (overrides default search path)
#[arg(long = "config-file", value_name = "FILE")]
pub config_file: Option<PathBuf>,
```

- [ ] **Step 2: Run `cargo clippy` and `cargo test` to verify no breakage**

Run: `cargo clippy && cargo test`
Expected: PASS (new field is optional, no existing code breaks)

- [ ] **Step 3: Commit**

```bash
git add src/cli.rs
git commit -m "feat: add --config-file CLI arg"
```

---

### Task 3: Config file search and loading in main.rs

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Add `find_config_file` function**

```rust
fn find_config_file(cli: &Cli) -> Option<std::path::PathBuf> {
    if let Some(path) = &cli.config_file {
        if path.exists() {
            return Some(path.clone());
        }
        eprintln!("⚠️  WARNING: Config file not found: {}", path.display());
        return None;
    }

    let cwd_config = std::path::PathBuf::from("anthropic-proxy.json");
    if cwd_config.exists() {
        return Some(cwd_config);
    }

    if let Ok(home) = std::env::var("HOME") {
        let home_config = std::path::PathBuf::from(home).join(".anthropic-proxy.json");
        if home_config.exists() {
            return Some(home_config);
        }
    }

    let etc_config = std::path::PathBuf::from("/etc/anthropic-proxy/config.json");
    if etc_config.exists() {
        return Some(etc_config);
    }

    None
}
```

- [ ] **Step 2: Add `merge_env_overrides` function**

```rust
fn merge_env_overrides(config: &mut Config, env: Config) {
    if env.port != 3000 {
        config.port = env.port;
    }
    if env.debug {
        config.debug = true;
    }
    if env.verbose {
        config.verbose = true;
    }
    if env.api_key.is_some() {
        config.api_key = env.api_key;
    }
    if env.reasoning_model.is_some() {
        config.reasoning_model = env.reasoning_model;
    }
    if env.completion_model.is_some() {
        config.completion_model = env.completion_model;
    }
    if !env.system_prompt_ignore_terms.is_empty() {
        config.system_prompt_ignore_terms.extend(env.system_prompt_ignore_terms);
        Config::dedupe_ignore_terms(&mut config.system_prompt_ignore_terms);
    }

    // UPSTREAM_BASE_URL env var overrides config file upstreams entirely
    if !env.upstream_urls.is_empty() && env.upstream_urls != vec!["http://localhost:11434".to_string()] {
        config.upstream_urls = env.upstream_urls.clone();
        config.api_key = env.api_key.clone();
        config.upstreams.clear();
        for url in &env.upstream_urls {
            let name = format!("upstream_{}", url.replace('.', "_").replace('/', "_").replace(':', ""));
            config.upstreams.insert(name, UpstreamConfig {
                base_url: url.clone(),
                api_key: env.api_key.clone(),
                models: BTreeMap::new(),
            });
        }
    }

    // ANTHROPIC_PROXY_MODEL_MAP is additive
    if !env.model_map.is_empty() {
        config.model_map.extend(env.model_map);
    }
}
```

- [ ] **Step 3: Update `async_main` config loading**

Replace the `Config::from_env_with_path(cli.config)?` call with:

```rust
let mut config = if let Some(config_path) = find_config_file(&cli) {
    eprintln!("📄 Loaded config from: {}", config_path.display());
    let mut config = Config::from_json_file(&config_path)?;
    let env_config = Config::from_env_with_path(cli.config)?;
    merge_env_overrides(&mut config, env_config);
    config
} else {
    eprintln!("ℹ️  No config file found, using environment variables");
    Config::from_env_with_path(cli.config)?
};
```

- [ ] **Step 4: Add startup logging for upstreams config**

Add after existing tracing info block:

```rust
if !config.upstreams.is_empty() {
    tracing::info!("Configured upstreams: {}", config.upstreams.keys().collect::<Vec<_>>().join(", "));
    let total_models = config.upstreams.values().map(|u| u.models.len()).sum::<usize>();
    tracing::info!("Configured models: {} across {} upstreams", total_models, config.upstreams.len());
}
```

- [ ] **Step 5: Run `cargo clippy` and `cargo test`**

Run: `cargo clippy && cargo test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat: wire config file loading and env var merge into main"
```

---

### Task 4: Per-upstream routing in proxy_handler

**Files:**
- Modify: `src/proxy.rs`

- [ ] **Step 1: Rewrite `proxy_handler` to use `resolve_model()`**

```rust
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

    let route = config.resolve_model(&requested_model)?;

    let policy = TranslationPolicy {
        reasoning_model: route.reasoning_model.clone()
            .or_else(|| config.reasoning_model.clone()),
        completion_model: route.completion_model.clone()
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
        handle_routed_streaming(config, client, openai_req, start, route).await
    } else {
        handle_routed_non_streaming(config, client, openai_req, start, route).await
    };

    let status = match &result {
        Ok(resp) => resp.status().as_u16(),
        Err(_) => 500,
    };
    metrics::request_finished(start, status, is_streaming);

    result
}
```

- [ ] **Step 2: Add `handle_routed_non_streaming`**

```rust
async fn handle_routed_non_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
    request_start: Instant,
    route: ResolvedRoute,
) -> ProxyResult<Response> {
    let model = openai_req.model.clone();

    let mut urls_to_try = vec![(route.base_url.clone(), route.api_key.clone())];

    if route.allow_failover {
        let failovers = config.failover_upstreams(&model, &route.upstream_name);
        for fo in &failovers {
            urls_to_try.push((fo.base_url.clone(), fo.api_key.clone()));
        }
    }

    let mut last_err = None;

    for (url, api_key) in &urls_to_try {
        tracing::debug!("Sending non-streaming request to {} (model: {})", url, model);

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
                metrics::upstream_latency(upstream_start.elapsed().as_secs_f64(), "chat_completions");
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
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
            metrics::upstream_error("chat_completions");

            if is_retriable_status(status.as_u16()) {
                last_err = Some(ProxyError::Upstream(format!("Upstream returned {}: {}", status, error_text)));
                continue;
            }
            return Err(ProxyError::Upstream(format!("Upstream returned {}: {}", status, error_text)));
        }

        let openai_resp: openai::OpenAIResponse = response.json().await?;
        let prompt_tokens = openai_resp.usage.prompt_tokens;
        let completion_tokens = openai_resp.usage.completion_tokens;

        tracing::info!(model = model.as_str(), ttfb_ms = request_start.elapsed().as_millis(), "First token received");
        metrics::tokens(prompt_tokens, completion_tokens, &model);

        if config.verbose {
            tracing::trace!("Received OpenAI response: {}", serde_json::to_string_pretty(&openai_resp).unwrap_or_default());
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
```

- [ ] **Step 3: Add `handle_routed_streaming`**

```rust
async fn handle_routed_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
    request_start: Instant,
    route: ResolvedRoute,
) -> ProxyResult<Response> {
    let model = openai_req.model.clone();

    let mut urls_to_try = vec![(route.base_url.clone(), route.api_key.clone())];

    if route.allow_failover {
        let failovers = config.failover_upstreams(&model, &route.upstream_name);
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
                metrics::upstream_latency(upstream_start.elapsed().as_secs_f64(), "chat_completions");
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
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
            metrics::upstream_error("chat_completions");

            if is_retriable_status(status.as_u16()) {
                last_err = Some(ProxyError::Upstream(format!("Upstream returned {}: {}", status, error_text)));
                continue;
            }
            return Err(ProxyError::Upstream(format!("Upstream returned {}: {}", status, error_text)));
        }

        let upstream = response.bytes_stream();
        let sse_stream = create_sse_stream(upstream, model.clone(), request_start);

        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        headers.insert("Connection", HeaderValue::from_static("keep-alive"));

        return Ok((headers, Body::from_stream(sse_stream)).into_response());
    }

    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("All upstreams failed".to_string())))
}
```

- [ ] **Step 4: Rename existing handlers to legacy variants**

Rename `handle_non_streaming` → `handle_legacy_non_streaming` and `handle_streaming` → `handle_legacy_streaming`. Bodies unchanged. Update the `translation_policy()` helper to stay unchanged.

- [ ] **Step 5: Add `use crate::config::ResolvedRoute;` to imports**

```rust
use crate::config::{Config, ResolvedRoute};
```

- [ ] **Step 6: Run `cargo clippy` and `cargo test`**

Run: `cargo clippy && cargo test`
Expected: PASS (all existing tests still work via legacy mode)

- [ ] **Step 7: Commit**

```bash
git add src/proxy.rs
git commit -m "feat: per-upstream routing in proxy_handler"
```

---

### Task 5: Models list endpoint with three modes

**Files:**
- Modify: `src/config.rs` (add `static_models_list` method)
- Modify: `src/proxy.rs`

- [ ] **Step 1: Add `static_models_list` method to `Config`**

```rust
pub fn static_models_list(&self) -> Vec<(String, String)> {
    let mut models = Vec::new();

    for (upstream_name, upstream) in &self.upstreams {
        for (bare_name, model_conf) in &upstream.models {
            let namespaced_id = format!("{}/{}", upstream_name, bare_name);
            let display_name = model_conf.target.as_deref().unwrap_or(bare_name);
            models.push((namespaced_id, display_name.to_string()));
        }
    }

    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}
```

- [ ] **Step 2: Rewrite `list_models_handler` for three modes**

```rust
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
            }).into_response())
        }
        ModelsListMode::Upstream => {
            list_models_from_upstream(&config, &client).await
        }
        ModelsListMode::Merge => {
            let mut config_models = config.static_models_list();
            let config_ids: std::collections::HashSet<String> = config_models.iter().map(|(id, _)| id.clone()).collect();
            let config_bare_targets: std::collections::HashSet<String> = config_models.iter().map(|(_, d)| d.clone()).collect();

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
            }).into_response())
        }
    }
}
```

Add helper functions:

```rust
async fn list_models_from_upstream(
    config: &Config,
    client: &Client,
) -> ProxyResult<Response> {
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
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
                if is_retriable_status(status.as_u16()) {
                    last_err = Some(format!("Upstream returned {}: {}", status, error_text));
                    continue;
                }
                return Err(ProxyError::Upstream(format!("Upstream returned {}: {}", status, error_text)));
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                last_err = Some(format!("HTTP error: {}", err));
                continue;
            }
        }
    }

    Err(ProxyError::Upstream(last_err.unwrap_or_else(|| "All upstreams failed".to_string())))
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
                return Err(ProxyError::Upstream(format!("Upstream returned {}", status)));
            }
            Err(err) => {
                last_err = Some(format!("HTTP error: {}", err));
                continue;
            }
        }
    }

    Err(ProxyError::Upstream(last_err.unwrap_or_else(|| "All upstreams failed".to_string())))
}
```

- [ ] **Step 3: Write test for static models list**

Add to `src/config.rs` tests:

```rust
#[test]
fn static_models_list_from_config() {
    let config = Config {
        upstreams: BTreeMap::from([
            ("openai".to_string(), UpstreamConfig {
                base_url: "https://api.openai.com".to_string(),
                api_key: None,
                models: BTreeMap::from([
                    ("gpt-4.1".to_string(), ModelConfig {
                        target: Some("gpt-4.1".to_string()),
                        allow_failover: false,
                        reasoning_model: None,
                        completion_model: None,
                    }),
                ]),
            }),
            ("openrouter".to_string(), UpstreamConfig {
                base_url: "https://openrouter.ai/api".to_string(),
                api_key: None,
                models: BTreeMap::from([
                    ("anthropic/claude-opus-4-7".to_string(), ModelConfig {
                        target: Some("anthropic/claude-opus-4-7".to_string()),
                        allow_failover: false,
                        reasoning_model: None,
                        completion_model: None,
                    }),
                ]),
            }),
        ]),
        models_list_mode: ModelsListMode::Static,
        ..Default::default()
    };

    let models = config.static_models_list();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].0, "openai/gpt-4.1");
    assert_eq!(models[0].1, "gpt-4.1");
    assert_eq!(models[1].0, "openrouter/anthropic/claude-opus-4-7");
}
```

- [ ] **Step 4: Run `cargo clippy` and `cargo test`**

Run: `cargo clippy && cargo test`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.rs src/proxy.rs
git commit -m "feat: models list endpoint with static/upstream/merge modes"
```

---

### Task 6: Integration test and final cleanup

**Files:**
- Modify: `src/config.rs` (add integration test)

- [ ] **Step 1: Write integration test**

Add to `src/config.rs` tests:

```rust
#[test]
fn full_config_json_round_trip() {
    let json = serde_json::json!({
        "port": 8080,
        "debug": true,
        "models_list_mode": "static",
        "merge_system_messages": true,
        "upstreams": {
            "openai": {
                "base_url": "https://api.openai.com",
                "api_key": "sk-test",
                "models": {
                    "gpt-4.1": { "allow_failover": true },
                    "o1-pro": { "target": "o1-pro" }
                }
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "sk-or",
                "models": {
                    "anthropic/claude-opus-4-7": { "target": "anthropic/claude-opus-4-7" },
                    "openai/gpt-4.1": { "target": "openai/gpt-4.1", "allow_failover": true }
                }
            }
        }
    });

    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    write!(tmp, "{}", serde_json::to_string_pretty(&json).unwrap()).unwrap();

    let config = Config::from_json_file(tmp.path()).unwrap();

    let route = config.resolve_model("openai/gpt-4.1").unwrap();
    assert_eq!(route.upstream_name, "openai");
    assert_eq!(route.target, "gpt-4.1");
    assert_eq!(route.base_url, "https://api.openai.com/v1/chat/completions");
    assert!(route.allow_failover);

    let route = config.resolve_model("o1-pro").unwrap();
    assert_eq!(route.upstream_name, "openai");
    assert_eq!(route.target, "o1-pro");

    let route = config.resolve_model("unknown").unwrap();
    assert!(route.is_legacy);

    let models = config.static_models_list();
    assert_eq!(models.len(), 4);
}
```

- [ ] **Step 2: Run all tests**

Run: `cargo test`
Expected: PASS

- [ ] **Step 3: Run `cargo clippy`**

Run: `cargo clippy`
Expected: No warnings

- [ ] **Step 4: Commit**

```bash
git add src/config.rs
git commit -m "test: add integration test for config file loading and routing"
```

---

## Self-Review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| JSON config file format | Task 1 |
| Config file search order | Task 3 |
| `--config-file` CLI arg | Task 2 |
| `UpstreamConfig`, `ModelConfig`, `ModelsListMode` structs | Task 1 |
| Bare model keys (no upstream prefix) | Task 1 |
| `target` defaults to bare key | Task 1 |
| `allow_failover` per-model | Task 1, 4 |
| `reasoning_model` / `completion_model` per-model | Task 1, 4 |
| Prefix match routing | Task 1 |
| Bare name match routing | Task 1 |
| No match → legacy fallback | Task 1 |
| Failover with `allow_failover` | Task 4 |
| Model name substitution | Task 4 |
| Per-upstream base_url + api_key | Task 4 |
| `/v1/models` static mode | Task 5 |
| `/v1/models` upstream mode | Task 5 |
| `/v1/models` merge mode with dedup | Task 5 |
| Env var coexistence (env overrides) | Task 3 |
| `UPSTREAM_BASE_URL` overrides upstreams | Task 3 |

All spec requirements covered.

**2. Placeholder scan:** No TBD, TODO, "implement later", or vague steps found.

**3. Type consistency:** `ResolvedRoute` defined in Task 1, used in Tasks 4-5. `ModelsListMode` enum in Task 1, used in Task 5. `UpstreamConfig`/`ModelConfig` in Task 1, referenced consistently. `static_models_list` returns `Vec<(String, String)>` — consistent between config.rs and proxy.rs.