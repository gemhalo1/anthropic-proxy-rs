use anyhow::{bail, Result};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    env,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelsListMode {
    #[default]
    Static,
    Upstream,
    Merge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamConfig {
    pub base_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    pub models: BTreeMap<String, ModelConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone)]
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

impl Config {
    fn load_dotenv(custom_path: Option<PathBuf>) -> Option<PathBuf> {
        if let Some(path) = custom_path {
            if path.exists() && dotenvy::from_path(&path).is_ok() {
                return Some(path);
            }
            eprintln!(
                "⚠️  WARNING: Custom config file not found: {}",
                path.display()
            );
        }

        if let Ok(path) = dotenvy::dotenv() {
            return Some(path);
        }

        if let Ok(home) = env::var("HOME") {
            let home_config = PathBuf::from(home).join(".anthropic-proxy.env");
            if home_config.exists() && dotenvy::from_path(&home_config).is_ok() {
                return Some(home_config);
            }
        }

        let etc_config = PathBuf::from("/etc/anthropic-proxy/.env");
        if etc_config.exists() && dotenvy::from_path(&etc_config).is_ok() {
            return Some(etc_config);
        }

        None
    }

    #[allow(dead_code)]
    pub fn from_env() -> Result<Self> {
        Self::from_env_with_path(None)
    }

    pub fn from_env_with_path(custom_path: Option<PathBuf>) -> Result<Self> {
        if let Some(path) = Self::load_dotenv(custom_path) {
            eprintln!("📄 Loaded config from: {}", path.display());
        } else {
            eprintln!("ℹ️  No .env file found, using environment variables only");
        }

        let port = env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(3000);

        let raw_urls = env::var("UPSTREAM_BASE_URL")
            .or_else(|_| env::var("ANTHROPIC_PROXY_BASE_URL"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "UPSTREAM_BASE_URL is required. Set it to your OpenAI-compatible endpoint.\n\
                Examples:\n\
                  - OpenRouter: https://openrouter.ai/api\n\
                  - OpenAI: https://api.openai.com\n\
                  - Multiple (failover): https://openrouter.ai/api;https://api.openai.com\n\
                  - Local: http://localhost:11434"
                )
            })?;

        let upstream_urls = Self::parse_upstream_urls(&raw_urls)?;

        let api_key = env::var("UPSTREAM_API_KEY")
            .or_else(|_| env::var("OPENROUTER_API_KEY"))
            .ok()
            .filter(|k| !k.is_empty());

        let model_map = env::var("ANTHROPIC_PROXY_MODEL_MAP")
            .ok()
            .map(|value| Self::parse_model_map(&value))
            .transpose()?
            .unwrap_or_default();

        let mut system_prompt_ignore_terms = env::var("ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS")
            .ok()
            .map(|value| Self::parse_system_prompt_ignore_terms(&value))
            .unwrap_or_default();
        Self::dedupe_ignore_terms(&mut system_prompt_ignore_terms);

        let reasoning_model = env::var("REASONING_MODEL").ok();
        let completion_model = env::var("COMPLETION_MODEL").ok();

        let debug = env::var("DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let verbose = env::var("VERBOSE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        Ok(Config {
            port,
            upstream_urls,
            api_key,
            model_map,
            system_prompt_ignore_terms,
            reasoning_model,
            completion_model,
            merge_system_messages: false,
            merge_user_messages: false,
            debug,
            verbose,
            models_list_mode: ModelsListMode::Static,
            upstreams: BTreeMap::new(),
        })
    }

    pub fn chat_completions_urls(&self) -> Vec<String> {
        self.upstream_urls
            .iter()
            .map(|url| {
                Self::resolve_chat_completions_url(url)
                    .expect("URLs should be validated during configuration loading")
            })
            .collect()
    }

    #[allow(dead_code)]
    pub fn models_urls(&self) -> Vec<String> {
        self.upstream_urls
            .iter()
            .map(|url| {
                Self::resolve_models_url(url)
                    .expect("URLs should be validated during configuration loading")
            })
            .collect()
    }

    pub fn parse_upstream_urls(raw: &str) -> Result<Vec<String>> {
        let urls: Vec<String> = raw
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect();

        if urls.is_empty() {
            bail!("UPSTREAM_BASE_URL must not be empty");
        }

        for url in &urls {
            Self::resolve_chat_completions_url(url)?;
        }

        Ok(urls)
    }

    fn resolve_chat_completions_url(base_url: &str) -> Result<String> {
        let (normalized, path_segments) = Self::parse_base_url(base_url)?;

        if Self::is_chat_completions_path(&path_segments) {
            return Ok(normalized.to_string());
        }

        let last_segment = path_segments.last().map(String::as_str);
        if matches!(last_segment, Some("chat") | Some("completions")) {
            bail!(
                "UPSTREAM_BASE_URL must be either a service base URL, a versioned base URL like https://gateway.example.com/v2, or the full .../chat/completions endpoint"
            );
        }

        if last_segment.is_some_and(Self::is_version_segment) {
            return Ok(format!("{}/chat/completions", normalized));
        }

        Ok(format!("{}/v1/chat/completions", normalized))
    }

    pub fn resolve_models_url(base_url: &str) -> Result<String> {
        let (normalized, path_segments) = Self::parse_base_url(base_url)?;

        if Self::is_chat_completions_path(&path_segments) {
            let base = normalized
                .trim_end_matches("/chat/completions")
                .trim_end_matches('/');
            return Ok(format!("{}/models", base));
        }

        let last_segment = path_segments.last().map(String::as_str);
        if matches!(last_segment, Some("chat") | Some("completions")) {
            bail!(
                "UPSTREAM_BASE_URL must be either a service base URL, a versioned base URL like https://gateway.example.com/v2, or the full .../chat/completions endpoint"
            );
        }

        if last_segment.is_some_and(Self::is_version_segment) {
            return Ok(format!("{}/models", normalized));
        }

        Ok(format!("{}/v1/models", normalized))
    }

    fn parse_base_url(base_url: &str) -> Result<(String, Vec<String>)> {
        let normalized = base_url.trim();

        if normalized.is_empty() {
            bail!("UPSTREAM_BASE_URL must not be empty");
        }

        let parsed = Url::parse(normalized).map_err(|err| {
            anyhow::anyhow!("UPSTREAM_BASE_URL must be a valid http(s) URL: {}", err)
        })?;

        if !matches!(parsed.scheme(), "http" | "https") {
            bail!("UPSTREAM_BASE_URL must use http or https");
        }

        if parsed.query().is_some() || parsed.fragment().is_some() {
            bail!("UPSTREAM_BASE_URL must not include query parameters or fragments");
        }

        let path_segments: Vec<_> = parsed
            .path_segments()
            .map(|segments| {
                segments
                    .filter(|segment| !segment.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();

        Ok((normalized.trim_end_matches('/').to_string(), path_segments))
    }

    fn is_chat_completions_path(segments: &[String]) -> bool {
        matches!(segments, [.., chat, completions] if chat == "chat" && completions == "completions")
    }

    fn is_version_segment(segment: &str) -> bool {
        let version = segment
            .strip_prefix('v')
            .or_else(|| segment.strip_prefix('V'));

        version
            .is_some_and(|value| !value.is_empty() && value.chars().all(|ch| ch.is_ascii_digit()))
    }

    pub fn parse_system_prompt_ignore_terms(value: &str) -> Vec<String> {
        value
            .split([';', '\n'])
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    }

    pub fn dedupe_ignore_terms(terms: &mut Vec<String>) {
        let mut deduped = Vec::new();
        let mut seen = Vec::new();
        for term in terms.drain(..) {
            let normalized = term
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_ascii_lowercase();
            if !seen.iter().any(|existing: &String| existing == &normalized) {
                seen.push(normalized);
                deduped.push(term);
            }
        }
        *terms = deduped;
    }

    pub fn parse_model_map(value: &str) -> Result<BTreeMap<String, String>> {
        let mut model_map = BTreeMap::new();

        for entry in value
            .split([';', '\n'])
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
        {
            let (source, target) = entry.split_once('=').ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid ANTHROPIC_PROXY_MODEL_MAP entry '{}'. Expected source=target",
                    entry
                )
            })?;

            let source = source.trim();
            let target = target.trim();

            if source.is_empty() || target.is_empty() {
                bail!(
                    "Invalid ANTHROPIC_PROXY_MODEL_MAP entry '{}'. Source and target models must be non-empty",
                    entry
                );
            }

            model_map.insert(source.to_string(), target.to_string());
        }

        Ok(model_map)
    }
}

#[allow(dead_code)]
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

#[allow(dead_code)]
impl Config {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let file_config: FileConfig = serde_json::from_str(&content)?;

        let mut system_prompt_ignore_terms =
            file_config.system_prompt_ignore_terms.unwrap_or_default();
        Self::dedupe_ignore_terms(&mut system_prompt_ignore_terms);

        let upstream_urls: Vec<String> = file_config
            .upstreams
            .values()
            .map(|u| u.base_url.clone())
            .collect();

        let api_key = None;

        let mut model_map = BTreeMap::new();
        for (upstream_name, upstream) in &file_config.upstreams {
            for (bare_name, model_conf) in &upstream.models {
                let namespaced_id = format!("{}/{}", upstream_name, bare_name);
                let target = model_conf.target.as_deref().unwrap_or(bare_name);
                if target != bare_name {
                    model_map.insert(namespaced_id, target.to_string());
                }
            }
        }

        // Validate upstream URLs
        for (name, upstream) in &file_config.upstreams {
            Self::resolve_chat_completions_url(&upstream.base_url)
                .map_err(|e| anyhow::anyhow!("Upstream '{}': {}", name, e))?;
        }

        Ok(Config {
            port: file_config.port.unwrap_or(3000),
            debug: file_config.debug.unwrap_or(false),
            verbose: file_config.verbose.unwrap_or(false),
            merge_system_messages: file_config.merge_system_messages.unwrap_or(false),
            merge_user_messages: file_config.merge_user_messages.unwrap_or(false),
            reasoning_model: file_config.reasoning_model,
            completion_model: file_config.completion_model,
            models_list_mode: file_config.models_list_mode.unwrap_or_default(),
            upstreams: file_config.upstreams,
            system_prompt_ignore_terms,
            upstream_urls,
            api_key,
            model_map,
        })
    }

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
                    });
                }

                // Upstream found but model not in models map — if models map is empty,
                // treat as "accepts any model" and use bare name as target
                if upstream.models.is_empty() {
                    let chat_url = Self::resolve_chat_completions_url(&upstream.base_url)?;
                    let api_key = upstream.api_key.clone().or_else(|| self.api_key.clone());

                    return Ok(ResolvedRoute {
                        upstream_name: prefix.to_string(),
                        target: remainder.to_string(),
                        base_url: chat_url,
                        api_key,
                        allow_failover: false,
                        reasoning_model: None,
                        completion_model: None,
                        is_legacy: false,
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
                });
            }

            // Upstream with empty models map accepts any bare model name
            if upstream.models.is_empty() {
                let chat_url = Self::resolve_chat_completions_url(&upstream.base_url)?;
                let api_key = upstream.api_key.clone().or_else(|| self.api_key.clone());

                return Ok(ResolvedRoute {
                    upstream_name: upstream_name.clone(),
                    target: model.to_string(),
                    base_url: chat_url,
                    api_key,
                    allow_failover: false,
                    reasoning_model: None,
                    completion_model: None,
                    is_legacy: false,
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
        })
    }

    pub fn failover_upstreams(&self, model: &str, exclude_upstream: &str) -> Vec<ResolvedRoute> {
        let mut alternatives = Vec::new();

        let resolved = self.resolve_model(model).ok();
        let original_target = resolved
            .as_ref()
            .map(|r| r.target.as_str())
            .unwrap_or(model);
        let bare_part = model
            .split_once('/')
            .map(|(_, remainder)| remainder)
            .unwrap_or(model);

        for (upstream_name, upstream) in &self.upstreams {
            if upstream_name == exclude_upstream {
                continue;
            }

            for (bare_name, model_conf) in &upstream.models {
                let target = model_conf.target.as_deref().unwrap_or(bare_name);
                if bare_name == bare_part || target == original_target {
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
                        });
                    }
                    break;
                }
            }
        }

        alternatives
    }

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn base_url_without_version_defaults_to_v1_endpoint() {
        let url = Config::resolve_chat_completions_url("https://api.openai.com").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn versioned_base_url_preserves_existing_version() {
        let url = Config::resolve_chat_completions_url("https://gateway.example.com/v2").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/chat/completions");
    }

    #[test]
    fn full_chat_completions_endpoint_is_used_as_is() {
        let url = Config::resolve_chat_completions_url(
            "https://gateway.example.com/v2/chat/completions/",
        )
        .unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/chat/completions");
    }

    #[test]
    fn models_url_without_version_defaults_to_v1_endpoint() {
        let url = Config::resolve_models_url("https://api.openai.com").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/models");
    }

    #[test]
    fn versioned_models_url_preserves_existing_version() {
        let url = Config::resolve_models_url("https://gateway.example.com/v2").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/models");
    }

    #[test]
    fn full_chat_completions_endpoint_resolves_models_url() {
        let url =
            Config::resolve_models_url("https://gateway.example.com/v2/chat/completions").unwrap();
        assert_eq!(url, "https://gateway.example.com/v2/models");
    }

    #[test]
    fn partial_chat_path_is_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2/chat")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("service base URL, a versioned base URL"));
    }

    #[test]
    fn query_strings_are_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2?foo=bar")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("must not include query parameters or fragments"));
    }

    #[test]
    fn fragments_are_rejected() {
        let err = Config::resolve_chat_completions_url("https://gateway.example.com/v2#section")
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("must not include query parameters or fragments"));
    }

    #[test]
    fn empty_url_is_rejected() {
        let err = Config::resolve_chat_completions_url("").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn non_http_scheme_is_rejected() {
        let err = Config::resolve_chat_completions_url("ftp://gateway.example.com").unwrap_err();
        assert!(err.to_string().contains("must use http or https"));
    }

    #[test]
    fn explicit_v1_is_preserved_not_doubled() {
        let url = Config::resolve_chat_completions_url("https://openrouter.ai/api/v1").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn trailing_slash_on_base_url_is_normalized() {
        let url = Config::resolve_chat_completions_url("https://api.openai.com/").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn models_url_from_explicit_v1() {
        let url = Config::resolve_models_url("https://openrouter.ai/api/v1").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/models");
    }

    #[test]
    fn models_url_with_trailing_slash() {
        let url = Config::resolve_models_url("https://api.openai.com/").unwrap();
        assert_eq!(url, "https://api.openai.com/v1/models");
    }

    #[test]
    fn url_with_subpath_and_no_version_defaults_to_v1() {
        let url = Config::resolve_chat_completions_url("https://openrouter.ai/api").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1/chat/completions");
    }

    #[test]
    fn only_completions_path_is_rejected() {
        let err =
            Config::resolve_chat_completions_url("https://gateway.example.com/v2/completions")
                .unwrap_err();
        assert!(err
            .to_string()
            .contains("service base URL, a versioned base URL"));
    }

    #[test]
    fn uppercase_version_prefix_is_accepted() {
        let url = Config::resolve_chat_completions_url("https://gateway.example.com/V2").unwrap();
        assert_eq!(url, "https://gateway.example.com/V2/chat/completions");
    }

    #[test]
    fn parse_system_prompt_ignore_terms_supports_semicolons_and_newlines() {
        let terms =
            Config::parse_system_prompt_ignore_terms("rm -rf;git reset --hard\nsudo rm -rf");

        assert_eq!(
            terms,
            vec![
                "rm -rf".to_string(),
                "git reset --hard".to_string(),
                "sudo rm -rf".to_string()
            ]
        );
    }

    #[test]
    fn dedupe_ignore_terms_normalizes_case_and_whitespace() {
        let mut terms = vec![
            "rm -rf".to_string(),
            " RM\t-rF ".to_string(),
            "git reset --hard".to_string(),
        ];

        Config::dedupe_ignore_terms(&mut terms);

        assert_eq!(
            terms,
            vec!["rm -rf".to_string(), "git reset --hard".to_string()]
        );
    }

    #[test]
    fn parse_model_map_supports_semicolons_and_newlines() {
        let model_map = Config::parse_model_map(
            "claude-3-5-sonnet=openai/gpt-5.2-chat\nclaude-haiku=openai/gpt-4.1-mini",
        )
        .unwrap();

        assert_eq!(
            model_map.get("claude-3-5-sonnet"),
            Some(&"openai/gpt-5.2-chat".to_string())
        );
        assert_eq!(
            model_map.get("claude-haiku"),
            Some(&"openai/gpt-4.1-mini".to_string())
        );
    }

    #[test]
    fn parse_model_map_rejects_invalid_entries() {
        let err = Config::parse_model_map("claude-3-5-sonnet").unwrap_err();

        assert!(err.to_string().contains("Expected source=target"));
    }

    #[test]
    fn parse_upstream_urls_splits_on_semicolons() {
        let urls = Config::parse_upstream_urls("https://openrouter.ai/api;https://api.openai.com")
            .unwrap();

        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://openrouter.ai/api");
        assert_eq!(urls[1], "https://api.openai.com");
    }

    #[test]
    fn parse_upstream_urls_single_url_still_works() {
        let urls = Config::parse_upstream_urls("https://api.openai.com").unwrap();
        assert_eq!(urls.len(), 1);
    }

    #[test]
    fn parse_upstream_urls_rejects_empty() {
        let err = Config::parse_upstream_urls("").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn parse_upstream_urls_validates_each_url() {
        let err = Config::parse_upstream_urls("https://api.openai.com;not-a-url").unwrap_err();
        assert!(err.to_string().contains("valid http"));
    }

    #[test]
    fn chat_completions_urls_resolves_all() {
        let config = Config {
            upstream_urls: vec![
                "https://openrouter.ai/api".to_string(),
                "https://api.openai.com".to_string(),
            ],
            ..Default::default()
        };

        let urls = config.chat_completions_urls();
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://openrouter.ai/api/v1/chat/completions");
        assert_eq!(urls[1], "https://api.openai.com/v1/chat/completions");
    }

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
        tmp.write_all(serde_json::to_string_pretty(&json).unwrap().as_bytes())
            .unwrap();

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
        assert!(gpt_model.target.is_none());
        assert!(!gpt_model.allow_failover);

        let o1_model = &upstream.models["o1-pro"];
        assert_eq!(o1_model.target.as_deref(), Some("o1-pro"));
    }

    #[test]
    fn resolve_model_prefix_match() {
        let config = Config {
            upstreams: BTreeMap::from([(
                "openai".to_string(),
                UpstreamConfig {
                    base_url: "https://api.openai.com".to_string(),
                    api_key: Some("sk-test".to_string()),
                    models: BTreeMap::from([(
                        "gpt-4.1".to_string(),
                        ModelConfig {
                            target: Some("gpt-4.1".to_string()),
                            allow_failover: false,
                            reasoning_model: None,
                            completion_model: None,
                        },
                    )]),
                },
            )]),
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
            upstreams: BTreeMap::from([(
                "openai".to_string(),
                UpstreamConfig {
                    base_url: "https://api.openai.com".to_string(),
                    api_key: None,
                    models: BTreeMap::from([(
                        "gpt-4.1".to_string(),
                        ModelConfig {
                            target: Some("gpt-4.1".to_string()),
                            allow_failover: false,
                            reasoning_model: None,
                            completion_model: None,
                        },
                    )]),
                },
            )]),
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

    #[test]
    fn static_models_list_from_config() {
        let config = Config {
            upstreams: BTreeMap::from([
                (
                    "openai".to_string(),
                    UpstreamConfig {
                        base_url: "https://api.openai.com".to_string(),
                        api_key: None,
                        models: BTreeMap::from([(
                            "gpt-4.1".to_string(),
                            ModelConfig {
                                target: Some("gpt-4.1".to_string()),
                                allow_failover: false,
                                reasoning_model: None,
                                completion_model: None,
                            },
                        )]),
                    },
                ),
                (
                    "openrouter".to_string(),
                    UpstreamConfig {
                        base_url: "https://openrouter.ai/api".to_string(),
                        api_key: None,
                        models: BTreeMap::from([(
                            "anthropic/claude-opus-4-7".to_string(),
                            ModelConfig {
                                target: Some("anthropic/claude-opus-4-7".to_string()),
                                allow_failover: false,
                                reasoning_model: None,
                                completion_model: None,
                            },
                        )]),
                    },
                ),
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
}
