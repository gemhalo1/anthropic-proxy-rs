mod cli;
mod config;
mod error;
mod metrics;
mod models;
mod proxy;
mod translate;

use axum::{
    routing::{get, post},
    Extension, Router,
};
use clap::Parser;
use cli::{Cli, Command};
use config::{Config, ModelsListMode, UpstreamConfig};
use daemonize::Daemonize;
use reqwest::Client;
use std::collections::BTreeMap;
use std::sync::Arc;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(command) = cli.command {
        match command {
            Command::Stop { pid_file } => {
                stop_daemon(&pid_file)?;
                return Ok(());
            }
            Command::Status { pid_file } => {
                check_status(&pid_file)?;
                return Ok(());
            }
        }
    }

    if cli.daemon {
        use std::fs::OpenOptions;

        let stdout = OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/anthropic-proxy.log")?;

        let stderr = OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/anthropic-proxy.log")?;

        let daemonize = Daemonize::new()
            .pid_file(&cli.pid_file)
            .working_directory(std::env::current_dir()?)
            .stdout(stdout)
            .stderr(stderr)
            .umask(0o027);

        match daemonize.start() {
            Ok(_) => {}
            Err(e) => {
                eprintln!("✗ Failed to daemonize: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("✓ Starting proxy in foreground mode");
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async_main(cli))
}

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

fn merge_env_overrides(config: &mut Config) {
    // Individual env var overrides (CLI args > env vars > config file > defaults)
    if let Ok(port) = std::env::var("PORT") {
        if let Ok(p) = port.parse::<u16>() {
            config.port = p;
        }
    }
    if std::env::var("DEBUG")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
    {
        config.debug = true;
    }
    if std::env::var("VERBOSE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
    {
        config.verbose = true;
    }
    if let Ok(key) =
        std::env::var("UPSTREAM_API_KEY").or_else(|_| std::env::var("OPENROUTER_API_KEY"))
    {
        if !key.is_empty() {
            config.api_key = Some(key);
        }
    }
    if let Ok(model) = std::env::var("REASONING_MODEL") {
        config.reasoning_model = Some(model);
    }
    if let Ok(model) = std::env::var("COMPLETION_MODEL") {
        config.completion_model = Some(model);
    }
    if let Ok(mode) = std::env::var("MODELS_LIST_MODE") {
        match mode.to_lowercase().as_str() {
            "static" => config.models_list_mode = ModelsListMode::Static,
            "upstream" => config.models_list_mode = ModelsListMode::Upstream,
            "merge" => config.models_list_mode = ModelsListMode::Merge,
            _ => eprintln!("⚠️  WARNING: Invalid MODELS_LIST_MODE '{}', expected static/upstream/merge", mode),
        }
    }

    let env_terms = std::env::var("ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS")
        .ok()
        .map(|value| Config::parse_system_prompt_ignore_terms(&value))
        .unwrap_or_default();
    if !env_terms.is_empty() {
        config.system_prompt_ignore_terms.extend(env_terms);
        Config::dedupe_ignore_terms(&mut config.system_prompt_ignore_terms);
    }

    // UPSTREAM_BASE_URL env var overrides config file upstreams entirely
    if let Ok(raw_urls) =
        std::env::var("UPSTREAM_BASE_URL").or_else(|_| std::env::var("ANTHROPIC_PROXY_BASE_URL"))
    {
        if let Ok(urls) = Config::parse_upstream_urls(&raw_urls) {
            config.upstream_urls = urls.clone();
            config.upstreams.clear();
            for url in &urls {
                let name = format!(
                    "upstream_{}",
                    url.replace(['.', '/', ':'], "_").trim_end_matches('_')
                );
                config.upstreams.insert(
                    name,
                    UpstreamConfig {
                        base_url: url.clone(),
                        api_key: config.api_key.clone(),
                        models: BTreeMap::new(),
                    },
                );
            }
        }
    }

    // ANTHROPIC_PROXY_MODEL_MAP is additive
    if let Ok(value) = std::env::var("ANTHROPIC_PROXY_MODEL_MAP") {
        if let Ok(env_map) = Config::parse_model_map(&value) {
            config.model_map.extend(env_map);
        }
    }
}

async fn async_main(cli: Cli) -> anyhow::Result<()> {
    let mut config = if let Some(config_path) = find_config_file(&cli) {
        eprintln!("📄 Loaded config from: {}", config_path.display());
        let mut config = Config::from_json_file(&config_path)?;
        merge_env_overrides(&mut config);
        config
    } else {
        Config::from_env_with_path(cli.config)?
    };

    if cli.debug {
        config.debug = true;
    }
    if cli.verbose {
        config.verbose = true;
    }
    if let Some(port) = cli.port {
        config.port = port;
    }
    if !cli.system_prompt_ignore.is_empty() {
        config.system_prompt_ignore_terms.extend(
            cli.system_prompt_ignore
                .into_iter()
                .map(|term| term.trim().to_string())
                .filter(|term| !term.is_empty()),
        );
        Config::dedupe_ignore_terms(&mut config.system_prompt_ignore_terms);
    }
    if cli.merge_system_messages {
        config.merge_system_messages = true;
    }
    if cli.merge_user_messages {
        config.merge_user_messages = true;
    }

    let log_level = if config.verbose {
        tracing::Level::TRACE
    } else if config.debug {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("anthropic_proxy={}", log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Anthropic Proxy v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Port: {}", config.port);
    tracing::info!("Upstream URLs: {}", config.upstream_urls.join("; "));
    tracing::info!(
        "Resolved chat completions URLs: {}",
        config.chat_completions_urls().join("; ")
    );
    if let Some(ref model) = config.reasoning_model {
        tracing::info!("Reasoning Model Override: {}", model);
    }
    if let Some(ref model) = config.completion_model {
        tracing::info!("Completion Model Override: {}", model);
    }
    if config.api_key.is_some() {
        tracing::info!("API Key: configured");
    } else {
        tracing::info!("API Key: not set (using unauthenticated endpoint)");
    }
    if !config.system_prompt_ignore_terms.is_empty() {
        tracing::info!(
            "System prompt ignore terms: {}",
            config.system_prompt_ignore_terms.join("; ")
        );
    }
    if !config.model_map.is_empty() {
        let entries = config
            .model_map
            .iter()
            .map(|(source, target)| format!("{source} -> {target}"))
            .collect::<Vec<_>>()
            .join("; ");
        tracing::info!("Model map: {}", entries);
    }

    if !config.upstreams.is_empty() {
        tracing::info!(
            "Configured upstreams: {}",
            config
                .upstreams
                .keys()
                .map(|k| k.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let total_models = config
            .upstreams
            .values()
            .map(|u| u.models.len())
            .sum::<usize>();
        tracing::info!(
            "Configured models: {} across {} upstreams",
            total_models,
            config.upstreams.len()
        );
    }

    let metrics_handle = metrics::install();

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(10))
        .pool_max_idle_per_host(10)
        .build()?;

    let config = Arc::new(config);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/messages", post(proxy::proxy_handler))
        .route("/v1/chat/completions", post(proxy::chat_completions_handler))
        .route("/v1/models", get(proxy::list_models_handler))
        .route("/health", axum::routing::get(health_handler))
        .route(
            "/metrics",
            get(move || {
                let handle = metrics_handle.clone();
                async move { handle.render() }
            }),
        )
        .layer(Extension(config.clone()))
        .layer(Extension(client))
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Listening on {}", addr);
    tracing::info!("Proxy ready to accept requests");

    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler() -> &'static str {
    "OK"
}

fn stop_daemon(pid_file: &std::path::Path) -> anyhow::Result<()> {
    if !pid_file.exists() {
        eprintln!("✗ PID file not found: {}", pid_file.display());
        eprintln!("  Daemon is not running or PID file was removed");
        std::process::exit(1);
    }

    let pid_str = std::fs::read_to_string(pid_file)?;
    let pid: i32 = pid_str
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid PID in file: {}", pid_str))?;

    #[cfg(unix)]
    {
        use std::process::Command;
        let output = Command::new("kill").arg(pid.to_string()).output()?;

        if output.status.success() {
            std::fs::remove_file(pid_file)?;
            eprintln!("✓ Daemon stopped (PID: {})", pid);
        } else {
            eprintln!("✗ Failed to stop daemon (PID: {})", pid);
            eprintln!("  Process may have already exited");
            std::fs::remove_file(pid_file)?;
            std::process::exit(1);
        }
    }

    #[cfg(not(unix))]
    {
        eprintln!("✗ Daemon stop is only supported on Unix systems");
        std::process::exit(1);
    }

    Ok(())
}

fn check_status(pid_file: &std::path::Path) -> anyhow::Result<()> {
    if !pid_file.exists() {
        eprintln!("✗ Daemon is not running");
        eprintln!("  PID file not found: {}", pid_file.display());
        std::process::exit(1);
    }

    let pid_str = std::fs::read_to_string(pid_file)?;
    let pid: i32 = pid_str
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid PID in file: {}", pid_str))?;

    #[cfg(unix)]
    {
        use std::process::Command;
        let output = Command::new("ps").arg("-p").arg(pid.to_string()).output()?;

        if output.status.success() {
            eprintln!("✓ Daemon is running (PID: {})", pid);
            eprintln!("  PID file: {}", pid_file.display());
        } else {
            eprintln!("✗ Daemon is not running");
            eprintln!(
                "  Stale PID file found: {} (PID: {})",
                pid_file.display(),
                pid
            );
            std::process::exit(1);
        }
    }

    #[cfg(not(unix))]
    {
        eprintln!("✗ Daemon status check is only supported on Unix systems");
        std::process::exit(1);
    }

    Ok(())
}
