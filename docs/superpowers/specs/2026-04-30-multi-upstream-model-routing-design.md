# Multi-Upstream Model Routing: Design Spec

## Summary

Add JSON config file support for per-model routing across multiple upstreams, with namespaced model IDs, configurable failover, and a controllable `/v1/models` endpoint.

## Motivation

Current proxy uses env vars for all configuration. Models share a single set of upstream URLs with round-robin failover. The `/v1/models` endpoint always fetches from upstream. This design adds:

1. **Per-model routing**: different models go to different upstreams
2. **Config ergonomics**: JSON config file for complex multi-upstream setups
3. **Static model list**: clients can see available models without hitting upstream

## Config File Format

JSON. File search order: `./anthropic-proxy.json`, `~/.anthropic-proxy.json`, `/etc/anthropic-proxy/config.json`. Override with `--config-file <PATH>` CLI arg.

Priority: CLI args > env vars > config file > defaults.

### Schema

```json
{
  "port": 3000,
  "debug": false,
  "verbose": false,
  "merge_system_messages": false,
  "merge_user_messages": false,
  "system_prompt_ignore_terms": ["rm -rf"],
  "reasoning_model": null,
  "completion_model": null,
  "models_list_mode": "static",
  "upstreams": {
    "openrouter": {
      "base_url": "https://openrouter.ai/api",
      "api_key": "sk-or-...",
      "models": {
        "openrouter/anthropic/claude-opus-4-7": {
          "target": "anthropic/claude-opus-4-7"
        },
        "openrouter/openai/gpt-4.1": {
          "target": "openai/gpt-4.1",
          "allow_failover": true,
          "reasoning_model": "anthropic/claude-opus-4-7",
          "completion_model": null
        }
      }
    },
    "openai": {
      "base_url": "https://api.openai.com",
      "api_key": "sk-...",
      "models": {
        "openai/gpt-4.1": {
          "target": "gpt-4.1",
          "allow_failover": true
        },
        "openai/o1-pro": {
          "target": "o1-pro"
        }
      }
    }
  }
}
```

### Field definitions

**Top-level fields** (override env vars):
- `port`, `debug`, `verbose`, `merge_system_messages`, `merge_user_messages`, `system_prompt_ignore_terms`, `reasoning_model`, `completion_model`: same semantics as current env vars
- `models_list_mode`: `static` (default), `upstream`, `merge` — controls `/v1/models` behavior

**Upstream entry** (`upstreams.<name>`):
- `base_url`: required. Same validation as current `UPSTREAM_BASE_URL`
- `api_key`: optional. If omitted, uses top-level `UPSTREAM_API_KEY` env var or no auth
- `models`: required. Map of model ID to model config

**Model entry** (`models.<id>`):
- Model ID format: `<upstream_name>/<original_model_id>`. Example: `openai/gpt-4.1`
- `target`: required. The actual model name sent to the upstream. Example: `gpt-4.1` (without prefix)
- `allow_failover`: optional, default `false`. If `true`, when this upstream fails (429/5xx), try other upstreams that have a model with the same `target`
- `reasoning_model`: optional. Per-model override for thinking/reasoning mode
- `completion_model`: optional. Per-model override for non-thinking mode

## Routing Logic

### Step 1: Resolve model name to (upstream, target)

Given the requested `model` string from the client:

1. **Prefix match**: Split on first `/`. If the prefix matches a configured upstream name (e.g., `openai` in `openai/gpt-4.1`), route to that upstream with `target` as the model name sent upstream.

2. **Bare name match**: If the prefix is not a known upstream (e.g., `gpt-4.1` or `claude-opus-4-6`), iterate all upstreams in config order. Find the first model entry where `target` equals the requested model name. Route to that upstream.

3. **No match found**: If neither prefix nor bare name matches any configured model, fall back to legacy behavior — send the original model name to all configured upstream URLs with current failover logic.

### Step 2: Failover

- When a model routes to a specific upstream and that upstream returns 429/5xx:
  - If `allow_failover: false` (default): return the error to the client immediately
  - If `allow_failover: true`: iterate other upstreams that have a model with the same `target`, in config order, and try each one
- Bare name matches: same `allow_failover` rules apply using the matched model entry's config

### Step 3: Model name substitution

When routing is resolved to (upstream, target):
- Replace the request's `model` field with `target`
- Use the upstream's `base_url` for the chat/completions endpoint
- Use the upstream's `api_key` (or fallback to global) for authentication
- Apply per-model `reasoning_model` / `completion_model` overrides if configured

## /v1/models Endpoint

### Modes

| Mode | Behavior | Default |
|------|----------|---------|
| `static` | Return only models defined in config file | Yes |
| `upstream` | Fetch from first available upstream (current behavior) | No |
| `merge` | Return config-defined models + upstream-fetched models, deduplicated | No |

### Static mode response format

```json
{
  "data": [
    {
      "id": "openai/gpt-4.1",
      "display_name": "gpt-4.1",
      "created_at": "1970-01-01T00:00:00Z",
      "model_type": "model"
    },
    {
      "id": "openrouter/anthropic/claude-opus-4-7",
      "display_name": "anthropic/claude-opus-4-7",
      "created_at": "1970-01-01T00:00:00Z",
      "model_type": "model"
    }
  ],
  "first_id": "openai/gpt-4.1",
  "last_id": "openrouter/anthropic/claude-opus-4-7",
  "has_more": false
}
```

### Merge mode deduplication

If config has `openai/gpt-4.1` and upstream returns `gpt-4.1`, the bare name entry is dropped — the namespaced version supersedes it. Upstream models that don't match any config entry are added with their bare ID.

### Upstream mode

Preserves current behavior: fetch from first available upstream, translate response format. No namespacing applied.

## Env Var Coexistence

Priority chain: CLI args > env vars > config file > defaults.

Env vars that overlap with config file fields:
- `PORT` overrides `port`
- `UPSTREAM_BASE_URL` / `ANTHROPIC_PROXY_BASE_URL` overrides `upstreams` entirely (if set, config file upstreams are ignored)
- `UPSTREAM_API_KEY` / `OPENROUTER_API_KEY` overrides global api_key; per-upstream `api_key` in config still applies
- `ANTHROPIC_PROXY_MODEL_MAP` is additive with config file models (env map entries added after config file models)
- `REASONING_MODEL` / `COMPLETION_MODEL` override config file values
- `DEBUG` / `VERBOSE` override config file boolean values
- `ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS` additive with config file terms

When `UPSTREAM_BASE_URL` env var is set (legacy mode), it takes precedence over config file upstreams. This ensures existing deployments continue working unchanged.

## New CLI Arg

`--config-file <PATH>`: Specify config file path. Overrides default search order.

## Implementation Scope

### New/modified files

- `src/config.rs`: Add `UpstreamConfig`, `ModelConfig`, `ModelsListMode` structs; add JSON file loading; merge with env vars; add `resolve_model()` routing method
- `src/cli.rs`: Add `--config-file` arg
- `src/main.rs`: Wire config file loading into `async_main`
- `src/proxy.rs`: Update `proxyHandler` to use new routing logic; update `list_models_handler` for three modes
- `src/translate/pipeline.rs`: Update `TranslationPolicy` to carry resolved (upstream, target, api_key, base_url); update `select_model`

### Dependencies

- `serde_json`: already present (for JSON parsing)
- No new dependencies needed

## Out of scope

- Hot-reload of config file (requires file watcher, separate feature)
- Rate limiting per upstream
- Request queuing / priority