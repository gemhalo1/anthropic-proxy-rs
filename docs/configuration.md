# 配置指南

anthropic-proxy 支持三种配置方式：**环境变量**、**`.env` 文件** 和 **JSON 配置文件**。优先级为：

```
CLI 参数 > 环境变量 > JSON 配置文件 > 默认值
```

## 环境变量

使用环境变量是最简单的配置方式，适合快速启动或单上游场景。

### 必需变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `UPSTREAM_BASE_URL` | 上游 OpenAI-compatible API 地址（不使用 JSON 配置文件时必需） | `https://openrouter.ai/api` |

> `ANTHROPIC_PROXY_BASE_URL` 是 `UPSTREAM_BASE_URL` 的别名，两者等价，设置任意一个即可。

**多上游（failover）：** 用分号分隔多个 URL，请求失败时按顺序尝试下一个：

```bash
UPSTREAM_BASE_URL="https://openrouter.ai/api;https://api.openai.com"
```

### 认证变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `UPSTREAM_API_KEY` | 上游 API 密钥 | `sk-or-v1-...` |
| `OPENROUTER_API_KEY` | `UPSTREAM_API_KEY` 的别名（兼容旧配置） | `sk-or-v1-...` |

> 优先读取 `UPSTREAM_API_KEY`，若不存在则尝试 `OPENROUTER_API_KEY`。如果上游不需要认证（如本地 Ollama），可以不设置。

### 服务变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `3000` | 代理监听端口 |
| `DEBUG` | `false` | 启用调试日志，设为 `1` 或 `true` |
| `VERBOSE` | `false` | 启用详细日志（记录完整请求/响应体），设为 `1` 或 `true` |
| `MODELS_LIST_MODE` | `static` | `/v1/models` 接口的行为模式，可选 `static`、`upstream`、`merge`（详见下方） |

### 模型路由变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `REASONING_MODEL` | 当请求启用 extended thinking 时使用的模型 | `anthropic/claude-3.5-sonnet` |
| `COMPLETION_MODEL` | 标准请求（无 thinking）使用的模型 | `anthropic/claude-3-haiku` |
| `ANTHROPIC_PROXY_MODEL_MAP` | 模型名称映射（`;` 或换行分隔） | `claude-opus-4-6=openai/gpt-4.1;claude-haiku-4-5=openai/gpt-4.1-mini` |

**模型路由优先级：** 代理检测请求中是否包含 `thinking` 参数，如果有则使用 `REASONING_MODEL`，否则使用 `COMPLETION_MODEL`。如果两者都没有设置，则使用客户端请求中的模型名。`ANTHROPIC_PROXY_MODEL_MAP` 在最终模型名确定后应用。

### 系统提示词过滤变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `ANTHROPIC_PROXY_SYSTEM_PROMPT_IGNORE_TERMS` | 转发上游前需要从 system prompt 中移除的关键词（`;` 或换行分隔） | `x-anthropic-billing-header:;rm -rf` |

> 此功能用于解决某些网关/WAF 因 Claude Code 的系统提示词触发 403 的问题。过滤时忽略大小写和多余空白，自动去重。

### .env 文件搜索路径

如果不使用 JSON 配置文件，代理会按以下顺序搜索 `.env` 文件：

1. `--config` 参数指定的路径
2. 当前目录 `./.env`
3. 用户主目录 `~/.anthropic-proxy.env`
4. 系统配置 `/etc/anthropic-proxy/.env`

找到第一个存在的文件即停止搜索，未找到则直接使用环境变量。

---

## JSON 配置文件

JSON 配置文件支持**多上游路由**、**按模型配置 failover** 和 **per-model 覆盖**，适合复杂部署场景。

### 文件搜索路径

1. `--config-file` 参数指定的路径
2. 当前目录 `./anthropic-proxy.json`
3. 用户主目录 `~/.anthropic-proxy.json`
4. 系统配置 `/etc/anthropic-proxy/config.json`

找到第一个存在的文件即停止搜索。如果找到 JSON 配置文件，环境变量仍然会叠加生效（优先级高于配置文件）。

### 配置文件结构

```json
{
  "port": 3000,
  "debug": false,
  "verbose": false,
  "merge_system_messages": false,
  "merge_user_messages": false,
  "models_list_mode": "static",
  "system_prompt_ignore_terms": ["x-anthropic-billing-header:"],
  "reasoning_model": "openai/o3",
  "completion_model": "openai/gpt-4.1-mini",
  "upstreams": {
    "<上游名称>": {
      "base_url": "<上游 URL>",
      "api_key": "<上游 API 密钥>",
      "models": {
        "<裸模型名>": {
          "target": "<实际发送给上游的模型名>",
          "allow_failover": true,
          "reasoning_model": "<此模型使用的推理模型>",
          "completion_model": "<此模型使用的补全模型>"
        }
      }
    }
  }
}
```

### 顶层字段详解

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `port` | number | `3000` | 代理监听端口 |
| `debug` | boolean | `false` | 启用调试日志 |
| `verbose` | boolean | `false` | 启用详细日志（完整请求/响应体） |
| `merge_system_messages` | boolean | `false` | 将多条 system message 合并为一条（用换行连接），适合不支持多条 system prompt 的上游（如 OpenRouter/Ollama） |
| `merge_user_messages` | boolean | `false` | 将连续的多条 user message 合并为一条（用换行连接），适合不支持多轮对话的上游 |
| `models_list_mode` | string | `"static"` | `/v1/models` 接口行为，详见下方"模型列表模式"章节 |
| `system_prompt_ignore_terms` | string[] | `[]` | 转发前从 system prompt 中移除的关键词列表 |
| `reasoning_model` | string | null | 全局推理模型覆盖（请求包含 `thinking` 参数时使用） |
| `completion_model` | string | null | 全局补全模型覆盖（标准请求时使用） |
| `upstreams` | object | `{}` | 上游服务配置映射，键名为上游名称 |

### UpstreamConfig 字段详解

每个上游的配置结构：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `base_url` | string | 是 | 上游 API 基础 URL。URL 解析规则：若末尾路径包含 `/v1` 等版本标识则保留，否则自动补 `/v1`；若已是 `/chat/completions` 完整路径则直接使用 |
| `api_key` | string | 否 | 此上游的 API 密钥。若不设置，会尝试使用全局 `UPSTREAM_API_KEY` 环境变量 |
| `models` | object | 是 | 模型配置映射，键名为裸模型名（不含上游前缀），值为 ModelConfig |

> **上游名称** 是 `upstreams` 对象的键名（如 `"openrouter"`、`"openai"`），用于生成命名空间模型 ID。建议使用简短英文字母和下划线。

### ModelConfig 字段详解

每个模型的配置结构：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target` | string | 裸模型名本身 | 实际发送给上游 API 的模型标识符。若不设置或为 null，则使用裸模型键名 |
| `allow_failover` | boolean | `false` | 是否允许此模型在当前上游请求失败时尝试其他上游的同名模型 |
| `reasoning_model` | string | null | 针对此模型的推理模型覆盖（优先级高于全局 `reasoning_model`） |
| `completion_model` | string | null | 针对此模型的补全模型覆盖（优先级高于全局 `completion_model`） |

**`target` 字段示例：**

```json
{
  "models": {
    "claude-sonnet-4-20250514": {
      "target": "anthropic/claude-sonnet-4-20250514"
    },
    "gpt-4.1": {
      "target": "openai/gpt-4.1"
    },
    "o3": {}
  }
}
```

- `claude-sonnet-4-20250514` 的 target 显式指定为 `anthropic/claude-sonnet-4-20250514`（OpenRouter 的命名格式）
- `gpt-4.1` 的 target 显式指定为 `openai/gpt-4.1`
- `o3` 的 target 留空，默认使用裸模型名 `o3`（适合 OpenAI 原生 API）

> 当 `target` 与裸模型名不同时，代理会自动生成模型映射，将命名空间 ID 映射到实际 target。例如 `"claude-sonnet-4-20250514"` 在 `"openrouter"` 上游下会生成映射：`openrouter/claude-sonnet-4-20250514` → `anthropic/claude-sonnet-4-20250514`。

---

## 模型路由机制

### 命名空间模型 ID

当使用 JSON 配置文件时，每个模型会被赋予**命名空间 ID**，格式为 `<上游名称>/<裸模型名>`。

例如以下配置：

```json
{
  "upstreams": {
    "openrouter": {
      "base_url": "https://openrouter.ai/api",
      "models": {
        "claude-sonnet-4-20250514": {}
      }
    }
  }
}
```

客户端请求中使用 `openrouter/claude-sonnet-4-20250514` 即可路由到 OpenRouter 上游的 `claude-sonnet-4-20250514` 模型。

### 路由匹配顺序

代理按以下顺序匹配请求中的模型名：

1. **前缀匹配** — 如果模型名包含 `/`，拆分为 `<上游名>/<裸模型名>`，在对应上游中查找
2. **裸名匹配** — 如果模型名不含 `/`，按字母顺序遍历所有上游查找同名模型，返回第一个匹配
3. **遗留模式** — 如果都不匹配，退回到环境变量方式，使用 `UPSTREAM_BASE_URL` 配置的全局上游

**示例：**

```
请求模型: openrouter/gpt-4.1
  → 前缀匹配: 在 "openrouter" 上游查找 "gpt-4.1" → 找到 → 路由到 OpenRouter

请求模型: gpt-4.1
  → 裸名匹配: 依次在 "ollama"、"openai"、"openrouter" 中查找 "gpt-4.1"
  → 在 "openai" 找到 → 路由到 OpenAI

请求模型: llama-99
  → 前缀匹配: 无上游叫 "llama-99"
  → 裸名匹配: 所有上游都没有 "llama-99"
  → 遗留模式: 路由到全局 UPSTREAM_BASE_URL
```

### Failover 机制

当请求一个设置了 `allow_failover: true` 的模型时，如果当前上游返回错误，代理会自动尝试其他上游中同名（裸名或 target 名匹配）的模型。

```json
{
  "upstreams": {
    "openrouter": {
      "models": {
        "gpt-4.1": { "target": "openai/gpt-4.1", "allow_failover": true }
      }
    },
    "openai": {
      "models": {
        "gpt-4.1": {}
      }
    }
  }
}
```

请求 `openrouter/gpt-4.1` 时，如果 OpenRouter 返回错误，代理会自动尝试 OpenAI 的 `gpt-4.1`。

> **注意：** `allow_failover` 默认为 `false`。只有在模型配置中显式设置为 `true` 时才会触发 failover。

---

## 模型列表模式

`models_list_mode` 控制 `/v1/models` 接口的行为：

| 模式 | 说明 |
|------|------|
| `static` | 返回 JSON 配置文件中声明的所有模型。每个模型以命名空间 ID 显示（如 `openai/gpt-4.1`），`display_name` 为 target 或裸模型名 |
| `upstream` | 实时请求上游的 `/v1/models` 接口，返回上游实际支持的模型列表。需要上游提供 models 接口 |
| `merge` | 合合 `static` 和 `upstream` 的结果，去除重复 |

环境变量方式：

```bash
MODELS_LIST_MODE=static    # 或 upstream 或 merge
```

JSON 配置文件：

```json
{
  "models_list_mode": "static"
}
```

CLI 参数目前没有直接选项，但可以通过环境变量或配置文件设置。

---

## URL 解析规则

`base_url` 字段（无论是配置文件中的还是 `UPSTREAM_BASE_URL` 环境变量）的解析规则：

| 输入 | 解析结果（chat completions） | 解析结果（models） |
|------|------|------|
| `https://api.openai.com` | `https://api.openai.com/v1/chat/completions` | `https://api.openai.com/v1/models` |
| `https://openrouter.ai/api` | `https://openrouter.ai/api/v1/chat/completions` | `https://openrouter.ai/api/v1/models` |
| `https://openrouter.ai/api/v1` | `https://openrouter.ai/api/v1/chat/completions` | `https://openrouter.ai/api/v1/models` |
| `https://gateway.example.com/v2` | `https://gateway.example.com/v2/chat/completions` | `https://gateway.example.com/v2/models` |
| `https://gateway.example.com/v2/chat/completions` | 原样使用 | `https://gateway.example.com/v2/models` |
| `http://localhost:11434` | `http://localhost:11434/v1/chat/completions` | `http://localhost:11434/v1/models` |

**规则总结：**

- 如果 URL 末尾已经包含 `/chat/completions` → 直接使用
- 如果 URL 末尾路径段是版本标识（如 `/v1`、`/v2`）→ 追加 `/chat/completions`
- 其他情况 → 追加 `/v1/chat/completions`

> **如果配置中已显式包含 `/v1`，不会重复添加。** 例如 `https://openrouter.ai/api/v1` 会解析为 `https://openrouter.ai/api/v1/chat/completions`，而不是 `https://openrouter.ai/api/v1/v1/chat/completions`。

**不支持的部分路径：** `.../chat` 和 `.../completions`（缺少另一半）会被拒绝。URL 中的查询参数和 fragment 也会被拒绝。

---

## CLI 参数

所有配置都可以通过 CLI 参数覆盖：

| 参数 | 短写 | 说明 |
|------|------|------|
| `--config <FILE>` | `-c` | 指定 `.env` 文件路径 |
| `--config-file <FILE>` | | 指定 JSON 配置文件路径（覆盖默认搜索路径） |
| `--debug` | `-d` | 启用调试日志 |
| `--verbose` | `-v` | 启用详细日志 |
| `--port <PORT>` | `-p` | 监听端口 |
| `--system-prompt-ignore <TEXT>` | | 移除 system prompt 中的关键词（可重复或用 `;` 分隔） |
| `--merge-system` | | 合合多条 system message |
| `--merge-user` | | 合合连续多条 user message |
| `--daemon` | | 以守护进程模式运行 |
| `--pid-file <FILE>` | | PID 文件路径（默认 `/tmp/anthropic-proxy.pid`） |

---

## 完整配置示例

参见项目根目录的 `anthropic-proxy.example.json` 文件。将其复制到搜索路径之一并填入真实 API 密钥即可使用：

```bash
# 复制到当前目录
cp anthropic-proxy.example.json anthropic-proxy.json

# 或复制到主目录
cp anthropic-proxy.example.json ~/.anthropic-proxy.json

# 或指定路径
anthropic-proxy --config-file /path/to/my-config.json
```

---

## 环境变量与 JSON 配置文件共存

当存在 JSON 配置文件时，环境变量会叠加生效，优先级高于配置文件：

| 场景 | 结果 |
|------|------|
| JSON 文件设置了 `port: 8080`，环境变量 `PORT=9090` | 使用 `9090`（环境变量优先） |
| JSON 文件设置了全局 `UPSTREAM_API_KEY`，某上游有自己的 `api_key` | 上游使用自己的 `api_key`；无 `api_key` 的上游使用全局 `UPSTREAM_API_KEY` |
| JSON 文件配置了 `upstreams`，环境变量设置了 `UPSTREAM_BASE_URL` | `UPSTREAM_BASE_URL` **完全覆盖** JSON 中的 upstreams 配置 |
| JSON 文件配置了 `model_map`，环境变量设置了 `ANTHROPIC_PROXY_MODEL_MAP` | `ANTHROPIC_PROXY_MODEL_MAP` **追加** 到 JSON 的 model_map |

> **注意：** `UPSTREAM_BASE_URL` 环境变量会清空 JSON 配置文件中的所有 upstreams，用环境变量中的 URL 生成自动命名的上游（如 `upstream_https_openrouter_ai_api`）。如果需要保留 JSON 中的上游配置，请勿设置 `UPSTREAM_BASE_URL`。