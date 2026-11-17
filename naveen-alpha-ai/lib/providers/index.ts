import { AnthropicProvider } from "./anthropic";
import { GoogleProvider } from "./google";
import { OpenAICompatProvider } from "./openai-compat";
import type { AIProvider, ProviderId } from "./types";
import { MODELS, ModelStatus } from "@/lib/types";

/** Any OpenAI-compatible API. Base URL can be overridden with <ID>_BASE_URL. */
function compat(id: ProviderId, label: string, keyEnv: string, defaultBase: string) {
  return new OpenAICompatProvider({ id, label, keyEnv, baseEnv: `${id.toUpperCase()}_BASE_URL`, defaultBase, tokenParam: "max_tokens" });
}

export const providers: Record<ProviderId, AIProvider> = {
  openai: new OpenAICompatProvider({
    id: "openai",
    label: "OpenAI",
    keyEnv: "OPENAI_API_KEY",
    baseEnv: "OPENAI_BASE_URL",
    defaultBase: "https://api.openai.com/v1",
    tokenParam: "max_completion_tokens",
  }),
  anthropic: new AnthropicProvider(),
  google: new GoogleProvider(),
  openrouter: new OpenAICompatProvider({
    id: "openrouter",
    label: "OpenRouter",
    keyEnv: "OPENROUTER_API_KEY",
    defaultBase: "https://openrouter.ai/api/v1",
    tokenParam: "max_tokens",
    headers: { "X-Title": "NAVEEN ALPHA AI" },
  }),
  groq: compat("groq", "Groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
  cerebras: new OpenAICompatProvider({
    id: "cerebras",
    label: "Cerebras",
    keyEnv: "CEREBRAS_API_KEY",
    baseEnv: "CEREBRAS_BASE_URL",
    defaultBase: "https://api.cerebras.ai/v1",
    tokenParam: "max_completion_tokens",
  }),
  mistral: compat("mistral", "Mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
  deepseek: compat("deepseek", "DeepSeek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
  xai: compat("xai", "xAI", "XAI_API_KEY", "https://api.x.ai/v1"),
  together: compat("together", "Together AI", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
};

/** Providers with a free tier. These are the only ones picked automatically. */
const FREE_PROVIDERS: ProviderId[] = ["groq", "cerebras", "google", "openrouter", "mistral"];

/**
 * Automatic routing (free tiers only). For each public model, the first entry whose provider key is set wins,
 * so you can add any subset of the free keys and everything still works. Free tiers have daily limits.
 * Paid providers (OpenAI, Anthropic, DeepSeek, xAI, Together) are never picked automatically — see resolveModel.
 */
const FREE_ROUTES: Record<string, [ProviderId, string][]> = {
  "naveen-alpha": [
    ["groq", "openai/gpt-oss-20b"],
    ["cerebras", "gpt-oss-120b"],
    ["openrouter", "openai/gpt-oss-20b:free"],
    ["google", "gemini-3.8-flash"],
    ["mistral", "mistral-small-latest"],
  ],
  "naveen-alpha-fast": [
    ["cerebras", "gpt-oss-120b"],
    ["groq", "openai/gpt-oss-20b"],
    ["google", "gemini-3.8-flash"],
    ["openrouter", "openai/gpt-oss-20b:free"],
    ["mistral", "mistral-small-latest"],
  ],
  "naveen-code": [
    ["cerebras", "gpt-oss-120b"],
    ["groq", "openai/gpt-oss-120b"],
    ["openrouter", "openai/gpt-oss-20b:free"],
    ["google", "gemini-3.8-flash"],
    ["mistral", "mistral-small-latest"],
  ],
  "naveen-vision": [
    ["google", "gemini-3.8-flash"],
    ["openrouter", "nvidia/nemotron-nano-12b-v2-vl:free"],
    ["mistral", "mistral-small-latest"],
  ],
};

const ENV_FOR: Record<string, string> = {
  "naveen-alpha": "NAVEEN_ALPHA_MODEL",
  "naveen-alpha-fast": "NAVEEN_ALPHA_FAST_MODEL",
  "naveen-vision": "NAVEEN_VISION_MODEL",
  "naveen-code": "NAVEEN_CODE_MODEL",
};

export interface Resolved {
  provider: AIProvider;
  model: string;
}

/**
 * Map a public Naveen model id to a concrete provider + model id, or null when nothing is configured.
 * 1) An explicit NAVEEN_*_MODEL=provider:model wins (any provider, including paid ones).
 * 2) Otherwise the first configured FREE provider in FREE_ROUTES is used.
 * Set NAVEEN_ALLOW_PAID=true to also let paid providers act as a last-resort fallback.
 */
export function resolveModel(modelId: string): Resolved | null {
  const spec = process.env[ENV_FOR[modelId] ?? ""];
  if (spec && spec.includes(":")) {
    const [pid, ...rest] = spec.split(":");
    const p = providers[pid as ProviderId];
    const model = rest.join(":");
    if (p && model && p.isConfigured()) return { provider: p, model };
  }
  for (const [pid, model] of FREE_ROUTES[modelId] ?? FREE_ROUTES["naveen-alpha"]) {
    const p = providers[pid];
    if (p.isConfigured()) return { provider: p, model };
  }
  if (process.env.NAVEEN_ALLOW_PAID === "true") {
    const paid: [ProviderId, string][] = [
      ["openai", "gpt-4o-mini"],
      ["anthropic", "claude-sonnet-4-5"],
      ["deepseek", "deepseek-chat"],
      ["xai", "grok-4"],
      ["together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"],
    ];
    for (const [pid, model] of paid) {
      if (providers[pid].isConfigured()) return { provider: providers[pid], model };
    }
  }
  return null;
}

export function isFreeProvider(id: ProviderId) {
  return FREE_PROVIDERS.includes(id);
}

/**
 * Primary model first, then optional backups from NAVEEN_FALLBACK_MODELS
 * (comma-separated "provider:model-id" entries, e.g. "google:gemini-3.8-flash,openrouter:openai/gpt-oss-20b:free").
 * Only providers that have a key configured are included.
 */
export function resolveChain(modelId: string): Resolved[] {
  const chain: Resolved[] = [];
  const seen = new Set<string>();
  const add = (r: Resolved | null) => {
    if (!r) return;
    const k = `${r.provider.id}:${r.model}`;
    if (!seen.has(k)) {
      seen.add(k);
      chain.push(r);
    }
  };
  add(resolveModel(modelId));
  for (const spec of (process.env.NAVEEN_FALLBACK_MODELS ?? "").split(",")) {
    const s = spec.trim();
    if (!s.includes(":")) continue;
    const [pid, ...rest] = s.split(":");
    const p = providers[pid as ProviderId];
    const model = rest.join(":");
    if (p && model && p.isConfigured()) add({ provider: p, model });
  }
  return chain;
}

/** Provider and model names are hidden from visitors unless NAVEEN_SHOW_PROVIDERS=true. */
export function showProviders() {
  return process.env.NAVEEN_SHOW_PROVIDERS === "true";
}

/** Remove provider names from error text shown to visitors. */
export function maskProviderNames(msg: string) {
  if (showProviders()) return msg;
  return msg
    .replace(/\b(Groq|Cerebras|Google|Gemini|OpenRouter|OpenAI|Anthropic|Mistral|DeepSeek|xAI|Together AI) request failed/gi, "Request failed")
    .replace(/\b(Groq|Cerebras|OpenRouter|OpenAI|Anthropic|Mistral|DeepSeek|xAI|Together AI)\b/g, "the AI service");
}

export function modelStatuses(): ModelStatus[] {
  const show = showProviders();
  return MODELS.map((m) => {
    const r = resolveModel(m.id);
    return {
      id: m.id,
      configured: !!r,
      provider: show ? r?.provider.id ?? null : null,
      providerLabel: show ? r?.provider.label ?? null : null,
      model: show ? r?.model ?? null : null,
    };
  });
}
