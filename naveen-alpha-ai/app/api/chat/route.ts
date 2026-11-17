import { NextRequest } from "next/server";
import { maskProviderNames, resolveChain, resolveModel, showProviders } from "@/lib/providers";
import { ProviderError, ProviderMessage } from "@/lib/providers/types";
import { buildSystem } from "@/lib/server/prompts";
import { clientIp, limits, rateLimit, sanitizeText } from "@/lib/server/security";
import { MODELS, Attachment } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

function cleanAttachments(raw: unknown): Attachment[] {
  if (!Array.isArray(raw)) return [];
  const out: Attachment[] = [];
  for (const a of raw.slice(0, 12)) {
    if (!a || typeof a !== "object") continue;
    const r = a as Record<string, unknown>;
    const kind = r.kind;
    if (kind !== "image" && kind !== "pdf" && kind !== "text" && kind !== "video" && kind !== "other") continue;
    const att: Attachment = {
      id: sanitizeText(r.id, 64),
      name: sanitizeText(r.name, 200) || "file",
      mime: sanitizeText(r.mime, 100),
      size: Number(r.size) || 0,
      kind,
    };
    if (typeof r.text === "string") att.text = sanitizeText(r.text, limits.maxExtractedChars);
    if (kind === "image" && typeof r.dataUrl === "string" && /^data:image\/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/=]+$/.test(r.dataUrl) && r.dataUrl.length < 12_000_000)
      att.dataUrl = r.dataUrl;
    if (r.truncated) att.truncated = true;
    out.push(att);
  }
  return out;
}

/** Keep only the most recent messages that fit within maxChars (the last message is always kept, truncated if needed). */
function fitMessages(msgs: ProviderMessage[], maxChars: number): ProviderMessage[] {
  const size = (m: ProviderMessage) => m.content.length + (m.attachments ?? []).reduce((n, a) => n + (a.text?.length ?? 0), 0);
  const out: ProviderMessage[] = [];
  let used = 0;
  for (let i = msgs.length - 1; i >= 0; i--) {
    const sz = size(msgs[i]);
    if (out.length && used + sz > maxChars) break;
    out.unshift(msgs[i]);
    used += sz;
  }
  // Drop leading assistant turns so the history starts with a user message.
  while (out.length > 1 && out[0].role === "assistant") out.shift();
  if (out.length === 1 && size(out[0]) > maxChars) {
    const m = out[0];
    out[0] = { ...m, content: m.content.slice(0, Math.max(0, maxChars - 200)) + "\n\n[Message shortened to fit this model's free-tier limit.]", attachments: m.attachments?.map((a) => ({ ...a, text: a.text ? a.text.slice(0, Math.max(0, maxChars - 200)) : a.text })) };
  }
  return out;
}

// Free tiers limit tokens per minute (Groq: 8,000). Keep each request small enough to fit.
const MAX_OUTPUT_TOKENS = Math.max(256, Number(process.env.NAVEEN_MAX_OUTPUT_TOKENS) || 3000);
const GROQ_CONTEXT_CHARS = Math.max(2000, Number(process.env.NAVEEN_GROQ_CONTEXT_CHARS) || 12000);

export async function POST(req: NextRequest) {
  const rl = rateLimit(`chat:${clientIp(req)}`);
  if (!rl.ok) return json(429, { error: `Rate limit reached. Try again in ${rl.retryAfter}s.` }, { "Retry-After": String(rl.retryAfter) });

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return json(400, { error: "Invalid JSON body" });
  }

  const modelId = typeof body.model === "string" ? body.model : "";
  if (!MODELS.some((m) => m.id === modelId)) return json(400, { error: "Unknown model" });

  const resolved = resolveModel(modelId);
  if (!resolved)
    return json(503, {
      error: "not_configured",
      message:
        "No AI provider is configured. Add a free-tier key such as GROQ_API_KEY, GOOGLE_API_KEY, CEREBRAS_API_KEY or OPENROUTER_API_KEY to your server environment.",
    });

  const rawMsgs = Array.isArray(body.messages) ? body.messages.slice(-limits.maxMessages) : [];
  const messages: ProviderMessage[] = [];
  let total = 0;
  for (const m of rawMsgs) {
    if (!m || typeof m !== "object") continue;
    const r = m as Record<string, unknown>;
    if (r.role !== "user" && r.role !== "assistant") continue;
    const content = sanitizeText(r.content, limits.maxMessageChars);
    const attachments = r.role === "user" ? cleanAttachments(r.attachments) : undefined;
    total += content.length + (attachments ?? []).reduce((n, a) => n + (a.text?.length ?? 0) + (a.dataUrl ? 2000 : 0), 0);
    messages.push({ role: r.role, content, attachments });
  }
  if (!messages.length || messages[messages.length - 1].role !== "user") return json(400, { error: "Last message must be from the user" });
  if (total > limits.maxTotalChars) return json(413, { error: "Conversation is too large. Start a new chat or remove large files." });

  const temperature = typeof body.temperature === "number" && body.temperature >= 0 && body.temperature <= 2 ? body.temperature : undefined;
  const system = buildSystem({
    agent: body.agent === true,
    projectContext: sanitizeText(body.projectContext, 300_000),
    custom: sanitizeText(body.customInstructions, 4000),
  });

  const hasImages = messages.some((m) => m.attachments?.some((a) => a.kind === "image" && a.dataUrl));
  const hasFiles = messages.some((m) => m.attachments?.length);

  const enc = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (o: unknown) => {
        try {
          controller.enqueue(enc.encode(`data: ${JSON.stringify(o)}\n\n`));
        } catch {}
      };
      send({ type: "stage", stage: "understanding" });
      if (hasFiles) send({ type: "stage", stage: "files" });
      if (hasImages) send({ type: "stage", stage: "vision" });
      send({ type: "stage", stage: "generating" });
      // Backups are skipped for image requests (a backup may be text-only).
      const chain = hasImages ? [resolved] : resolveChain(modelId);
      try {
        for (let i = 0; i < chain.length; i++) {
          const cand = chain[i];
          let started = false;
          if (showProviders()) send({ type: "meta", provider: cand.provider.label, model: cand.model });
          else send({ type: "meta", provider: "Naveen", model: modelId });
          try {
            for await (const delta of cand.provider.stream({
              model: cand.model,
              system,
              messages: cand.provider.id === "groq" ? fitMessages(messages, GROQ_CONTEXT_CHARS) : messages,
              temperature,
              maxTokens: MAX_OUTPUT_TOKENS,
              signal: req.signal,
            })) {
              started = true;
              send({ type: "delta", text: delta });
            }
            send({ type: "done" });
            break;
          } catch (e) {
            if (req.signal.aborted) break;
            const status = e instanceof ProviderError ? e.status : 500;
            const retryable = status === 403 || status === 404 || status === 408 || status === 413 || status === 429 || status >= 500;
            // Fail over only if nothing has been shown yet and a backup exists.
            if (!started && retryable && i < chain.length - 1) {
              console.error(`provider ${cand.provider.id} failed (${status}), trying backup`);
              continue;
            }
            const msg = e instanceof ProviderError ? e.message : "The AI provider returned an unexpected error.";
            if (!(e instanceof ProviderError)) console.error("chat stream error", e);
            send({ type: "error", message: maskProviderNames(msg) });
            break;
          }
        }
      } finally {
        try {
          controller.close();
        } catch {}
      }
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-store, no-transform", Connection: "keep-alive", "X-Accel-Buffering": "no" },
  });
}
