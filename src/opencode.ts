import { reductionRatio } from './compact.js';
import type {
  CallDecision,
  CompactResult,
  Message,
  ToolCall,
} from './types.js';

/**
 * opencode adapter for fast-jev-compaction.
 *
 * Maps opencode's `experimental.chat.messages.transform` payload
 * (`{ info, parts }[]`) onto the library's host-agnostic `Message[]`, and
 * applies Jev decisions back onto the opencode parts **in place** (opencode
 * keeps its own array reference after the hook, so reassigning
 * `output.messages` is a silent no-op; mutate with `splice`).
 *
 * This module is dependency-free: it only describes the shapes it touches,
 * so the npm library never depends on `@opencode-ai/plugin` or
 * `@opencode-ai/sdk`.
 */

export interface OpencodeTextPart {
  type: 'text';
  text: string;
  [key: string]: unknown;
}

export interface OpencodeToolState {
  status: string;
  input?: Record<string, unknown>;
  output?: string;
  error?: string;
  title?: string;
  time?: { start: number; end?: number; compacted?: number };
  [key: string]: unknown;
}

export interface OpencodeToolPart {
  type: 'tool';
  callID: string;
  tool: string;
  state: OpencodeToolState;
  [key: string]: unknown;
}

export type OpencodePart = OpencodeTextPart | OpencodeToolPart | { type: string; [key: string]: unknown };

export interface OpencodeMessage {
  info: { role: string; [key: string]: unknown };
  parts: OpencodePart[];
}

export function isToolPart(part: OpencodePart): part is OpencodeToolPart {
  return part.type === 'tool';
}

function toolOutput(state: OpencodeToolState): { text: string; isError: boolean } | undefined {
  if (state.status === 'completed' && typeof state.output === 'string') {
    return { text: state.output, isError: false };
  }
  if (state.status === 'error' && typeof state.error === 'string') {
    return { text: state.error, isError: true };
  }
  return undefined;
}

/**
 * Maps opencode messages onto library messages. Each completed/error tool
 * part becomes one tool use plus its result inside the same message, so
 * `collectToolCalls` pairs them by `callID`. Pending/running parts are
 * skipped (nothing to drop yet); text is the concatenation of text parts;
 * messages holding a `compaction` part are carried over as text-only so they
 * are never candidates.
 */
export function opencodeToLibrary(messages: readonly OpencodeMessage[]): Message[] {
  return messages.map((message) => {
    const role = message.info.role === 'assistant' ? 'assistant' : 'user';
    const text = message.parts
      .filter((part): part is OpencodeTextPart => part.type === 'text')
      .map((part) => part.text)
      .join('\n');
    const library: Message = { role, text, toolUses: [] };
    const results: Message['toolResults'] = [];
    for (const part of message.parts) {
      if (!isToolPart(part)) continue;
      const out = toolOutput(part.state);
      if (!out) continue;
      const input =
        part.state.input && typeof part.state.input === 'object' ? part.state.input : {};
      library.toolUses.push({
        tool_use_id: part.callID,
        tool: part.tool,
        input,
        text: out.text,
        isError: out.isError,
      });
      results.push({ tool_use_id: part.callID, text: out.text, isError: out.isError });
    }
    if (results.length > 0) library.toolResults = results;
    return library;
  });
}

export interface OpencodeConfig {
  apiKey?: string;
  model?: string;
  keepThreshold?: number;
  preserveRecentMessages?: number;
  maxStateTokens?: number;
  maxRequestTokens?: number;
  truncateHeadChars?: number;
  /** Below this estimated reduction the compaction hook falls back to the default prompt. */
  minReductionRatio?: number;
  /** Fewer completed tool calls than this skips Jev entirely (fail-cheap). */
  pruneMinToolCalls?: number;
  /** When true the compacting hook replaces the summary prompt instead of appending context. */
  replacePrompt?: boolean;
  /** Master switch; when false the plugin registers no-op hooks. */
  enabled?: boolean;
}

export interface ResolvedOpencodeConfig {
  apiKey?: string;
  model: string;
  keepThreshold: number;
  preserveRecentMessages: number;
  maxStateTokens: number;
  maxRequestTokens: number;
  truncateHeadChars: number;
  minReductionRatio: number;
  pruneMinToolCalls: number;
  replacePrompt: boolean;
  enabled: boolean;
}

function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOpencodeConfig(raw: Record<string, unknown> = {}): ResolvedOpencodeConfig {
  const apiKey = typeof raw.apiKey === 'string' && raw.apiKey.length > 0 ? raw.apiKey : undefined;
  const model = typeof raw.model === 'string' && raw.model.length > 0 ? raw.model : 'jev-latest';
  return {
    apiKey,
    model,
    keepThreshold: finite(raw.keepThreshold, 0.5),
    preserveRecentMessages: Math.max(0, Math.floor(finite(raw.preserveRecentMessages, 6))),
    maxStateTokens: Math.max(1, finite(raw.maxStateTokens, 25_000)),
    maxRequestTokens: Math.max(1, finite(raw.maxRequestTokens, 30_000)),
    truncateHeadChars: Math.max(0, Math.floor(finite(raw.truncateHeadChars, 300))),
    minReductionRatio: finite(raw.minReductionRatio, 0.25),
    pruneMinToolCalls: Math.max(0, Math.floor(finite(raw.pruneMinToolCalls, 8))),
    replacePrompt: raw.replacePrompt === true,
    enabled: raw.enabled !== false,
  };
}

/** Stable cache key for a set of tool calls: ids plus result sizes. */
export function fingerprintCalls(calls: readonly ToolCall[]): string {
  return calls.map((call) => `${call.tool_use_id}:${call.resultChars}`).join('|');
}

export function countCompletedToolParts(messages: readonly OpencodeMessage[]): number {
  let count = 0;
  for (const message of messages) {
    for (const part of message.parts) {
      if (isToolPart(part) && toolOutput(part.state)) count += 1;
    }
  }
  return count;
}

function truncatedOutputText(text: string, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-jev-compaction truncated ${text.length - headChars} chars of this tool result; re-run the tool if needed]`;
}

export interface PruneStats {
  truncated: number;
  removed: number;
  messagesRemoved: number;
}

/**
 * Applies Jev decisions onto opencode messages in place: `drop_result`
 * truncates the tool output to a bounded head plus note, `drop_call` removes
 * the tool part, and messages left with zero parts are removed. Every
 * mutation uses `splice` or direct field writes so the host observes it.
 */
export function pruneOpencodeMessagesInPlace(
  messages: OpencodeMessage[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
): PruneStats {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallDecision['action']>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
  }
  if (actions.size === 0) return { truncated: 0, removed: 0, messagesRemoved: 0 };
  const stats: PruneStats = { truncated: 0, removed: 0, messagesRemoved: 0 };

  for (const message of messages) {
    const keep: OpencodePart[] = [];
    for (const part of message.parts) {
      if (!isToolPart(part)) {
        keep.push(part);
        continue;
      }
      const action = actions.get(part.callID);
      if (action === 'drop_call') {
        stats.removed += 1;
        continue;
      }
      if (action === 'drop_result') {
        const out = toolOutput(part.state);
        if (out) {
          const next = truncatedOutputText(out.text, headChars);
          if (next !== out.text) {
            if (part.state.status === 'completed') part.state.output = next;
            else part.state.error = next;
            if (part.state.time) part.state.time.compacted = Date.now();
            else part.state.time = { start: Date.now(), compacted: Date.now() };
            stats.truncated += 1;
          }
        }
        keep.push(part);
        continue;
      }
      keep.push(part);
    }
    if (keep.length !== message.parts.length) {
      message.parts.splice(0, message.parts.length, ...keep);
    }
  }

  const remaining = messages.filter((message) => message.parts.length > 0);
  if (remaining.length !== messages.length) {
    stats.messagesRemoved = messages.length - remaining.length;
    messages.splice(0, messages.length, ...remaining);
  }
  return stats;
}

export function summarizeOpencodeResult(result: CompactResult): string {
  const { stats } = result;
  const percent = `${Math.round(reductionRatio(result) * 100)}%`;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  return `${percent} reduction; ${parts.join(', ') || 'no tool calls'}; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

/**
 * Context injected into `experimental.session.compacting`: the Jev verdict
 * summary plus the per-call actions, so the summarizer omits what Jev
 * already judged droppable instead of re-summarizing stale tool outputs.
 */
export function buildCompactingContext(result: CompactResult): string {
  const lines = result.decisions
    .filter((d) => d.action !== 'keep' || d.reason !== 'pinned')
    .map((d) => `- ${d.id} ${d.tool}: ${d.action} (call=${d.keepCall.toFixed(2)}, result=${d.keepResult.toFixed(2)})`);
  return [
    'fast-jev-compaction (Jev verbatim pass): tool outputs below were already judged.',
    `Summary: ${summarizeOpencodeResult(result)}.`,
    ...lines,
    'Omit dropped calls/results from the summary; keep kept ones verbatim and short.',
  ].join('\n');
}

/** Generic guidance when no Jev pass has run yet for the session history. */
export function buildCompactingFallbackContext(): string {
  return [
    'fast-jev-compaction: no Jev pass ran for this history (too few tool calls,',
    'missing TYPESAFE_API_KEY, or a Jev failure). Summarize as usual, but prefer',
    'keeping exact file paths, commands, error text, and constraints verbatim;',
    'drop stale tool outputs that the next steps no longer need.',
  ].join(' ');
}

/**
 * Full replacement prompt for `experimental.session.compacting` when
 * `replacePrompt` is enabled: verbatim-preserving compaction guided by the
 * Jev decisions instead of a free-form summary.
 */
export function buildVerbatimPrompt(result: CompactResult): string {
  return [
    'You are compacting a coding session. Do NOT paraphrase tool outputs.',
    'Carry forward verbatim: the current task/goal, exact file paths, commands,',
    'constraints, and error text the next steps still need.',
    `Jev verdicts for this history (${summarizeOpencodeResult(result)}):`,
    ...result.decisions.map((d) => `- ${d.id} ${d.tool}: ${d.action}`),
    'Rules: omit every drop_call item entirely; for drop_result keep at most the',
    'first line plus why it is no longer needed; keep every keep item verbatim',
    'but short. Format with ## Objective, ## Kept context, and ## Next steps.',
  ].join('\n');
}
