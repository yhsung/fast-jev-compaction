// fast-jev-compaction opencode plugin.
//
// Verbatim, Jev-guided context pruning for opencode: every completed tool
// call is scored by TypeSafe's Jev model (keep the call? keep its output?),
// stale outputs are truncated or removed, everything kept stays verbatim.
// No summaries are ever written into your context.
//
// How it works:
// - `experimental.chat.messages.transform` runs the library over the request
//   payload (an in-memory copy) and truncates/removes tool outputs Jev says
//   are no longer needed. Stored session history is untouched; only the
//   tokens sent to the LLM shrink. Runs at most once per history shape
//   (fingerprinted cache) and fails open.
// - `experimental.session.compacting` injects the Jev verdicts into the
//   compaction prompt so the summarizer omits what Jev already dropped.
//   Set `replacePrompt: true` to replace the summary prompt with a
//   verbatim-preserving one instead.
//
// Install (this repo): open the repo in opencode; `.opencode/plugins/` loads
// automatically. Set `TYPESAFE_API_KEY` in the environment.
// Install (any other project): copy this file to
// `~/.config/opencode/plugins/fast-jev-compaction.ts` (global) or
// `.opencode/plugins/` (project), then change the relative `../../src/`
// imports below to `fast-jev-compaction` and `npm install fast-jev-compaction`.
//
// Options (plugin options or opencode.json `plugin` entry args):
//   apiKey, model, keepThreshold, preserveRecentMessages, maxStateTokens,
//   maxRequestTokens, truncateHeadChars, minReductionRatio,
//   pruneMinToolCalls (default 8), replacePrompt (default false),
//   enabled (default true).
//
// IMPORTANT: opencode keeps its own array reference after this hook, so all
// mutations below are in place (`splice`, direct field writes). Reassigning
// `output.messages` would be a silent no-op.

import type { Plugin } from '@opencode-ai/plugin';
import { collectToolCalls, compact, reductionRatio } from '../../src/compact.ts';
import { JevClient } from '../../src/client.ts';
import {
  buildCompactingContext,
  buildCompactingFallbackContext,
  buildVerbatimPrompt,
  countCompletedToolParts,
  fingerprintCalls,
  opencodeToLibrary,
  pruneOpencodeMessagesInPlace,
  resolveOpencodeConfig,
  summarizeOpencodeResult,
  type OpencodeMessage,
} from '../../src/opencode.ts';
import type { CompactResult, ToolCall } from '../../src/types.ts';

const CACHE_LIMIT = 5;

export const FastJevCompaction: Plugin = async ({ client }, rawOptions) => {
  const fromEnv =
    typeof process !== 'undefined' ? process.env?.TYPESAFE_API_KEY : undefined;
  const config = resolveOpencodeConfig({
    ...((rawOptions ?? {}) as Record<string, unknown>),
    apiKey:
      (rawOptions as Record<string, unknown> | undefined)?.apiKey ?? fromEnv,
  });

  // Fingerprint -> { result, calls } so repeated transforms of an unchanged
  // history reuse Jev verdicts without another request. Call ids (`t1…`)
  // are positional, and the fingerprint covers ids plus result sizes, so
  // cached decisions align with freshly collected calls on a cache hit.
  const cache = new Map<string, { result: CompactResult; calls: ToolCall[] }>();
  let latest: CompactResult | undefined;

  async function log(level: 'info' | 'warn' | 'error', message: string): Promise<void> {
    try {
      await client.app.log({ body: { service: 'fast-jev-compaction', level, message } });
    } catch {
      // Logging must never break the session.
    }
  }

  function remember(key: string, result: CompactResult, calls: ToolCall[]): void {
    cache.set(key, { result, calls });
    if (cache.size > CACHE_LIMIT) {
      const oldest = cache.keys().next();
      if (!oldest.done) cache.delete(oldest.value);
    }
    latest = result;
  }

  if (!config.enabled) return {};

  return {
    'experimental.chat.messages.transform': async (_input, output) => {
      try {
        const messages = output.messages as OpencodeMessage[];
        if (countCompletedToolParts(messages) < config.pruneMinToolCalls) return;
        const library = opencodeToLibrary(messages);
        const calls = collectToolCalls(library, config.preserveRecentMessages);
        if (calls.every((call) => call.pinned)) return;
        const key = fingerprintCalls(calls);
        let hit = cache.get(key);
        if (!hit) {
          if (!config.apiKey && !fromEnv) return;
          const asker = new JevClient({ apiKey: config.apiKey ?? fromEnv, model: config.model });
          const result = await compact(library, asker, {
            keepThreshold: config.keepThreshold,
            preserveRecentMessages: config.preserveRecentMessages,
            maxStateTokens: config.maxStateTokens,
            maxRequestTokens: config.maxRequestTokens,
            truncateHeadChars: config.truncateHeadChars,
          });
          hit = { result, calls };
          remember(key, result, calls);
        }
        // Below-minimum reduction: leave the payload alone, like the
        // Claude Code hook falling back to the built-in summary.
        if (reductionRatio(hit.result) < config.minReductionRatio) return;
        const stats = pruneOpencodeMessagesInPlace(
          messages,
          hit.result.decisions,
          hit.calls,
          config.truncateHeadChars,
        );
        if (stats.truncated + stats.removed > 0) {
          await log(
            'info',
            `pruned context: ${stats.truncated} outputs truncated, ${stats.removed} calls removed (${summarizeOpencodeResult(hit.result)})`,
          );
        }
      } catch (error) {
        await log(
          'warn',
          `Jev pass skipped (${error instanceof Error ? error.message : String(error)})`,
        );
      }
    },

    'experimental.session.compacting': async (_input, output) => {
      if (latest) {
        output.context.push(buildCompactingContext(latest));
        if (config.replacePrompt) output.prompt = buildVerbatimPrompt(latest);
      } else {
        output.context.push(buildCompactingFallbackContext());
      }
    },
  };
};
