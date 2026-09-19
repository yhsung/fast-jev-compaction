import { describe, expect, it } from 'vitest';
import { collectToolCalls, compact, decideCall, reductionRatio } from '../src/index.js';
import {
  buildCompactingContext,
  buildCompactingFallbackContext,
  buildVerbatimPrompt,
  countCompletedToolParts,
  fingerprintCalls,
  isToolPart,
  opencodeToLibrary,
  pruneOpencodeMessagesInPlace,
  resolveOpencodeConfig,
  summarizeOpencodeResult,
  type OpencodeMessage,
} from '../src/opencode.js';
import type { JevAsker, JevQuestions } from '../src/index.js';

function textMsg(role: 'user' | 'assistant', text: string): OpencodeMessage {
  return { info: { role, id: `m-${text.slice(0, 8)}` }, parts: [{ type: 'text', text }] };
}

function toolMsg(
  callID: string,
  tool: string,
  input: Record<string, unknown>,
  output: string,
  status: 'completed' | 'error' | 'pending' = 'completed',
): OpencodeMessage {
  const state =
    status === 'completed'
      ? { status, input, output, title: tool, metadata: {}, time: { start: 1, end: 2 } }
      : status === 'error'
        ? { status, input, error: output, metadata: {}, time: { start: 1, end: 2 } }
        : { status, input, raw: '' };
  return {
    info: { role: 'assistant', id: `m-${callID}` },
    parts: [{ type: 'tool', id: `p-${callID}`, callID, tool, state } as OpencodeMessage['parts'][number]],
  };
}

const big = 'x'.repeat(2000);

function history(): OpencodeMessage[] {
  return [
    textMsg('user', 'Fix the failing test.'),
    toolMsg('call-1', 'Read', { file_path: 'src/a.ts' }, big),
    toolMsg('call-2', 'Bash', { command: 'npm test' }, 'FAIL: expected 2 to be 3', 'error'),
    textMsg('assistant', 'Fixing now.'),
    textMsg('user', 'go ahead'),
  ];
}

function fakeJev(answer: (name: string) => number): JevAsker {
  return {
    async ask(state, questions: JevQuestions) {
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: answer(key) }]),
        ),
      };
    },
  };
}

describe('opencode config', () => {
  it('resolves defaults and reads overrides', () => {
    expect(resolveOpencodeConfig()).toMatchObject({
      model: 'jev-latest',
      keepThreshold: 0.5,
      preserveRecentMessages: 6,
      maxStateTokens: 25_000,
      maxRequestTokens: 30_000,
      truncateHeadChars: 300,
      minReductionRatio: 0.25,
      pruneMinToolCalls: 8,
      replacePrompt: false,
      enabled: true,
    });
    expect(
      resolveOpencodeConfig({ apiKey: 'k', pruneMinToolCalls: 2, replacePrompt: true, enabled: false }),
    ).toMatchObject({ apiKey: 'k', pruneMinToolCalls: 2, replacePrompt: true, enabled: false });
    expect(resolveOpencodeConfig({ keepThreshold: Number.NaN }).keepThreshold).toBe(0.5);
  });
});

describe('opencode message mapping', () => {
  it('joins text parts and pairs tool calls with their outputs', () => {
    const library = opencodeToLibrary(history());
    expect(library.map((m) => m.role)).toEqual(['user', 'assistant', 'assistant', 'assistant', 'user']);
    expect(library[0]?.text).toBe('Fix the failing test.');
    expect(library[1]?.toolUses[0]).toMatchObject({ tool_use_id: 'call-1', tool: 'Read' });
    expect(library[1]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'call-1', isError: false });
    expect(library[1]?.toolResults?.[0]?.text).toHaveLength(big.length);
    expect(library[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'call-2', isError: true });
    const calls = collectToolCalls(library, 0);
    expect(calls.map((c) => [c.id, c.tool, c.callIndex, c.resultIndex])).toEqual([
      ['t1', 'Read', 1, 1],
      ['t2', 'Bash', 2, 2],
    ]);
    expect(fingerprintCalls(calls)).toBe(`call-1:${big.length}|call-2:24`);
  });

  it('skips pending tool parts and counts only completed outputs', () => {
    const messages = [toolMsg('call-9', 'Read', {}, '', 'pending')];
    expect(opencodeToLibrary(messages)[0]?.toolUses).toHaveLength(0);
    expect(countCompletedToolParts(messages)).toBe(0);
    expect(countCompletedToolParts(history())).toBe(2);
  });
});

describe('opencode pruning', () => {
  it('removes dropped calls and truncates dropped results in place', async () => {
    const messages = history();
    // Identity witnesses: the host only observes in-place mutation.
    const userParts = messages[0]?.parts;
    const bashParts = messages[2]?.parts;
    const library = opencodeToLibrary(messages);
    const calls = collectToolCalls(library, 1);
    const result = await compact(
      library,
      fakeJev((name) => (name === 'call_t1' || name === 'result_t1' ? 0.1 : 0.9)),
      { preserveRecentMessages: 1 },
    );
    expect(result.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    const stats = pruneOpencodeMessagesInPlace(messages, result.decisions, calls, 300);
    expect(stats).toMatchObject({ removed: 1, truncated: 0 });
    // Same array references: the host observes in-place mutation.
    expect(messages[0]?.parts).toBe(userParts);
    const bash = messages.find((m) =>
      m.parts.some((p) => isToolPart(p) && p.callID === 'call-2'),
    );
    expect(bash?.parts).toBe(bashParts);
    expect(messages.filter((m) => m.parts.length === 0)).toHaveLength(0);
    expect(messages).toHaveLength(4);
  });

  it('truncates dropped results to a bounded head plus note', async () => {
    const messages = history();
    const library = opencodeToLibrary(messages);
    const calls = collectToolCalls(library, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const stats = pruneOpencodeMessagesInPlace(messages, decisions, calls, 300);
    expect(stats).toMatchObject({ truncated: 1, removed: 0 });
    const part = messages[1]?.parts[0];
    expect(isToolPart(part!)).toBe(true);
    if (isToolPart(part!)) {
      const output = (part.state as { output?: string }).output ?? '';
      expect(output.startsWith('x'.repeat(300))).toBe(true);
      expect(output).toMatch(/\[fast-jev-compaction truncated 1700 chars/);
      expect(part.state.time?.compacted).toBeTypeOf('number');
    }
  });

  it('leaves short dropped results untouched', () => {
    const messages = [toolMsg('call-1', 'Read', {}, 'y'.repeat(100))];
    const library = opencodeToLibrary(messages);
    const calls = collectToolCalls(library, 0);
    const decisions = [decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 })];
    const stats = pruneOpencodeMessagesInPlace(messages, decisions, calls, 300);
    expect(stats).toMatchObject({ truncated: 0, removed: 0 });
    const part = messages[0]?.parts[0];
    if (isToolPart(part!)) {
      expect((part.state as { output?: string }).output).toHaveLength(100);
    }
  });
});

describe('opencode compaction prompt', () => {
  it('builds context and verbatim prompt from a Jev result', async () => {
    const library = opencodeToLibrary(history());
    const result = await compact(library, fakeJev(() => 0.1), { preserveRecentMessages: 1 });
    expect(reductionRatio(result)).toBeGreaterThan(0);
    expect(summarizeOpencodeResult(result)).toMatch(/reduction; .*call_dropped/);
    expect(buildCompactingContext(result)).toMatch(/t1 Read: drop_call/);
    expect(buildVerbatimPrompt(result)).toMatch(/Do NOT paraphrase/);
    expect(buildCompactingFallbackContext()).toMatch(/TYPESAFE_API_KEY/);
  });
});
