import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useResearch } from "../stores/research";
import {
  sendResearch,
  setResearchSession,
  stopResearch,
} from "./research-stream";
import type { WorkflowEvent } from "../types";

const pending: Promise<void>[] = [];

it("keeps a visible draft stable through validation correction and publishes the final once", () => {
  const receive = useResearch.getState().receiveEvent;
  receive({ type: "start", conversation_id: "conversation", run_id: "run" });
  receive({ type: "answer_reset", attempt: 1 });
  receive({ type: "token", text: "Initial answer [W1]." });
  receive({ type: "answer_reset", attempt: 2 });
  receive({ type: "answer_reset", attempt: 2, model: "replacement" });
  receive({ type: "token", text: "Corrected grounded answer [W1]." });
  expect(useResearch.getState().draft).toBe("Initial answer [W1].");
  expect(useResearch.getState().correctingDraft).toBe(true);
  const message = {
    id: "final",
    role: "assistant" as const,
    content: "Corrected grounded answer [W1].",
    sources: [],
  };
  receive({ type: "final", message });
  receive({ type: "final", message }); // Reconnection may replay the same publication.
  expect(
    useResearch.getState().messages.filter((item) => item.id === "final"),
  ).toHaveLength(1);
  expect(useResearch.getState().draft).toBe("");
  expect(useResearch.getState().correctingDraft).toBe(false);
});

it("still discards partial tokens on provider fallback during the initial generation", () => {
  const receive = useResearch.getState().receiveEvent;
  receive({ type: "answer_reset", attempt: 1, model: "first" });
  receive({ type: "token", text: "Incomplete draft" });
  receive({ type: "answer_reset", attempt: 1, model: "second" });
  receive({ type: "token", text: "Complete replacement" });
  expect(useResearch.getState().draft).toBe("Complete replacement");
  expect(useResearch.getState().correctingDraft).toBe(false);
});

beforeEach(() => {
  setResearchSession("user_alice");
  useResearch.getState().update({
    workspaceId: "workspace",
    selectedModel: "gemini-test",
    relevanceThreshold: 0.75,
  });
});

afterEach(async () => {
  setResearchSession("");
  await Promise.allSettled(pending.splice(0));
  vi.restoreAllMocks();
});

function openStream() {
  let frames: ReadableStreamDefaultController<Uint8Array>;
  let signal: AbortSignal;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      frames = controller;
    },
  });
  const fetcher = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (url, init) => {
      if (String(url).endsWith("/chat/stream")) {
        signal = init!.signal as AbortSignal;
        signal.addEventListener(
          "abort",
          () => {
            frames.error(new DOMException("Stopped", "AbortError"));
          },
          { once: true },
        );
        return new Response(stream);
      }
      return Response.json(
        String(url).endsWith("/conversations")
          ? []
          : { messages: useResearch.getState().messages },
      );
    });
  return {
    fetcher,
    signal: () => signal,
    emit(event: WorkflowEvent) {
      frames.enqueue(
        new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`),
      );
    },
    close() {
      frames.close();
    },
  };
}

async function start(stream: ReturnType<typeof openStream>) {
  const running = sendResearch("A question");
  pending.push(running);
  stream.emit({
    type: "start",
    conversation_id: "conversation",
    run_id: "run",
  });
  await vi.waitFor(() =>
    expect(useResearch.getState().conversationId).toBe("conversation"),
  );
  return { running };
}

it("continues across view switches and same-account page remounts", async () => {
  const stream = openStream();
  const { running } = await start(stream);
  useResearch.getState().update({ view: "documents" });
  setResearchSession("user_alice"); // Returning to the research page.
  expect(stream.signal().aborted).toBe(false);
  expect(useResearch.getState().streaming).toBe(true);
  expect(useResearch.getState().relevanceThreshold).toBe(0.75);
  stream.emit({ type: "token", text: "Background evidence" });
  await vi.waitFor(() =>
    expect(useResearch.getState().draft).toBe("Background evidence"),
  );
  useResearch.getState().update({ view: "research" });
  stream.emit({
    type: "final",
    message: {
      id: "answer",
      role: "assistant",
      content: "Completed answer",
      sources: [],
    },
  });
  stream.emit({ type: "done" });
  stream.close();
  await running;
  expect(useResearch.getState().streaming).toBe(false);
  expect(useResearch.getState().messages.at(-1)?.content).toBe(
    "Completed answer",
  );
  expect(useResearch.getState().error).toBe("");
});

it("allows the remounted chat's Stop button to cancel the original stream", async () => {
  const stream = openStream();
  const { running } = await start(stream);
  useResearch.getState().update({ view: "documents" });
  stopResearch();
  await running;
  expect(stream.signal().aborted).toBe(true);
  expect(useResearch.getState().streaming).toBe(false);
  expect(useResearch.getState().draft).toBe("");
  expect(useResearch.getState().error).toContain("Cancellation requested");
  expect(
    useResearch
      .getState()
      .messages.some((message) => message.role === "assistant"),
  ).toBe(false);
});

it.each(["", "user_bob"])(
  "cancels and clears private state on auth change to %s",
  async (userId) => {
    const stream = openStream();
    const { running } = await start(stream);
    setResearchSession(userId);
    await running;
    expect(stream.signal().aborted).toBe(true);
    expect(useResearch.getState().sessionId).toBe(userId);
    expect(useResearch.getState().messages).toEqual([]);
    expect(useResearch.getState().events).toEqual([]);
    expect(useResearch.getState().error).toBe("");
    expect(useResearch.getState().streaming).toBe(false);
  },
);

it("prevents a second chat instance from starting an overlapping request", async () => {
  const stream = openStream();
  const { running } = await start(stream);
  await sendResearch("Another question");
  expect(stream.fetcher).toHaveBeenCalledTimes(1);
  expect(
    JSON.parse(stream.fetcher.mock.calls[0][1]!.body as string)
      .relevance_threshold,
  ).toBe(0.75);
  stopResearch();
  await running;
});

it("does not publish a delayed refresh into another account", async () => {
  const stream = openStream();
  const { running } = await start(stream);
  let resolveRefresh!: (response: Response) => void;
  stream.fetcher.mockImplementationOnce(
    () =>
      new Promise<Response>((resolve) => {
        resolveRefresh = resolve;
      }),
  );
  stream.emit({ type: "done" });
  stream.close();
  await vi.waitFor(() => expect(resolveRefresh).toBeDefined());
  setResearchSession("user_bob");
  resolveRefresh(
    Response.json([{ id: "private-conversation", title: "Private title" }]),
  );
  await running;
  expect(useResearch.getState().conversations).toEqual([]);
  expect(useResearch.getState().messages).toEqual([]);
  expect(stream.fetcher).toHaveBeenCalledTimes(3);
});

it("does not send research without an authenticated session", async () => {
  const stream = openStream();
  setResearchSession("");
  useResearch
    .getState()
    .update({ workspaceId: "workspace", selectedModel: "gemini-test" });
  await sendResearch("Question");
  expect(stream.fetcher).not.toHaveBeenCalled();
});

it("refreshes the conversation list and saved messages concurrently", async () => {
  const stream = openStream();
  const { running } = await start(stream);
  let resolveList!: (value: Response) => void;
  let resolveChat!: (value: Response) => void;
  stream.fetcher
    .mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          resolveList = resolve;
        }),
    )
    .mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          resolveChat = resolve;
        }),
    );
  stream.emit({ type: "done" });
  stream.close();
  await vi.waitFor(() => expect(stream.fetcher).toHaveBeenCalledTimes(3));
  expect(resolveList).toBeDefined();
  expect(resolveChat).toBeDefined();
  resolveList(Response.json([]));
  resolveChat(Response.json({ messages: [] }));
  await running;
  expect(useResearch.getState().error).toBe("");
});
