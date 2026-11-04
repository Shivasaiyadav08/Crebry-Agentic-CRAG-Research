"use client";

import { API_URL, api, readSSE } from "./api";
import { useResearch } from "../stores/research";
import type { Conversation, Message, WorkflowRun } from "../types";

// Owned by the signed-in browser session, independent of mounted views/pages.
let activeRun: { controller: AbortController; sequence: number } | null = null;
let sequence = 0;

useResearch.subscribe((state, previous) => {
  if (
    typeof window !== "undefined" &&
    state.sessionId &&
    state.sessionId === previous.sessionId &&
    state.workspaceId === previous.workspaceId &&
    state.conversationId !== previous.conversationId
  ) {
    const key = `crebry_chat_${state.sessionId}_${state.workspaceId}`;
    if (state.conversationId) localStorage.setItem(key, state.conversationId);
    else localStorage.removeItem(key);
  }
  if (state.sessionId === previous.sessionId) return;
  sequence += 1;
  activeRun?.controller.abort();
  activeRun = null;
});

export async function resumeResearch(runId: string) {
  const state = useResearch.getState();
  if (activeRun || !state.sessionId || !state.workspaceId) return;
  const run = { controller: new AbortController(), sequence: ++sequence };
  activeRun = run;
  const current = () =>
    sequence === run.sequence &&
    useResearch.getState().sessionId === state.sessionId;
  state.update({
    streaming: true,
    draft: "",
    correctingDraft: false,
    error: "",
    events: [],
    runId,
  });
  try {
    const response = await fetch(
      `${API_URL}/workspaces/${state.workspaceId}/workflows/${runId}/events`,
      { signal: run.controller.signal, cache: "no-store" },
    );
    if (!response.ok)
      throw new Error(
        "Could not reconnect to saved research. Try again after signing in.",
      );
    await readSSE(response, (event) => {
      if (current()) useResearch.getState().receiveEvent(event);
    });
  } catch (error) {
    if (current())
      state.update({
        error:
          error instanceof Error
            ? error.message
            : "Research connection interrupted. Progress is saved.",
        draft: "",
      });
  } finally {
    if (activeRun === run) activeRun = null;
    if (current()) state.update({ streaming: false });
  }
}

export async function restoreResearchConversation(
  workspaceId: string,
  conversationId: string,
) {
  const sessionId = useResearch.getState().sessionId;
  const chat = await api<{
    messages: Message[];
    latest_run: { id: string; status: string; resumable: boolean } | null;
  }>(`/workspaces/${workspaceId}/conversations/${conversationId}`);
  const run = chat.latest_run
    ? await api<WorkflowRun>(
        `/workspaces/${workspaceId}/workflows/${chat.latest_run.id}`,
      )
    : null;
  const state = useResearch.getState();
  if (
    state.sessionId !== sessionId ||
    state.workspaceId !== workspaceId ||
    state.streaming
  )
    return;
  state.update({
    view: "research",
    conversationId,
    messages: chat.messages,
    draft: "",
    correctingDraft: false,
    runId: chat.latest_run?.resumable ? (run?.id ?? null) : null,
    events: run ? [{ type: "start", run_id: run.id }, ...run.events] : [],
    elapsedMs: run?.duration_ms ?? 0,
    warnings:
      run?.events.findLast((event) => event.type === "final")?.warnings ?? [],
    error: run?.error ?? "",
  });
  if (
    run &&
    ["queued", "running"].includes(run.status) &&
    chat.latest_run?.resumable
  )
    void resumeResearch(run.id);
}

export async function recoverResearch() {
  const state = useResearch.getState();
  if (!state.runId || !state.conversationId || state.streaming) return;
  try {
    const run = await api<WorkflowRun>(
      `/workspaces/${state.workspaceId}/workflows/${state.runId}`,
    );
    if (useResearch.getState().sessionId !== state.sessionId) return;
    if (["failed", "cancelled", "interrupted"].includes(run.status))
      await api(
        `/workspaces/${state.workspaceId}/workflows/${state.runId}/retry`,
        { method: "POST" },
      );
    if (["completed", "validation_failed"].includes(run.status))
      await restoreResearchConversation(
        state.workspaceId,
        state.conversationId,
      );
    else await resumeResearch(state.runId);
  } catch (error) {
    if (useResearch.getState().sessionId === state.sessionId)
      state.update({
        error:
          error instanceof Error
            ? error.message
            : "Could not retry saved research.",
      });
  }
}

export function setResearchSession(sessionId: string) {
  if (useResearch.getState().sessionId === sessionId) return;
  useResearch.setState({ ...useResearch.getInitialState(), sessionId }, true);
}

export function stopResearch() {
  const state = useResearch.getState();
  if (state.runId && state.workspaceId)
    void api(
      `/workspaces/${state.workspaceId}/workflows/${state.runId}/cancel`,
      { method: "POST" },
    ).catch((error) => state.update({ error: error.message }));
  activeRun?.controller.abort();
}

export async function sendResearch(question: string, regenerate = false) {
  const state = useResearch.getState();
  if (
    activeRun ||
    state.streaming ||
    !state.sessionId ||
    !state.workspaceId ||
    !state.selectedModel ||
    !question.trim()
  )
    return;
  const oldMessages = state.messages;
  const lastUser = oldMessages.findLastIndex(
    (message) => message.role === "user",
  );
  const visible = regenerate
    ? oldMessages.slice(0, lastUser + 1)
    : [
        ...oldMessages,
        {
          id: crypto.randomUUID(),
          role: "user" as const,
          content: question.trim(),
          sources: [],
        },
      ];
  state.update({
    streaming: true,
    draft: "",
    correctingDraft: false,
    error: "",
    events: [],
    warnings: [],
    messages: visible,
    elapsedMs: 0,
    modelNotice: "",
  });
  const run = { controller: new AbortController(), sequence: ++sequence };
  activeRun = run;
  const isCurrent = () =>
    sequence === run.sequence &&
    useResearch.getState().sessionId === state.sessionId;
  let receivedStart = false;
  let receivedFinal = false;
  try {
    const response = await fetch(`${API_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: run.controller.signal,
      body: JSON.stringify({
        workspace_id: state.workspaceId,
        conversation_id: state.conversationId,
        question: question.trim(),
        regenerate,
        model: state.selectedModel,
        automatic_fallback: state.automaticFallback,
        idempotency_key: crypto.randomUUID(),
        relevance_threshold: state.relevanceThreshold,
      }),
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(
        typeof body.detail === "string"
          ? body.detail
          : `Request failed (${response.status}).`,
      );
    }
    await readSSE(response, (event) => {
      if (!isCurrent()) return;
      if (event.type === "start") receivedStart = true;
      if (event.type === "final") receivedFinal = true;
      useResearch.getState().receiveEvent(event);
    });
  } catch (error) {
    if (!isCurrent()) return;
    const stopped = error instanceof Error && error.name === "AbortError";
    useResearch.getState().update({
      error: stopped
        ? "Cancellation requested. Completed research steps are saved."
        : error instanceof Error
          ? error.message
          : "Could not complete the request.",
      draft: "",
      ...(!receivedStart || (regenerate && !receivedFinal)
        ? { messages: oldMessages }
        : {}),
    });
  } finally {
    if (activeRun === run) activeRun = null;
    if (isCurrent()) {
      const completedConversationId = useResearch.getState().conversationId;
      useResearch.getState().update({ streaming: false });
      if (receivedStart) {
        try {
          const [conversations, chat] = await Promise.all([
            api<Conversation[]>(
              `/workspaces/${state.workspaceId}/conversations`,
            ),
            api<{ messages: Message[] }>(
              `/workspaces/${state.workspaceId}/conversations/${completedConversationId}`,
            ),
          ]);
          if (isCurrent()) {
            const current = useResearch.getState();
            if (isCurrent() && current.workspaceId === state.workspaceId) {
              current.update({ conversations });
              if (current.conversationId === completedConversationId)
                current.update({ messages: chat.messages });
            }
          }
        } catch {
          /* Keep already received text if a follow-up refresh fails. */
        }
      }
    }
  }
}
