import { beforeEach, expect, it, vi } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { Chat } from "../components/chat";

const context = vi.hoisted(() => ({ state: {} as Record<string, unknown> }));
vi.mock("@/stores/research", () => ({
  useResearch: Object.assign(
    (selector?: (state: Record<string, unknown>) => unknown) =>
      selector ? selector(context.state) : context.state,
    { getState: () => context.state },
  ),
}));
vi.mock("@/lib/research-stream", () => ({
  recoverResearch: vi.fn(),
  sendResearch: vi.fn(),
  stopResearch: vi.fn(),
}));

beforeEach(() => {
  context.state = {
    messages: [
      {
        id: "question",
        role: "user",
        content: "who is captain of brazil football team",
        sources: [],
      },
    ],
    draft: "",
    correctingDraft: false,
    streaming: false,
    health: { gemini_configured: true, tavily_configured: true },
    error: "",
    warnings: [],
    events: [],
    workspaces: [],
    workspaceId: "workspace",
    conversations: [],
    runId: null,
    selectedModel: "gemini-3.5-flash-lite",
    actualModel: "",
    automaticFallback: false,
    models: [
      {
        id: "gemini-3.5-flash-lite",
        name: "Gemini 3.5 Flash Lite",
        available: true,
      },
    ],
    relevanceThreshold: 0.5,
    modelNotice: "",
    update: vi.fn(),
  };
});

it("renders a typed question once without duplicate Y/You labels or changing its content", () => {
  const html = renderToStaticMarkup(createElement(Chat));
  expect(html.match(/who is captain of brazil football team/g)).toHaveLength(1);
  expect(html).not.toContain(">You<");
  expect(html).not.toContain(">Y<");
  expect(html).toContain('translate="no"');
});

it("keeps removed models out of the UI and gives fallback controls their own layout", () => {
  context.state.selectedModel = "gemini-3.8-flash";
  (context.state.models as object[]).push(
    { id: "gemini-3.8-flash", name: "Gemini 3.8 Flash" },
    { id: "gemini-3.6-flash", name: "Gemini 3.6 Flash" },
  );
  const html = renderToStaticMarkup(createElement(Chat));
  expect(html).not.toContain("gemini-3.8-flash");
  expect(html).not.toContain("gemini-3.6-flash");
  expect(html).toContain("model-fallback-control");
});

it("shows missing web search visibly and preserves web citation rendering", () => {
  context.state.selectedModel = "auto";
  context.state.actualModel = "gemini-3.5-flash-lite";
  context.state.automaticFallback = true;
  context.state.health = { gemini_configured: true, tavily_configured: false };
  (context.state.messages as object[]).push({
    id: "answer",
    role: "assistant",
    content:
      "Based on the provided documents, web evidence supports this claim [W1, W2].",
    sources: [
      {
        id: "W1",
        kind: "web",
        name: "Match report",
        text: "Evidence",
        url: "https://example.org/report",
      },
      {
        id: "W2",
        kind: "web",
        name: "Second report",
        text: "Supporting evidence",
        url: "https://example.org/second",
      },
    ],
  });
  const html = renderToStaticMarkup(createElement(Chat));
  expect(html).toContain("Web search is not connected");
  expect(html).toContain('aria-label="View source W1"');
  expect(html).toContain('aria-label="View source W2"');
  expect(html).toContain("Web sources");
  expect(html).toContain("Web evidence supports this claim");
  expect(html).not.toContain("Based on the provided documents");
  expect(html).toContain('class="composer-model-status"');
  expect(html).not.toContain('class="actual-model-status"');
  expect(html).not.toContain("Uploaded documents");
  expect(html.match(/value="auto"/g)).toHaveLength(1);
  expect(html).toContain("model-picker-controls");
  // Ignored artifact for checking the actual component CSS at desktop/mobile widths.
  mkdirSync(".tmp", { recursive: true });
  const css = ["globals", "design-tokens", "workspace"]
    .map((file) => readFileSync(`app/${file}.css`, "utf8"))
    .join("\n");
  writeFileSync(
    ".tmp/chat-preview.html",
    `<!doctype html><html class="dark"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>${css}\nbody{height:100vh;display:block}.chat-view{height:100vh;max-width:900px;margin:auto}</style><body>${html}</body></html>`,
  );
});

it("replaces the streaming bubble with the accepted answer without a duplicate placeholder", () => {
  context.state.streaming = true;
  context.state.events = [{ type: "final" }];
  (context.state.messages as object[]).push({
    id: "answer",
    role: "assistant",
    content: "Saved grounded answer",
    sources: [],
  });
  const html = renderToStaticMarkup(createElement(Chat));
  expect(html.match(/Saved grounded answer/g)).toHaveLength(1);
  expect(html).not.toContain('class="thinking"');
  expect(html).not.toContain("Draft · checking evidence");
});
