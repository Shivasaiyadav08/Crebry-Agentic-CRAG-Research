"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  Copy,
  FileText,
  Globe2,
  LoaderCircle,
  RotateCcw,
  Square,
  Telescope,
  UserRound,
} from "lucide-react";
import { useSmoothedText } from "@/hooks/use-smoothed-text";
import { useResearch } from "@/stores/research";
import { Button } from "@/components/ui/button";
import type { Source } from "@/types";
import { FreeModelPicker } from "@/components/free-model-picker";
import { RelevancePicker } from "@/components/relevance-picker";
import { BrandMark } from "@/components/brand";
import {
  recoverResearch,
  sendResearch as send,
  stopResearch as stop,
} from "@/lib/research-stream";
import { withoutDocumentIntro } from "@/lib/answer-format";

export function Answer({
  content,
  sources,
}: {
  content: string;
  sources: Source[];
}) {
  const update = useResearch((state) => state.update);
  const linked = withoutDocumentIntro(content).replace(
    /\[([DW]\d+(?:\s*,\s*[DW]\d+)*)\]/g,
    (_match, group: string) =>
      group
        .split(/\s*,\s*/)
        .map((id) =>
          sources.some((source) => source.id === id)
            ? `[${id}](#source-${id})`
            : `[${id}]`,
        )
        .join(" "),
  );
  return (
    <div className="markdown" translate="no">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const source = href?.startsWith("#source-")
              ? sources.find((item) => item.id === href.slice(8))
              : undefined;
            return source ? (
              <button
                className="inline-citation"
                onClick={() => update({ selectedSource: source })}
                aria-label={`View source ${source.id}`}
              >
                {source.kind === "web" ? (
                  <Globe2 size={10} />
                ) : (
                  <FileText size={10} />
                )}
                {children}
              </button>
            ) : (
              <span>{children}</span>
            );
          },
          img: () => null,
        }}
      >
        {linked}
      </ReactMarkdown>
    </div>
  );
}

function EvidenceOrigin({ sources }: { sources: Source[] }) {
  const web = sources.some((source) => source.kind === "web");
  const documents = sources.some((source) => source.kind === "document");
  if (!web && !documents) return null;
  return (
    <span className="evidence-origin">
      {web && documents
        ? "Document and web sources"
        : web
          ? "Web sources"
          : "Uploaded documents"}
    </span>
  );
}

const suggestions = [
  {
    icon: FileText,
    label: "Understand my documents",
    text: "Summarize the key findings in my documents and cite the evidence.",
  },
  {
    icon: Telescope,
    label: "Explore a research question",
    text: "What are the main challenges in evaluating retrieval-augmented generation?",
  },
  {
    icon: Globe2,
    label: "Connect the dots",
    text: "Compare the evidence in my documents with recent research on the same topic.",
  },
];

export function Chat() {
  const {
    messages,
    draft,
    correctingDraft,
    streaming,
    actualModel,
    health,
    error,
    warnings,
    events,
    workspaces,
    workspaceId,
    conversations,
    runId,
    update,
  } = useResearch();
  const reducedMotion = useReducedMotion();
  const smoothDraft = useSmoothedText(draft, streaming && !reducedMotion);
  const workspace = workspaces.find((item) => item.id === workspaceId);
  const [question, setQuestion] = useState("");
  const [copied, setCopied] = useState("");
  const [nearBottom, setNearBottom] = useState(true);
  const scrollArea = useRef<HTMLDivElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const empty = messages.length === 0;
  const available = [
    ...(events.findLast((event) => event.type === "retrieval")?.sources ?? []),
    ...(events.findLast((event) => event.type === "web_sources")?.sources ??
      []),
  ];
  const lastUser = messages.findLast((message) => message.role === "user");
  const activeNode = events
    .findLast((event) => event.type === "node_start")
    ?.node?.replaceAll("_", " ");

  useEffect(() => {
    if (nearBottom && !empty)
      scrollArea.current?.scrollTo({
        top: scrollArea.current.scrollHeight,
        behavior: "instant",
      });
  }, [messages, smoothDraft, events.length, nearBottom, empty]);

  function submit() {
    if (question.trim() && !streaming) {
      void send(question);
      setQuestion("");
      setNearBottom(true);
    }
  }
  async function copy(content: string, id: string) {
    try {
      await navigator.clipboard.writeText(withoutDocumentIntro(content));
      setCopied(id);
      setTimeout(() => setCopied(""), 2000);
    } catch {
      update({ error: "Clipboard access was denied by your browser." });
    }
  }

  return (
    <div className={`chat-view ${empty ? "welcome" : ""}`}>
      <div
        className="chat-scroll"
        ref={scrollArea}
        onScroll={(event) => {
          const el = event.currentTarget;
          setNearBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 120);
        }}
      >
        {empty ? (
          <motion.div
            className="welcome-content"
            initial={reducedMotion ? false : { opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className="welcome-label">
              <BrandMark /> Your research workspace
            </div>
            <h1>
              What would you like
              <br />
              to understand?
            </h1>
            <p>
              Start with a question. Bring your sources.
              <br />
              Follow the evidence wherever it leads.
            </p>
            <div className="welcome-context">
              <span>
                <FileText size={14} /> {workspace?.document_count ?? 0}{" "}
                documents
              </span>
              <span>
                <BookOpen size={14} /> {conversations.length} saved
                conversations
              </span>
              <button onClick={() => update({ view: "documents" })}>
                Manage sources <ArrowRight size={13} />
              </button>
            </div>
            <div className="suggestion-label">A few places to start</div>
            <div className="suggestion-grid">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion.label}
                  className="suggestion-card"
                  onClick={() => {
                    setQuestion(suggestion.text);
                    textarea.current?.focus();
                  }}
                >
                  <suggestion.icon size={18} />
                  <strong>{suggestion.label}</strong>
                  <span>{suggestion.text}</span>
                  <ArrowUp size={14} className="suggestion-arrow" />
                </button>
              ))}
            </div>
          </motion.div>
        ) : (
          <div className="message-list">
            {messages.map((message, index) => (
              <motion.article
                key={message.id}
                className={`message ${message.role}`}
                aria-label={
                  message.role === "user" ? "Your question" : "Research answer"
                }
                translate="no"
                initial={reducedMotion ? false : { opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2 }}
              >
                <div className="message-avatar" aria-hidden="true">
                  {message.role === "assistant" ? (
                    <BrandMark />
                  ) : (
                    <UserRound size={15} />
                  )}
                </div>
                <div className="message-body">
                  {message.role === "assistant" && (
                    <div className="message-heading">
                      <strong>Crebry</strong>
                      <span>Research answer</span>
                    </div>
                  )}
                  {message.role === "assistant" ? (
                    <>
                      <EvidenceOrigin sources={message.sources} />
                      <Answer
                        content={message.content}
                        sources={message.sources}
                      />
                      {message.sources.length > 0 && (
                        <div className="answer-sources">
                          {message.sources.map((source) => (
                            <button
                              key={source.id}
                              onClick={() => update({ selectedSource: source })}
                            >
                              <span>{source.id}</span>
                              {source.kind === "web" ? (
                                <Globe2 size={12} />
                              ) : (
                                <FileText size={12} />
                              )}
                              <span className="source-name">{source.name}</span>
                              <ArrowRight size={12} />
                            </button>
                          ))}
                        </div>
                      )}
                      <div className="answer-actions">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => void copy(message.content, message.id)}
                        >
                          {copied === message.id ? (
                            <Check size={13} />
                          ) : (
                            <Copy size={13} />
                          )}
                          {copied === message.id ? "Copied" : "Copy"}
                        </Button>
                        {index === messages.length - 1 && lastUser && (
                          <Button
                            variant="ghost"
                            size="sm"
                            disabled={streaming}
                            onClick={() => void send(lastUser.content, true)}
                          >
                            <RotateCcw size={13} /> Regenerate
                          </Button>
                        )}
                        {message.run_id && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => update({ intelligenceOpen: true })}
                          >
                            <BookOpen size={13} /> Inspect evidence
                          </Button>
                        )}
                      </div>
                    </>
                  ) : (
                    <p className="user-text">{message.content}</p>
                  )}
                </div>
              </motion.article>
            ))}
            {streaming && !events.some((event) => event.type === "final") && (
              <article className="message assistant">
                <div className="message-avatar">
                  <BrandMark />
                </div>
                <div className="message-body">
                  <div className="message-heading">
                    <strong>Crebry</strong>
                    <span className="stream-label">
                      <LoaderCircle size={11} className="spin" />{" "}
                      {activeNode ?? "Starting research"}
                    </span>
                  </div>
                  {draft ? (
                    <>
                      <EvidenceOrigin
                        sources={[
                          ...events
                            .filter(
                              (event) =>
                                event.type === "grade" &&
                                event.source?.accepted,
                            )
                            .map((event) => event.source!),
                          ...(events.find(
                            (event) => event.type === "web_sources",
                          )?.sources ?? []),
                        ]}
                      />
                      <Answer content={smoothDraft} sources={available} />
                      <span className="provisional">
                        {correctingDraft
                          ? "Checking a correction against the evidence · your draft stays visible"
                          : "Draft · checking evidence before saving"}
                      </span>
                    </>
                  ) : (
                    <div className="thinking">
                      <span />
                      <span />
                      <span />
                    </div>
                  )}
                </div>
              </article>
            )}
          </div>
        )}
      </div>
      {!nearBottom && (
        <div className="scroll-bottom-anchor">
          <Button
            variant="secondary"
            size="icon"
            className="scroll-bottom"
            onClick={() => {
              setNearBottom(true);
              scrollArea.current?.scrollTo({
                top: scrollArea.current.scrollHeight,
                behavior: "smooth",
              });
            }}
            aria-label="Scroll to latest message"
          >
            <ArrowDown size={16} />
          </Button>
        </div>
      )}
      <div className="composer-area">
        {error && (
          <div className="alert" role="alert">
            {error}
            {runId && !streaming && (
              <button type="button" onClick={() => void recoverResearch()}>
                Resume saved research
              </button>
            )}
            <button
              onClick={() => update({ error: "" })}
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}
        {warnings.length > 0 && (
          <details
            className="warnings"
            open={events.some(
              (event) =>
                event.type === "web_sources" &&
                typeof event.message === "string" &&
                !!event.message,
            )}
          >
            <summary>
              {warnings.length} research{" "}
              {warnings.length === 1 ? "note" : "notes"}
            </summary>
            {warnings.map((warning, i) => (
              <p key={i}>{warning}</p>
            ))}
          </details>
        )}
        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
        >
          <textarea
            ref={textarea}
            aria-label="Research question"
            placeholder="What would you like to understand?"
            value={question}
            maxLength={4000}
            disabled={!health}
            onChange={(event) => {
              setQuestion(event.target.value);
              event.target.style.height = "auto";
              event.target.style.height = `${Math.min(event.target.scrollHeight, 150)}px`;
            }}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault();
                submit();
              }
            }}
            rows={2}
          />
          <div className="composer-footer">
            <div className="composer-tools">
              <Button
                variant="ghost"
                size="sm"
                type="button"
                onClick={() => update({ view: "documents" })}
              >
                <BookOpen size={14} />
                <span>Sources</span>
              </Button>
              <span className="composer-divider" />
              <FreeModelPicker showActualModel={false} />
              <RelevancePicker />
            </div>
            {streaming ? (
              <Button
                size="icon"
                type="button"
                onClick={stop}
                aria-label="Stop research"
              >
                <Square size={14} fill="currentColor" />
              </Button>
            ) : (
              <Button
                size="icon"
                type="submit"
                disabled={
                  !question.trim() ||
                  !health ||
                  !useResearch.getState().selectedModel
                }
                aria-label="Send question"
              >
                <ArrowUp size={18} />
              </Button>
            )}
          </div>
          {actualModel && (
            <span
              className="composer-model-status"
              role="status"
              title={`Actual model: ${actualModel}`}
            >
              Using {actualModel}
            </span>
          )}
        </form>
        <div className="composer-hint">
          <ShieldHint />
          <span>Grounded in your knowledge. Expanded when needed.</span>
          <span className="key-hint">↵ to send</span>
        </div>
        {health && !health.gemini_configured && (
          <p className="setup-hint">
            Research is unavailable until the application owner connects Gemini.
          </p>
        )}
        {health && !health.tavily_configured && (
          <p className="setup-hint" role="status">
            Web search is not connected. Research can use uploaded documents
            only.
          </p>
        )}
      </div>
    </div>
  );
}

function ShieldHint() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    >
      <path d="M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6Z" />
      <path d="m8 12 3 3 5-6" />
    </svg>
  );
}
