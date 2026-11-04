"use client";

import { useState } from "react";

import {
  Activity,
  ArrowUpRight,
  Database,
  ChevronDown,
  Globe2,
  ScanLine,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import { useResearch } from "@/stores/research";
import { formatTime } from "@/lib/utils";
import { Workflow } from "@/components/workflow";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";

export function Intelligence() {
  const [workflowOpen, setWorkflowOpen] = useState(false);
  const { events, streaming, elapsedMs, update } = useResearch();
  const latestRetrievalIndex = events.findLastIndex(
    (event) => event.type === "retrieval",
  );
  const retrieved = events[latestRetrievalIndex]?.sources ?? [];
  const graded = events
    .slice(latestRetrievalIndex + 1)
    .filter((event) => event.type === "grade")
    .map((event) => event.source!);
  const usedThreshold = events.findLast(
    (event) => event.type === "node_end" && event.node === "grade",
  )?.details?.threshold;
  const web =
    events.findLast((event) => event.type === "web_sources")?.sources ?? [];
  const decision = events.findLast((event) => event.type === "decision");
  const correction = events.findLast(
    (event) => event.type === "correction_plan",
  );
  const rewritten = events.findLast((event) => event.type === "rewrite")?.query;
  const latestNode = events.findLast(
    (event) => event.type === "node_start",
  )?.node;
  const final = events.findLast((event) => event.type === "final");
  const failed = events.some((event) => event.type === "error");
  const candidates = retrieved.map(
    (source) => graded.find((item) => item.id === source.id) ?? source,
  );

  return (
    <aside className="intelligence-panel">
      <div className="intelligence-heading">
        <div>
          <ScanLine size={17} />
          <strong>Research evidence</strong>
          <span className="beta-label">LIVE</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Close intelligence panel"
          onClick={() => update({ intelligenceOpen: false })}
        >
          <X size={16} />
        </Button>
      </div>
      <div className="intelligence-scroll">
        <div className="run-status">
          <span className={`status-dot ${streaming ? "live" : ""}`} />
          <span>
            {streaming
              ? `Running ${latestNode?.replaceAll("_", " ") ?? "workflow"}`
              : failed
                ? "Run failed"
                : final
                  ? final.valid
                    ? "Research complete"
                    : "Validation failed"
                  : "Awaiting your question"}
          </span>
          <code>{events.length ? formatTime(elapsedMs) : "—"}</code>
        </div>
        <details className="panel-section process-disclosure">
          <summary className="eyebrow">
            <Activity size={13} /> Research process
            <ChevronDown size={13} />
          </summary>
          <div className="mini-flow">
            <Workflow compact />
          </div>
          <small className="muted">Select a node to learn what it does.</small>
          <Button
            variant="ghost"
            size="sm"
            className="expand-process"
            onClick={() => setWorkflowOpen(true)}
          >
            Expand process diagram <ArrowUpRight size={12} />
          </Button>
        </details>
        <div className="panel-section">
          <div className="eyebrow">
            <Database size={13} /> Document evidence{" "}
            <span>{candidates.length || "—"}</span>
          </div>
          {typeof usedThreshold === "number" &&
            Number.isFinite(usedThreshold) && (
              <p className="muted">
                Minimum relevance used: {Math.round(usedThreshold * 100)}%
              </p>
            )}
          {!candidates.length && (
            <div className="panel-empty">
              Relevant passages from your knowledge will appear here.
            </div>
          )}
          {candidates.map((source) => (
            <button
              key={source.id}
              className="evidence-card"
              onClick={() => update({ selectedSource: source })}
            >
              <div>
                <span className="citation-tag">{source.id}</span>
                <strong>{source.name}</strong>
                <span className={source.accepted ? "score accepted" : "score"}>
                  {source.relevance === undefined
                    ? "grading…"
                    : `${Math.round(source.relevance * 100)}%`}
                </span>
              </div>
              <p>
                {source.text.slice(0, 130)}
                {source.text.length > 130 ? "…" : ""}
              </p>
              {source.reason && <small>{source.reason}</small>}
            </button>
          ))}
        </div>
        <div className="panel-section">
          <div className="eyebrow">
            <ShieldCheck size={13} /> Evidence assessment
          </div>
          {decision ? (
            <div className="decision-card">
              <span
                className={`decision-tag ${decision.route === "web" ? "web" : ""}`}
              >
                {decision.route === "web" ? (
                  <Globe2 size={13} />
                ) : (
                  <Database size={13} />
                )}
                {decision.route === "web"
                  ? decision.verdict === "AMBIGUOUS"
                    ? "Partial evidence · augmenting with web"
                    : "Insufficient documents · web correction"
                  : "Documents sufficient"}
              </span>
              {decision.verdict && <small>CRAG: {decision.verdict}</small>}
              <p>{decision.reason}</p>
              {decision.missing_information && (
                <small>Missing: {decision.missing_information}</small>
              )}
            </div>
          ) : (
            <div className="panel-empty">
              The agent will check whether your documents contain enough
              evidence.
            </div>
          )}
        </div>
        {correction && (
          <div className="panel-section">
            <div className="eyebrow">
              <Activity size={13} /> CORRECTIVE ACTION
            </div>
            <p>
              {correction.action?.replaceAll("_", " ") ?? "Reviewing options"}
            </p>
            {correction.reason && <small>{correction.reason}</small>}
          </div>
        )}
        {rewritten && (
          <div className="panel-section">
            <div className="eyebrow">
              <Sparkles size={13} /> REWRITTEN QUERY
            </div>
            <p className="rewritten-query">{rewritten}</p>
          </div>
        )}
        {web.length > 0 && (
          <div className="panel-section">
            <div className="eyebrow">
              <Globe2 size={13} /> Web sources <span>{web.length}</span>
            </div>
            {web.map((source) => (
              <button
                className="web-source"
                key={source.id}
                onClick={() => update({ selectedSource: source })}
              >
                <span className="citation-tag">{source.id}</span>
                <span>{source.name}</span>
                <ArrowUpRight size={14} />
              </button>
            ))}
          </div>
        )}
        <div className="panel-footnote">
          <ShieldCheck size={14} /> Evidence first. Every step visible.
        </div>
      </div>
      <Dialog open={workflowOpen} onOpenChange={setWorkflowOpen}>
        <DialogContent className="workflow-dialog">
          <DialogTitle>The research process</DialogTitle>
          <DialogDescription>
            Inspect each step from retrieval to the final answer.
          </DialogDescription>
          <div className="expanded-workflow">
            <Workflow />
          </div>
        </DialogContent>
      </Dialog>
    </aside>
  );
}
