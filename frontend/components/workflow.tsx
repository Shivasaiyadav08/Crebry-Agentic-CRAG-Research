"use client";

import { useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { BookOpen, Check, Circle, Clock3, LoaderCircle, X } from "lucide-react";
import { useTheme } from "next-themes";
import { workflowConnections, workflowSteps } from "@/lib/workflow";
import { formatTime } from "@/lib/utils";
import { useResearch } from "@/stores/research";
import { Button } from "@/components/ui/button";

type AgentNode = Node<
  {
    title: string;
    subtitle: string;
    status: string;
    duration?: number;
    reverse?: boolean;
  },
  "agent"
>;

function Agent({ data }: NodeProps<AgentNode>) {
  return (
    <div className={`agent-node ${data.status}`}>
      <Handle
        type="target"
        position={data.reverse ? Position.Right : Position.Left}
      />
      <div className="agent-icon">
        {data.status === "running" ? (
          <LoaderCircle className="spin" size={17} />
        ) : data.status === "complete" ? (
          <Check size={17} />
        ) : (
          <Circle size={17} />
        )}
      </div>
      <div>
        <strong>{data.title}</strong>
        <small>{data.subtitle}</small>
      </div>
      {data.duration !== undefined && (
        <span className="node-time">{formatTime(data.duration)}</span>
      )}
      <Handle
        type="source"
        position={data.reverse ? Position.Left : Position.Right}
      />
    </div>
  );
}
const nodeTypes = { agent: Agent };

export function Workflow({ compact = false }: { compact?: boolean }) {
  const { events, streaming } = useResearch();
  const { resolvedTheme } = useTheme();
  const [learn, setLearn] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const { nodes, edges } = useMemo(() => {
    const starts = events.filter((event) => event.type === "node_start");
    const latestStart = starts.at(-1)?.node;
    const lastEnd = events.findLast((event) => event.type === "node_end")?.node;
    const failed = events.some((event) => event.type === "error");
    const active =
      streaming && latestStart !== lastEnd ? latestStart : undefined;
    const documentRoute =
      events.findLast((event) => event.type === "decision")?.verdict ===
      "CORRECT";
    const webUsed = starts.some((event) => event.node === "web_search");
    const nodes: AgentNode[] = workflowSteps.map((step, displayIndex) => {
      const ended = events.findLast(
        (event) => event.type === "node_end" && event.node === step.id,
      );
      const began = starts.some((event) => event.node === step.id);
      const skipped =
        documentRoute &&
        !webUsed &&
        ["rewrite", "web_search"].includes(step.id);
      const status =
        active === step.id
          ? "running"
          : ended
            ? "complete"
            : skipped
              ? "skipped"
              : began && failed
                ? "failed"
                : "idle";
      return {
        id: step.id,
        type: "agent",
        position: {
          x: (displayIndex % 4) * 290,
          y: Math.floor(displayIndex / 4) * 210,
        },
        data: {
          title: step.title,
          subtitle: skipped ? "Skipped · documents sufficient" : step.subtitle,
          status,
          duration: ended?.duration_ms,
          reverse: displayIndex >= 4,
        },
      };
    });
    const edges = workflowConnections.map(([source, target, label], i) => ({
      id: `edge-${i}`,
      source,
      target,
      label: compact ? undefined : label,
      animated:
        target === active &&
        nodes.find((node) => node.id === source)?.data.status === "complete",
      type: "smoothstep",
      style: {
        stroke: target === active ? "var(--accent)" : "var(--edge)",
        strokeWidth: 1.5,
      },
      labelStyle: { fill: "var(--muted)", fontSize: 10 },
      labelBgStyle: { fill: "var(--panel)" },
    }));
    return { nodes, edges };
  }, [events, streaming, compact]);
  const step = workflowSteps.find((item) => item.id === selected);
  const execution = events.findLast(
    (event) => event.type === "node_end" && event.node === selected,
  );

  return (
    <div className={`workflow ${compact ? "compact" : ""}`}>
      {!compact && (
        <div className="workflow-toolbar">
          <span>
            <span className={`status-dot ${streaming ? "live" : ""}`} />
            {streaming
              ? "Executing live"
              : events.length
                ? "Recorded execution"
                : "Ready for a question"}
          </span>
          <Button
            variant={learn ? "secondary" : "ghost"}
            size="sm"
            onClick={() => setLearn(!learn)}
            aria-pressed={learn}
          >
            <BookOpen size={14} /> Learn mode
          </Button>
        </div>
      )}
      <div className="flow-canvas">
        {compact ? (
          <div className="process-list">
            {nodes.map((node) => (
              <button
                key={node.id}
                className={`process-step ${node.data.status}`}
                onClick={() => setSelected(node.id)}
              >
                <span>
                  {node.data.status === "complete" ? (
                    <Check size={13} />
                  ) : node.data.status === "running" ? (
                    <LoaderCircle className="spin" size={13} />
                  ) : (
                    <Circle size={9} />
                  )}
                </span>
                <strong>{node.data.title}</strong>
                <small>
                  {node.data.duration === undefined
                    ? node.data.status === "idle"
                      ? "Waiting"
                      : node.data.status
                    : formatTime(node.data.duration)}
                </small>
              </button>
            ))}
          </div>
        ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.12 }}
            minZoom={0.25}
            maxZoom={1.5}
            colorMode={resolvedTheme === "dark" ? "dark" : "light"}
            nodesDraggable={!compact}
            nodesConnectable={false}
            onNodeClick={(_, node) => setSelected(node.id)}
            proOptions={{ hideAttribution: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={22}
              size={1}
              color="var(--grid)"
            />
            {!compact && <Controls showInteractive={false} />}
          </ReactFlow>
        )}
      </div>
      {step && (
        <div className="learn-card">
          <div className="section-heading">
            <span>
              <BookOpen size={16} /> {step.title}
            </span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSelected(null)}
              aria-label="Close node details"
            >
              <X size={15} />
            </Button>
          </div>
          {learn && (
            <>
              <p>{step.purpose}</p>
              <dl>
                <dt>INPUT</dt>
                <dd>{step.input}</dd>
                <dt>OUTPUT</dt>
                <dd>{step.output}</dd>
              </dl>
              <div className="code-file">
                backend/app/agents/
                {step.file}
              </div>
              <pre>
                <code>{step.code}</code>
              </pre>
            </>
          )}
          <div className="execution-detail">
            <Clock3 size={14} />{" "}
            {execution
              ? `Completed in ${formatTime(execution.duration_ms ?? 0)}`
              : "This node has not completed in this run."}
          </div>
          {execution?.details && (
            <pre className="event-data">
              {JSON.stringify(execution.details, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
