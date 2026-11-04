/** Accurate, human-readable depiction of the backend LangGraph CRAG graph. */
export const workflowSteps = [
  {
    id: "retrieve",
    title: "Retrieve",
    subtitle: "Workspace documents",
    file: "nodes.py",
    purpose:
      "Search this workspace for relevant document chunks. A corrective plan can issue a focused second retrieval.",
    input:
      "Current question, workspace ID, recent context or a focused correction query.",
    output:
      "Source chunks with stable D citation IDs and original document metadata.",
    code: "results = await asyncio.to_thread(\n    vectors.retrieve, workspace_id, query, count\n)",
  },
  {
    id: "grade",
    title: "Grade relevance",
    subtitle: "Combined evidence evaluation",
    file: "nodes.py",
    purpose:
      "Score document relevance and assess coverage in one structured AI call. Coverage may use only chunks that pass the relevance cutoff.",
    input: "Question, retrieved chunks, request-scoped relevance threshold.",
    output:
      "Validated relevance scores, accepted documents, and a coverage judgment.",
    code: "result = await ask(state, EvidenceEvaluation, instruction, data)\naccepted = score >= relevance_threshold",
  },
  {
    id: "assess",
    title: "CRAG verdict",
    subtitle: "Correct · Ambiguous · Incorrect",
    file: "nodes.py + graph.py",
    purpose:
      "Apply the CRAG bands to the combined evaluation. If its coverage references fail validation or a legacy checkpoint lacks an evaluation, assess the accepted evidence separately. The verdict determines whether web correction is needed.",
    input: "Question, selected documents, evidence coverage assessment.",
    output:
      "CORRECT → refine; AMBIGUOUS → augment via web; INCORRECT → replace with web evidence.",
    code: "verdict = (\n  'INCORRECT' if not accepted or strongest < 0.3 else\n  'CORRECT' if sufficient and strongest >= 0.7 else\n  'AMBIGUOUS'\n)",
  },
  {
    id: "rewrite",
    title: "Focus search",
    subtitle: "Find the evidence gap",
    file: "nodes.py",
    purpose:
      "Use the focused query from the combined evaluation or correction plan when available; otherwise rewrite it separately. Standalone questions keep their original intent.",
    input:
      "Question, missing evidence, relevant history or approved corrective plan.",
    output:
      "A standalone search query or the original question on safe fallback.",
    code: "query = planned_query or rewritten_query or question",
  },
  {
    id: "web_search",
    title: "Web correction",
    subtitle: "Tavily evidence",
    file: "nodes.py",
    purpose:
      "Augment partial document evidence or replace irrelevant documents with external evidence. A corrective planner cannot repeat a web search indefinitely.",
    input: "Focused query, server-side search configuration.",
    output:
      "Web source evidence with W citation IDs and warnings when unavailable.",
    code: "web = await search_web(query)\nsources = accepted_documents + web",
  },
  {
    id: "refine",
    title: "Refine evidence",
    subtitle: "Source-preserving strips",
    file: "nodes.py",
    purpose:
      "Select relevant original sentences instead of inventing a compressed summary. Retain exact source IDs, and keep graded evidence when selection cannot be verified.",
    input: "Question and source-backed evidence sentence IDs.",
    output:
      "Citation-safe refined evidence used for generation and validation.",
    code: "keep_ids = await ask(state, EvidenceSelection, ...)\nrefined = original_sentences_for(keep_ids)",
  },
  {
    id: "generate",
    title: "Generate",
    subtitle: "Streaming cited answer",
    file: "nodes.py",
    purpose:
      "Create a cited Markdown draft using only the refined evidence, with verified free-model fallback preserved.",
    input:
      "Question, bounded chat history, source evidence and correction feedback.",
    output: "Streamed provisional draft with D/W citations.",
    code: "async for chunk in llm.astream(messages):\n    emit('token', text=message_text(chunk))",
  },
  {
    id: "validate",
    title: "Validate",
    subtitle: "Citations and grounding",
    file: "nodes.py + graph.py",
    purpose:
      "Check citation IDs and, when enabled, assess whether claims are supported by the evidence actually supplied to generation.",
    input: "Draft, refined evidence and configured validation policy.",
    output: "Approved answer or feedback for a bounded corrective plan.",
    code: "route = 'finalize' if valid or retries_exhausted else 'plan_correction'",
  },
  {
    id: "plan_correction",
    title: "Plan correction",
    subtitle: "Agent-chosen recovery",
    file: "nodes.py + graph.py",
    purpose:
      "On an invalid answer, select an allowed recovery action: regenerate, retrieve documents, search the web, or stop. Guardrails cap retries and repeated tool calls.",
    input:
      "Validation feedback, current evidence, allowed tools and remaining budget.",
    output:
      "One validated tool choice and focused query, or a safe abstention.",
    code: "action = plan.action if plan.action in allowed else 'regenerate'",
  },
  {
    id: "finalize",
    title: "Finalize",
    subtitle: "Publish or abstain",
    file: "nodes.py",
    purpose:
      "Publish a validated answer and only cited sources, or refuse to present unsupported claims after the correction budget is exhausted.",
    input: "Answer, validation result, citation set, warnings.",
    output: "Durable final event and saved answer or abstention.",
    code: "emit('final', answer=answer, sources=cited, valid=valid)",
  },
];

export const workflowConnections = [
  ["retrieve", "grade", ""],
  ["grade", "assess", ""],
  ["assess", "refine", "Correct"],
  ["assess", "rewrite", "Ambiguous / Incorrect"],
  ["rewrite", "web_search", ""],
  ["web_search", "refine", ""],
  ["refine", "generate", ""],
  ["generate", "validate", ""],
  ["validate", "plan_correction", "Needs correction"],
  ["validate", "finalize", "Valid / exhausted"],
  ["plan_correction", "generate", "Regenerate"],
  ["plan_correction", "retrieve", "Re-retrieve"],
  ["plan_correction", "rewrite", "Web correction"],
  ["plan_correction", "finalize", "Stop"],
] as const;
