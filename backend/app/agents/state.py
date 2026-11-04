from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class RelevanceGrade(BaseModel):
    score: float = Field(
        ge=0, le=1, description="How directly this chunk helps answer the question, from 0 to 1"
    )
    reason: str = Field(description="Brief explanation of relevance, based only on the chunk")


class ChunkRelevanceGrade(RelevanceGrade):
    id: str = Field(description="Exact supplied document chunk ID, such as D1")


class RelevanceBatch(BaseModel):
    grades: list[ChunkRelevanceGrade] = Field(
        min_length=1, description="Exactly one relevance judgment for each supplied chunk"
    )


class EvidenceAssessment(BaseModel):
    sufficient: bool = Field(
        description="True only if the evidence supports every important part of the question"
    )
    reason: str
    missing_information: str


class EvidenceSelection(BaseModel):
    """Select only exact source-derived sentence IDs; never generate new evidence."""

    keep_ids: list[str] = Field(description="Identifiers of passages relevant to the question")


class CorrectionPlan(BaseModel):
    """Allowed adaptive actions after a failed answer validation."""

    action: Literal["regenerate", "retrieve", "web_search", "stop"]
    reason: str = Field(description="Why this action addresses the validation feedback")
    query: str = Field(
        default="", max_length=500, description="Focused query for a retrieval or search action"
    )


class RewrittenQuery(BaseModel):
    query: str = Field(
        min_length=1, max_length=500, description="Standalone web search query preserving user intent"
    )


class AnswerValidation(BaseModel):
    grounded: bool = Field(description="Every factual claim is supported by its cited evidence")
    reason: str
    needs_more_evidence: bool = Field(
        default=False, description="True when available sources cannot support the required claims"
    )


class GraphState(TypedDict, total=False):
    resume_node: str
    question: str
    workspace_id: str
    settings: dict
    history: list[dict]
    retrieved: list[dict]
    sources: list[dict]
    assessment: dict
    rewritten_query: str
    web_search_error: str
    answer: str
    attempts: int
    valid: bool
    feedback: str
    warnings: list[str]
    model_plan: dict
    crag_verdict: Literal["CORRECT", "AMBIGUOUS", "INCORRECT"]
    refined_sources: list[dict]
    retrieval_rounds: int
    retrieval_query: str
    web_attempts: int
    correction_action: str
    correction_query: str
    needs_more_evidence: bool
