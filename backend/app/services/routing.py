"""Shared, transactional Gemini health, quota reservations and conversation affinity.

SQLite supports one persistent service; PostgreSQL supports multiple instances.
No prompts, credentials or provider response bodies are stored in this ledger.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import asyncio
import logging
import random
from math import ceil
from time import time
from time import monotonic
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.config import config
from app.models.database import ConversationModelPreference, RoutingState

_reservation = ContextVar("gemini_reservation", default=None)


class QuotaReservationFailed(RuntimeError):
    code = 429


def reserve_additional_call():
    current = _reservation.get()
    if current:
        store, lease, model, task, tokens, ttl = current
        if not store.reserve(model, task, tokens, ttl, lease=lease):
            raise QuotaReservationFailed("Shared free-tier request budget exhausted.")


class RoutingStore:
    def __init__(self, sessions, project: str):
        self.sessions, self.project = sessions, project
        # Initialize separately: a competing instance may create the same row.
        with sessions() as db:
            if db.get(RoutingState, project) is None:
                db.add(RoutingState(id=project, payload={}))
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()

    @contextmanager
    def transaction(self):
        with self.sessions() as db:
            if db.bind.dialect.name == "sqlite":
                db.connection().exec_driver_sql("BEGIN IMMEDIATE")
            row = db.scalar(select(RoutingState).where(RoutingState.id == self.project).with_for_update())
            payload = dict(row.payload)
            yield db, payload
            row.payload = payload
            db.commit()

    @staticmethod
    def health_key(model, task):
        return f"{task}:{model}"

    def ordered(self, models, task, conversation_id=None, selected=None):
        with self.sessions() as db:
            state = db.get(RoutingState, self.project).payload
            preference = db.get(ConversationModelPreference, (conversation_id, task)) if conversation_id else None
            if preference is None and conversation_id:
                preference = db.get(ConversationModelPreference, (conversation_id, "*"))
            sticky = preference.model if preference else None
        now = time()
        healthy = [m for m in dict.fromkeys(models) if self.available(state, m, task, now)]

        def score(model):
            health = state.get(self.health_key(model, task), {})
            lightweight = task in {"EvidenceEvaluation", "RelevanceBatch", "EvidenceAssessment", "EvidenceSelection", "RewrittenQuery", "CorrectionPlan"}
            # Prefer observed task latency. An unmeasured Flash/Lite is only a cold-start
            # prior, never a claimed benchmark; generation/validation retain affinity.
            latency = health.get("latency_ms") if health.get("successes", 0) else None
            prior = 0 if "flash-lite" in model else 1 if "flash" in model else 2
            if lightweight:
                return (model != selected if selected else False, health.get("failures", 0),
                        latency is None, latency if latency is not None else prior, model != sticky, models.index(model))
            return (model != selected if selected else False, model != sticky,
                    health.get("failures", 0), health.get("latency_ms", 0), models.index(model))

        return sorted(healthy, key=score)

    @staticmethod
    def available(state, model, task, now):
        return max(state.get(f"blocked:{model}", 0),
                   state.get(f"{task}:{model}", {}).get("until", 0)) <= now

    def availability(self, model, task):
        return self.availability_many([model], task)[model]

    def availability_many(self, models, task):
        """One ledger read for the entire catalog, without per-model round trips."""
        with self.sessions() as db:
            state = db.get(RoutingState, self.project).payload
        now = time()
        result = {}
        for model in models:
            remaining = max(0, ceil(max(state.get(f"blocked:{model}", 0),
                                      state.get(self.health_key(model, task), {}).get("until", 0)) - now))
            result[model] = {"available": remaining == 0, "retry_after_seconds": remaining}
        return result

    def reserve(self, model, task, tokens, ttl, lease=None):
        """Atomically reserve project-wide concurrency and per-model free budgets.

        Failed calls count towards budgets; expired leases release concurrency only.
        Quota values must be configured at or below the project's AI Studio limits.
        """
        now = time()
        with self.transaction() as (_, state):
            leases = {k: v for k, v in state.get("leases", {}).items() if v["until"] > now}
            state["leases"] = leases
            if not self.available(state, model, task, now):
                return None
            if (lease and lease not in leases) or (not lease and len(leases) >= config.gemini_max_concurrency):
                return None
            key = f"quota:{model}"
            quota = dict(state.get(key, {}))
            minute, day = int(now // 60), int(now // 86400)
            if quota.get("minute") != minute:
                quota.update(minute=minute, requests=0, tokens=0)
            if quota.get("day") != day:
                quota.update(day=day, daily=0)
            if (quota["requests"] >= config.gemini_free_rpm
                    or quota["daily"] >= config.gemini_free_rpd
                    or quota["tokens"] + tokens > config.gemini_free_tpm):
                state[f"blocked:{model}"] = (day + 1) * 86400 if quota["daily"] >= config.gemini_free_rpd else (minute + 1) * 60
                return None
            quota.update(requests=quota["requests"] + 1, daily=quota["daily"] + 1,
                         tokens=quota["tokens"] + tokens)
            state[key] = quota
            token = lease or uuid4().hex
            leases[token] = {"until": now + ttl}
            return token

    def finish(self, lease, model, task, latency_ms, *, status=None, success=False, conversation_id=None, retry_after=0, cancelled=False):
        with self.transaction() as (db, state):
            leases = dict(state.get("leases", {}))
            leases.pop(lease, None)
            state["leases"] = leases
            if cancelled:
                return
            key = self.health_key(model, task)
            health = dict(state.get(key, {}))
            health["latency_ms"] = round(0.7 * health.get("latency_ms", latency_ms) + 0.3 * latency_ms)
            health["last_status"] = status
            health["updated_at"] = time()
            if success:
                health.update(failures=0, until=0, successes=health.get("successes", 0) + 1)
                if conversation_id:
                    for kind in (task, "*"):
                        preference = db.get(ConversationModelPreference, (conversation_id, kind))
                        if preference is None:
                            db.add(ConversationModelPreference(conversation_id=conversation_id, task=kind, model=model))
                        else:
                            preference.model = model
            else:
                health["failures"] = health.get("failures", 0) + 1
                # An invalid task schema does not poison other capabilities.
                delay = min(60, 2 ** min(health["failures"], 5))
                health["until"] = time() + max(delay, retry_after)
                if status in {404, 429}:
                    state[f"blocked:{model}"] = max(state.get(f"blocked:{model}", 0), time() + max(300 if status == 404 else 60, retry_after))
            state[key] = health


def retry_after_seconds(error):
    """Honor a bounded numeric Retry-After without logging provider bodies."""
    from app.services.ai import error_chain
    for item in error_chain(error):
        response = getattr(item, "response", None)
        headers = getattr(response, "headers", {}) or {}
        try:
            return max(0, min(86400, float(headers.get("retry-after", 0))))
        except (ValueError, TypeError):
            continue
    return 0


async def execute_fallback(plan, operation, notify, *, store=None, task="generate", input_tokens=0, output_tokens=8192):
    from app.services.ai import ModelFallbackExhausted, is_model_unavailable, retryable_model_error, safe_service_error, service_status
    from app.services.free_models import mark_model_available, mark_model_unavailable, model_availability, model_allowed

    # A persisted job may outlive its verified catalog or use a replaced key.
    # Revalidate rather than assuming a former credential's access or pricing.
    if plan.get("credential_fingerprint"):
        from app.services.free_models import credential_fingerprint, get_free_models
        if (plan["credential_fingerprint"] != credential_fingerprint()
                or time() - plan.get("catalog_verified_at", 0) >= 300):
            changed_key = plan["credential_fingerprint"] != credential_fingerprint()
            fresh = await get_free_models()
            plan["catalog"] = fresh
            plan["models"] = [item["id"] for item in fresh]
            plan["credential_fingerprint"] = credential_fingerprint()
            plan["catalog_verified_at"] = time()
            if changed_key:
                plan["unavailable"] = []
                plan["failure_reasons"] = {}

    candidates = [model for model in dict.fromkeys(plan["models"]) if model_allowed(model)]
    catalog = plan.get("catalog")
    if catalog is not None:
        by_id = {item["id"]: item for item in catalog if item.get("free_tier_eligible")}
        capability = "vision" if task == "ocr" else "streaming" if task == "generate" else "structured_outputs"
        candidates = [model for model in candidates if model in by_id
                      and capability in by_id[model].get("supported_parameters", [])
                      and input_tokens <= by_id[model].get("context_length", 0)
                      and output_tokens <= by_id[model].get("output_token_limit", 0)]
    strict = not plan.get("automatic", True)
    selected = plan.get("selected") if strict and plan.get("selected") != "auto" else None
    if selected:
        candidates = [selected] if selected in candidates else []
    if store:
        candidates = await asyncio.to_thread(store.ordered, candidates, task, plan.get("conversation_id"), selected)
    else:
        active = plan.get("active")
        if active in candidates:
            candidates.remove(active)
            candidates.insert(0, active)
        failures = plan.setdefault("failure_reasons", {})
        for model in list(plan.setdefault("unavailable", [])):
            if model in failures and model_availability(model)["available"]:
                plan["unavailable"].remove(model)
                failures.pop(model, None)
        candidates = [model for model in candidates if model not in plan["unavailable"]]
    if strict and not selected:
        candidates = candidates[:1]
    # A wall-clock deadline survives checkpoint persistence and process restarts.
    remaining_request = plan.get("deadline_at", time() + config.gemini_operation_timeout_seconds) - time()
    deadline = monotonic() + min(config.gemini_operation_timeout_seconds, remaining_request)
    failures = plan.setdefault("failure_reasons", {})
    retries, last_error = [], None
    budget_expired = False
    attempted = 0
    for round_number in range(config.gemini_retry_rounds + 1):
        pending = candidates if round_number == 0 else retries[:config.gemini_max_model_attempts]
        retries = []
        if round_number and pending:
            delay = min(0.25 * 2 ** (round_number - 1) + random.uniform(0, 0.1), max(0, deadline - monotonic()))
            await asyncio.sleep(delay)
        for index, model in enumerate(pending):
            remaining = deadline - monotonic()
            if remaining <= 0:
                budget_expired = True
                break
            # Reserve time for other distinct candidates rather than spending it all on A.
            attempt_timeout = min(30, remaining / max(1, len(pending) - index))
            lease = None
            if store:
                lease = await asyncio.to_thread(store.reserve, model, task, input_tokens + output_tokens, attempt_timeout + 5)
                if lease is None:
                    failures[model] = "Model cooling down, project quota reserved, or concurrency limit reached."
                    continue
            started = monotonic()
            attempted += 1
            error = None
            succeeded = False
            status = None
            context_token = _reservation.set((store, lease, model, task, input_tokens + output_tokens, attempt_timeout + 5) if store else None)
            try:
                async with asyncio.timeout(attempt_timeout):
                    result = await operation(model)
                succeeded = True
                plan["active"] = model
                failures.pop(model, None)
                mark_model_available(model)
                return result
            except Exception as caught:
                error = caught
                status = service_status(caught)
                if not is_model_unavailable(caught):
                    raise
                last_error = caught
                reason = safe_service_error(caught)
                failures[model] = reason
                if status in {404, 429}:
                    if model not in plan.setdefault("unavailable", []):
                        plan["unavailable"].append(model)
                    mark_model_unavailable(model, max(60 if status == 429 else 300, int(retry_after_seconds(caught))))
                if retryable_model_error(caught):
                    retries.append(model)
                replacement = pending[index + 1] if index + 1 < len(pending) and monotonic() < deadline else None
                notify(model, replacement, reason)
                logging.getLogger(__name__).warning("Gemini failure model=%s task=%s status=%s error_type=%s", model, task, status, type(caught).__name__)
            finally:
                _reservation.reset(context_token)
                if store and lease:
                    # Shield cancellation so stopping a stream does not strand a quota lease.
                    await asyncio.shield(asyncio.to_thread(store.finish, lease, model, task, round((monotonic() - started) * 1000),
                        status=status, success=succeeded, conversation_id=plan.get("conversation_id"),
                        retry_after=retry_after_seconds(error) if error else 0, cancelled=error is None and not succeeded))
        if budget_expired or selected:
            break
    reason = safe_service_error(last_error) if last_error else next(iter(failures.values()), "No healthy compatible free-tier models are available.")
    suffix = " The shared time limit was reached." if budget_expired or monotonic() >= deadline else ""
    raise ModelFallbackExhausted("No eligible Gemini model completed this request. " + reason + suffix + " Progress is saved; retry when service or quota recovers.") from last_error
