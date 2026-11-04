# Agentic CRAG optimization report

Reviewed on 4 November 2026. Baseline: `origin/main` at `3ac095b`.

## Scope and implementation

0. **Synchronization:** fetched GitHub, fast-forwarded the clean checkout, and created `perf/agentic-crag-optimization` from the latest remote. The newer HTML/MDX pricing parser was retained.
1. **CRAG:** relevance grading, accepted-evidence coverage, and an optional gap query now share one structured evaluation. All chunk IDs and supporting IDs are checked. Invalid coverage references and legacy checkpoints still receive a separate coverage assessment. CORRECT / AMBIGUOUS / INCORRECT routing, exact sentence refinement, independent grounding validation, and LLM-selected corrective tools remain.
2. **Gemini:** live pricing and authenticated model listing must agree. Removed the static pricing-outage fallback; verification failures return actionable diagnostics. Preserved five-minute caching, added a bounded async discovery deadline and credential-rotation checks, and revalidate persisted plans after expiry/key changes. Lightweight tasks use observed successful task latency with a Flash/Lite cold-start preference; generation and validation retain model affinity and strict manual selection. Retry backoff is shorter only for transient errors; quota/auth failures are not retried blindly.
3. **Pipeline:** remove repeated evidence bookkeeping from prompts, bound whole-message history, deduplicate only identical sources with the same attribution, skip sentence selection for already atomic evidence, and reject duplicate corrective retrieval queries. The web-search cap now also applies when re-retrieval returns to web correction. PostgreSQL reads no longer wait behind the global indexing lock; workspace and published-version joins remain. Job events batch with checkpoints atomically, token-only writes avoid rewriting trace JSON, and first tokens flush immediately. Independent frontend refreshes run concurrently.
4. **Verification:** baseline had 198 passing backend tests. Final suite: **232 backend tests**, **29 frontend tests**, backend Ruff (`app tests scripts`), frontend TypeScript/ESLint, and production build all pass. Installed backend dependencies are compatible. New regressions cover combined-evaluation safety, old checkpoints, source attribution, bounded recovery, catalog/key isolation, task routing, atomic event writes, first-token persistence, and cross-workspace retrieval concurrency.

## Before and after

Before: retrieve → grading call → coverage call → optional query-rewrite call/search → optional refinement call → generation → validation → bounded LLM correction.

After: retrieve → combined grading/coverage/gap-query call → CRAG routing → optional search → refinement only when useful → generation → independent validation → bounded LLM correction. The existing graph node names, checkpoint transition rules, and frontend progress events remain compatible. The `assess` and `rewrite` nodes still exist even when they reuse the combined result.

Calls eliminated: the separate coverage request on verified evaluations, the separate query-rewrite request when the evaluation supplies a gap query, and sentence refinement when every source is already a single sentence. Generation and grounding validation remain separate.

## Comparable benchmark

**These are simulated provider timings with actual local SQLite job persistence, not live Gemini or Supabase measurements.** Each scenario uses one warmup and five measured runs; table entries are medians. Both versions run sequentially on the same host with 250 ms per structured request, two 50 ms generation chunks, and 10 ms web search. Quota fallback injects one synthetic 429 with 10 ms failure latency. No real provider calls, user files, or production records are involved.

| Scenario | Total ms before → after | First token ms before → after | LLM requests before → after | SQLite commits before → after |
|---|---|---|---|---|
| Correct documents | 1251.51 → 945.17 | 904.03 → 598.66 | 5 → 4 | 39 → 21 |
| Ambiguous + web | 1478.47 → 962.22 | 1138.27 → 616.57 | 6 → 4 | 48 → 25 |
| Incorrect documents + web | 963.75 → 947.52 | 635.65 → 616.60 | 4 → 4 | 46 → 25 |
| No evidence / abstention | 76.27 → 57.53 | 54.41 → 37.94 | 0 → 0 | 38 → 24 |
| Corrective retrieval | 2430.02 → 1861.01 | 878.26 → 610.74 | 10 → 8 | 75 → 38 |
| Single-sentence evidence | 978.58 → 688.89 | 629.46 → 354.72 | 4 → 3 | 38 → 21 |
| Synthetic 429 fallback | 1199.66 → 962.08 | 847.18 → 612.77 | 6 → 5 | 42 → 23 |

The sum of scenario medians fell by **23.3%**. This is an equal-scenario synthetic comparison, not a traffic-weighted production speed claim. The incorrect-evidence route makes the same number of requests; it does not get a guaranteed provider-latency improvement.

### Token estimates

Input/output figures below estimate serialized prompt/schema and synthetic response characters divided by four. They are **not provider token counts**, and do not include hidden thinking tokens or protocol overhead. The combined schema can increase output and incorrect-route input size; token reductions are not uniform.

| Scenario | Input estimate before → after | Output estimate before → after |
|---|---|---|
| Correct documents | 1674 → 1632 | 69 → 84 |
| Ambiguous + web | 1991 → 1750 | 83 → 95 |
| Incorrect documents + web | 1342 → 1641 | 51 → 91 |
| No evidence / abstention | 0 → 0 | 0 → 0 |
| Corrective retrieval | 3392 → 3335 | 136 → 166 |
| Single-sentence evidence | 1394 → 1367 | 63 → 78 |
| Synthetic 429 fallback | 2135 → 2061 | 69 → 84 |

Per-node retrieval, grading, assessment, rewrite, refinement, generation, validation, and correction timings are retained in [crag-before.json](crag-before.json) and [crag-after.json](crag-after.json). Fallback counts and the injected fallback-delay budget are recorded there. The fallback scenario adds one failed request in both versions; real fallback latency and backoff effects require live measurement. Sub-millisecond retrieval timings round to zero because this benchmark injects synthetic vectors; no PostgreSQL speed claim is inferred.

Reproduce the optimized run from `backend`:

```powershell
python scripts/benchmark_crag.py --durable --runs 5 --output ../docs/crag-after.json
```

To reproduce the baseline, export `backend/app` from `3ac095b` into an isolated directory and set `CREBRY_BENCH_BACKEND` to that exported backend directory before running the same script. The script never resets or overwrites the checkout.

## Gemini audit

Google public pricing, model, and deprecation documentation were fetched and inspected. `scripts/audit_gemini_models.py` also parsed the actual current pricing response successfully.

**No Gemini key is configured in this checkout.** Authenticated Models.list, generation, remaining quota, and production-key access were not tested. No model is claimed accessible or temporarily blocked for that specific key. The runtime catalog intersects current pricing with the configured key’s Models.list response; the audit script performs that read-only check when credentials are available. Listing a model does not prove generation will succeed or quota remains.

| Exact model / group | Verified public status and application handling |
|---|---|
| `gemini-3.1-flash-lite` | Stable, free Standard text input/output, multimodal input and structured output supported. Kept; not confused with its retired preview. |
| `gemini-3.1-flash-lite-preview` | Retired 25 May 2026; rejected even in saved plans. |
| `gemini-3.5-flash-lite` | Free Standard text input/output and compatible capabilities; kept. |
| `gemini-3-flash-preview` | Reviewed text/vision/structured preview with free Standard input/output; kept only when listed for the key. |
| `gemini-2.5-pro`, `gemini-2.5-flash` | Currently priced free but deprecated and limited to existing users. Selection stops on 1 December 2026 before documented endpoint redirection. |
| `gemini-2.5-flash-lite` | Free Standard input/output; no shutdown date announced in the reviewed docs. |
| `gemini-3.1-pro-preview` | Not verified as free by the current pricing parse; not offered. No claim that all 3.1 variants are retired. |
| `gemini-3.6-flash`, `gemini-3.8-flash` | Existing operator exclusions preserved. These are policy exclusions, not claims of retirement. Their 3.5/3.7 Flash redirect aliases are also excluded. |
| Older 1.x/2.0 text families; retired 2.5 previews; `gemini-3-pro-preview` | Rejected by permanent policy, independent of temporary availability. |
| Image/audio/live/TTS/embedding-only models and unreviewed previews | Excluded from text-generation choices. The working embedding path is unchanged. |
| 429, 503, timeouts, and key-specific 404 | Temporary failures remain cooldown/fallback conditions, not permanent retirement. Model entries are retained when eligible. |

Sources: [models](https://ai.google.dev/gemini-api/docs/models), [pricing](https://ai.google.dev/gemini-api/docs/pricing), [deprecations](https://ai.google.dev/gemini-api/docs/deprecations), [3.1 Flash-Lite capabilities](https://aistudio.google.com/docs/models/gemini-3.1-flash-lite).

## Simplification and removed files

The configuration and database-model tutorial commentary was shortened; executable ASTs match `origin/main` for both files. No columns, constraints, provider settings, or migrations changed. Removed unused API/AI loggers, the unused API SSE helper, and the chat hook that only repackaged two imported functions. Chat imports the same stream functions directly.

Removed files:

- `frontend/hooks/use-chat.ts`
- `frontend/artifacts/homepage-desktop.png`
- `frontend/artifacts/workspace-desktop.png`
- `docs/clerk-auth-controls.jpg`
- `docs/desktop-preview.jpg`
- `docs/landing-phone.jpg`
- `docs/preview.jpg`
- `image.png`

The seven removed images had no repository references and total 2,178,468 bytes. The screenshot referenced by verification notes and all application assets remain. Generated frontend screenshots are now ignored.

## Changed and added files

- Backend flow: `backend/app/agents/nodes.py`, `state.py`, and new `context.py`. `graph.py` was reviewed and its topology intentionally retained.
- Backend services: `backend/app/services/ai.py`, `free_models.py`, `routing.py`, `jobs.py`, and `postgres_vector.py`.
- API/configuration: `backend/app/api/routes.py`, `backend/app/core/config.py`, `backend/app/models/database.py`.
- Existing regressions updated: `backend/tests/test_ai.py`, `test_api.py`, `test_graph.py`, `test_jobs.py`, `test_routing.py`, `test_supabase.py`, `test_web_fallback.py`.
- New regressions: `backend/tests/test_catalog_policy.py`, `test_crag_optimization.py`, `test_job_optimization.py`.
- New tools: `backend/scripts/audit_gemini_models.py`, `benchmark_crag.py`.
- Frontend: `frontend/components/chat.tsx`, `frontend/lib/chat-render.test.ts`, `model-policy.ts`, `model-policy.test.ts`, `research-stream.ts`, `research-stream.test.ts`, `workflow.ts`.
- Documentation/housekeeping: `.gitignore`, `README.md`, this report, and the two benchmark JSON files.

## Limitations and rollout

- Live Gemini key access, production token usage, real research quality/latency, hosted OCR, and hosted PostgreSQL concurrency remain unverified. Mocked correctness checks are not a research-quality evaluation.
- The current Next.js 16.4.0 build reports an existing security advisory. Compilation, type checking, page generation, and the build retry pass, but a security dependency update is still required before claiming a fully hardened production deployment.
- Free-tier catalog failures deliberately fail closed. No key-specific availability is invented; operators should run the audit with their server-side configuration.
- Five-minute catalog caching and the reviewed capability registry need maintenance as Google changes models and pricing. Internal token reservations remain conservative estimates, not exact billing counts.
- Exact pgvector search is retained. No destructive migration, new dependency, production data mutation, or provider change was introduced.
- Existing graph nodes and legacy assessment checkpoints remain supported. Completed node events and checkpoints are committed together; an in-flight step can still repeat after a crash.
- Tests use isolated local databases and synthetic data. No production database was queried or changed during this implementation. The user subsequently authorized merging verified changes into `main`; deployment success must be checked separately.
