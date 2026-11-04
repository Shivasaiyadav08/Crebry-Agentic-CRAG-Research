# Crebry — AI Research Workspace

Crebry helps you research uploaded documents and web sources. Ask a question, follow the research steps, and read an answer with citations you can inspect.

## Using Crebry

1. Sign in and create a workspace.
2. Upload PDF, DOCX, TXT, or Markdown files.
3. Choose an available Gemini model and ask a question.
4. Crebry checks your documents and searches the web when more evidence is needed and web search is configured.
5. Read the answer, open its citations, and ask follow-up questions. Conversations are saved in your workspace.

## Project flow

### When you upload a document

The backend checks that you own the workspace, extracts the text, and divides it into smaller passages. Scanned PDFs can use Gemini OCR to read text from images.

Gemini turns each passage into an embedding: a list of numbers used to find passages with similar meaning. In the hosted setup, Supabase stores these searchable passages, document details, and the original file.

### When you ask a question

```mermaid
flowchart TD
    Question[Ask a question] --> Retrieve[Find relevant document passages]
    Retrieve --> Grade[Score relevance and check coverage]
    Grade --> Decision{Is the evidence strong and complete?}
    Decision -->|Yes| Refine[Select useful source sentences]
    Decision -->|No| Search[Focus the query and search the web]
    Search --> Refine
    Refine --> Generate[Generate an answer with citations]
    Generate --> Validate[Check citations and supporting evidence]
    Validate -->|Pass| Save[Save and display the final answer]
    Validate -->|Fail| Correct[Choose a correction or stop]
    Correct -->|Rewrite the answer| Generate
    Correct -->|Find better documents| Retrieve
    Correct -->|Find web evidence| Search
```

LangGraph controls this workflow. It uses Corrective Retrieval-Augmented Generation (**CRAG**) to decide how to handle document evidence:

Relevance scoring and evidence coverage share one AI evaluation. When more evidence is needed, that evaluation can also prepare the web query. The workflow keeps separate progress steps, and uses a separate coverage check when the combined result cannot be trusted or an older saved job needs it.

| Verdict | What Crebry does |
|---|---|
| Correct | Uses strong, sufficiently complete document evidence. |
| Ambiguous | Keeps useful document evidence and looks for missing information on the web. |
| Incorrect | Removes untrustworthy document evidence and tries web sources instead. |

Refinement selects original source sentences. Gemini uses that evidence to write a draft. Citation checks and, when enabled, grounding validation check the draft before publication.

If the answer fails validation, a planner can rewrite it, retrieve more documents, search the web, or stop. Retry and tool limits keep this process bounded. When evidence cannot support an answer, Crebry explains the gap rather than publishing unsupported claims.

Citations such as `[D1]` point to uploaded documents; `[W1]` points to a web source. Click a citation to inspect its source. Streaming text is a provisional draft until the checks finish.

## How the services connect

```mermaid
flowchart LR
    Browser[Next.js frontend on Vercel] -->|Session token| API[FastAPI backend on Render]
    Clerk[Clerk sign-in] --> Browser
    API --> Workflow[LangGraph research workflow]
    Workflow --> Gemini[Gemini AI]
    Workflow --> Tavily[Tavily web search]
    API --> Supabase[Supabase database and private file storage]
    Workflow --> Supabase
```

The frontend sends requests through its Next.js API proxy. The backend verifies the Clerk session and workspace ownership before accessing data. Provider secrets stay on the server.

Research runs in a saved background job. The frontend receives progress events and draft text as they arrive. Reconnecting can recover saved progress; an interrupted step may run again.

## Technology stack

| Tool | Role |
|---|---|
| Next.js, React, TypeScript | Pages, chat, document library, and workflow display |
| FastAPI | Backend API and progress streaming |
| LangGraph | Research steps, decisions, and correction paths |
| Gemini | Embeddings, evidence judgments, answer generation, and OCR |
| Tavily | Web search when document evidence needs help |
| Clerk | Sign-in and session verification |
| Supabase PostgreSQL and pgvector | Workspaces, conversations, research progress, and searchable passages |
| Supabase Storage | Private original documents |
| SQLite and Chroma | Local development alternatives |

## Run locally

You need **Python 3.12**, **Node.js 24**, npm, uv, and a Clerk application. Gemini is required for AI features. Tavily is optional for web search.

### Backend

From the repository root:

```powershell
cd backend
uv sync --frozen
```

Create `backend/.env` with these local settings and your own credentials:

```dotenv
ENVIRONMENT=development
GEMINI_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
EMBEDDING_PROVIDER=gemini
VECTOR_PROVIDER=chroma
STORAGE_PROVIDER=auto
DATA_DIR=data
CLERK_SECRET_KEY=your_clerk_secret_key
CLERK_ISSUER=https://your-clerk-instance-domain
FRONTEND_ORIGIN=http://localhost:3000
CLERK_AUTHORIZED_PARTIES=http://localhost:3000,http://127.0.0.1:3000
```

Leave `DATABASE_URL` unset to use SQLite. The checked-in `backend/.env.example` describes hosted Supabase settings, so use the settings above for local development.

Start the backend:

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend

In another terminal, from the repository root:

```powershell
cd frontend
npm ci
```

Create `frontend/.env.local` using the same Clerk application as the backend:

```dotenv
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
CLERK_SECRET_KEY=your_clerk_secret_key
BACKEND_API_URL=http://127.0.0.1:8000/api
```

Start the frontend:

```powershell
npm run dev
```

Open [localhost:3000](http://localhost:3000), sign in, and upload [the sample document](examples/research-notes.md). Restart the relevant service after changing its environment variables.

## Hosted setup

| Service | Configuration |
|---|---|
| Vercel | Frontend root: `frontend`. Set Clerk variables and `BACKEND_API_URL` to your Render backend URL ending in `/api`. |
| Render | Backend root: `backend`. Use `Dockerfile`, build context `.`, and health endpoint `/api/health`. |
| Supabase | Run `backend/supabase/setup.sql`, then configure the PostgreSQL Session pooler and private Storage bucket. |

Follow the [Supabase deployment guide](docs/supabase-deployment.md) for the complete environment settings. Keep database passwords and backend API keys out of Git and frontend public variables.

Supabase stores hosted user data; Render's temporary filesystem is only scratch space. Changing storage providers does not move existing data.

### Free hosting and paused projects

Supabase Free is a free plan with usage limits, not a one-month trial. Projects with low database activity over a week can pause. See [Supabase's pause policy](https://supabase.com/docs/guides/platform/free-project-pausing).

If the project pauses, resume it in Supabase, wait until it is active, and redeploy the backend on Render if it has not recovered. Check `/api/health` before using the app again. Render Free can also sleep when idle, so the first request may take longer.

Gemini and Tavily have separate quotas. The app selects eligible free-tier Gemini models, but your Google project's billing settings still determine charges. Use a project with billing disabled when you need free-only use.

## Where to read the code

| Path | What it does |
|---|---|
| `backend/app/api/routes.py` | Receives requests and checks workspace access |
| `backend/app/agents/graph.py` | Decides which research step runs next |
| `backend/app/agents/nodes.py` | Implements retrieval, grading, search, generation, and validation |
| `backend/app/agents/state.py` | Defines data passed between research steps |
| `backend/app/services/ai.py` | Calls Gemini and Tavily |
| `backend/app/services/routing.py` | Selects models and handles fallback |
| `backend/app/services/jobs.py` | Saves jobs, checkpoints, and progress events |
| `backend/app/services/documents.py` | Extracts and prepares document text |
| `frontend/lib/research-stream.ts` | Sends questions and receives progress |
| `frontend/stores/research.ts` | Holds the current workspace and chat state |
| `frontend/components/` | Displays chat, documents, citations, and the workflow |
