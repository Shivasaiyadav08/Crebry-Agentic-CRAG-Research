"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FileText,
  FolderOpen,
  HardDrive,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { API_URL, api } from "@/lib/api";
import { formatSize } from "@/lib/utils";
import { useResearch } from "@/stores/research";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import type { DocumentRecord, Workspace } from "@/types";

type UploadResult = {
  document?: DocumentRecord;
  duplicate?: boolean;
  name?: string;
  error?: string;
};

export function Documents() {
  const { workspaceId, health, uploading, streaming, update } = useResearch();
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<{
    name: string;
    percent: number;
    stage: string;
  } | null>(null);
  const [notices, setNotices] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<DocumentRecord | null>(null);
  const [deleting, setDeleting] = useState<DocumentRecord | null>(null);
  const [busy, setBusy] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const xhrRef = useRef<XMLHttpRequest | null>(null);
  const reload = useCallback(async () => {
    if (!workspaceId) {
      setDocuments([]);
      setLoading(false);
      return;
    }
    try {
      setDocuments(
        await api<DocumentRecord[]>(`/workspaces/${workspaceId}/documents`),
      );
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Could not load documents.",
      );
    } finally {
      setLoading(false);
    }
  }, [workspaceId]);
  useEffect(() => {
    void reload();
  }, [reload]);
  useEffect(
    () => () => {
      xhrRef.current?.abort();
      useResearch.getState().update({ uploading: false });
    },
    [],
  );

  async function refreshWorkspaceCounts() {
    try {
      const workspaces = await api<Workspace[]>("/workspaces");
      const state = useResearch.getState();

      if (state.workspaceId === workspaceId) {
        state.update({ workspaces });
      }
    } catch (error) {
      console.error("Failed to refresh workspace counts:", error);
    }
  }

  async function upload(files: FileList | File[]) {
    if (uploading || streaming) return;
    if (!workspaceId) {
      setError("Please select or create a workspace before uploading documents.");
      return;
    }
    const items = Array.from(files);
    if (!items.length) return;
    if (items.length > 10) {
      setError("Choose up to 10 files at a time.");
      return;
    }
    const limit = health?.max_upload_mb ?? 10;
    const invalid = items.find(
      (file) =>
        !/\.(pdf|docx|txt|md|markdown)$/i.test(file.name) ||
        !file.size ||
        file.size > limit * 1024 * 1024,
    );
    if (invalid) {
      setError(
        `“${invalid.name}” is empty, unsupported, or larger than ${limit} MB. Choose PDF, DOCX, TXT, or Markdown.`,
      );
      return;
    }
    update({ uploading: true });
    setError("");
    setNotices([]);
    for (const file of items) {
      setProgress({ name: file.name, percent: 0, stage: "Uploading" });
      try {
        const results = await new Promise<UploadResult[]>((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          xhrRef.current = xhr;
          xhr.open("POST", `${API_URL}/workspaces/${workspaceId}/documents`);
          xhr.upload.onprogress = (event) => {
            if (event.lengthComputable)
              setProgress({
                name: file.name,
                percent: Math.round((event.loaded / event.total) * 100),
                stage: "Uploading",
              });
          };
          xhr.upload.onload = () =>
            setProgress({
              name: file.name,
              percent: 100,
              stage: "Extracting, embedding & indexing",
            });
          xhr.onload = () => {
            try {
              const body = JSON.parse(xhr.responseText);
              if (xhr.status >= 200 && xhr.status < 300) resolve(body);
              else
                reject(
                  new Error(
                    typeof body.detail === "string"
                      ? body.detail
                      : "Upload failed.",
                  ),
                );
            } catch {
              reject(new Error("Invalid backend response."));
            }
          };
          xhr.onerror = () => reject(new Error("Cannot reach the backend."));
          xhr.onabort = () => reject(new Error("Upload cancelled."));
          const form = new FormData();
          form.append("files", file);
          xhr.send(form);
        });
        for (const result of results) {
          setNotices((previous) => [
            ...previous,
            result.error
              ? `${file.name}: ${result.error}`
              : result.duplicate
                ? `${file.name}: already in this workspace.`
                : result.document?.status === "error"
                  ? `${file.name}: ${result.document.error}`
                  : `${file.name}: indexed ${result.document?.chunk_count} chunks.`,
          ]);
        }
      } catch (error) {
        setNotices((previous) => [
          ...previous,
          error instanceof Error ? error.message : "Upload failed.",
        ]);
      }
    }
    xhrRef.current = null;
    update({ uploading: false });
    setProgress(null);
    if (fileInput.current) fileInput.current.value = "";
    await reload();
  }
  async function act(
    document: DocumentRecord,
    action: "preview" | "reindex" | "delete",
  ) {
    setBusy(document.id);
    setError("");
    try {
      const path = `/workspaces/${workspaceId}/documents/${document.id}`;
      if (action === "preview") setPreview(await api<DocumentRecord>(path));
      if (action === "reindex") {
        await api(path + "/reindex", { method: "POST" });
        await reload();
      }
      if (action === "delete") {
        await api(path, { method: "DELETE" });
        setDeleting(null);
        await reload();
      }
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Document action failed.",
      );
    } finally {
      setBusy("");
    }
  }
  const filtered = documents.filter((document) =>
    document.name.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <div className="page-content documents-view">
      <div className="page-heading">
        <div>
          <div className="eyebrow">Workspace sources</div>
          <h1>Knowledge library</h1>
          <p>Keep the material behind your research in one place.</p>
        </div>
        <Button
          disabled={!workspaceId || uploading || streaming}
          onClick={() => fileInput.current?.click()}
        >
          <Plus size={16} /> Add documents
        </Button>
      </div>
      <input
        ref={fileInput}
        type="file"
        multiple
        accept=".pdf,.docx,.txt,.md,.markdown"
        className="sr-only"
        aria-label="Upload documents"
        onChange={(event) => {
          if (event.target.files) void upload(event.target.files);
        }}
      />
      <div className="library-stats">
        <div>
          <FileText size={18} />
          <strong>{documents.length}</strong>
          <span>Documents</span>
        </div>
        <div>
          <FolderOpen size={18} />
          <strong>
            {documents.reduce((sum, document) => sum + document.chunk_count, 0)}
          </strong>
          <span>Indexed chunks</span>
        </div>
        <div>
          <HardDrive size={18} />
          <strong>
            {formatSize(
              documents.reduce((sum, document) => sum + document.size, 0),
            )}
          </strong>
          <span>Source files · local</span>
        </div>
      </div>
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${!workspaceId || uploading || streaming ? "disabled" : ""}`}
        role="button"
        tabIndex={!workspaceId || uploading || streaming ? -1 : 0}
        aria-label="Upload research material. Click or drag files into this area."
        onClick={() => {
          if (!workspaceId) {
            setError("Please select or create a workspace before uploading documents.");
            return;
          }
          if (!uploading && !streaming) {
            fileInput.current?.click();
          }
        }}
        onKeyDown={(event) => {
          if ((event.key === "Enter" || event.key === " ") && !uploading && !streaming) {
            event.preventDefault();
            if (!workspaceId) {
              setError("Please select or create a workspace before uploading documents.");
              return;
            }
            fileInput.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          if (!uploading && !streaming && workspaceId) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!workspaceId) {
            setError("Please select or create a workspace before uploading documents.");
            return;
          }
          void upload(event.dataTransfer.files);
        }}
      >
        <div className="upload-icon">
          {uploading ? (
            <LoaderCircle size={24} className="spin" />
          ) : (
            <Upload size={24} />
          )}
        </div>
        {progress ? (
          <>
            <strong>{progress.name}</strong>
            <p>
              {progress.stage}
              {progress.stage === "Uploading"
                ? ` · ${progress.percent}%`
                : " · please wait"}
            </p>
            <div
              className={`upload-progress ${progress.stage !== "Uploading" ? "indexing" : ""}`}
            >
              <div style={{ width: `${progress.percent}%` }} />
            </div>
            <small>
              Transfer progress is measured. Indexing has no estimated
              percentage.
            </small>
          </>
        ) : (
          <>
            <strong>Add material to your research</strong>
            <p>
              Click or drag files into this area, or{" "}
              <button
                type="button"
                className="text-button"
                disabled={!workspaceId || streaming}
                onClick={(event) => {
                  event.stopPropagation();
                  fileInput.current?.click();
                }}
              >
                browse files
              </button>
            </p>
            <small>
              PDF, DOCX, TXT, Markdown · up to {health?.max_upload_mb ?? 10} MB
              each
            </small>
          </>
        )}
      </div>
      {error && (
        <div className="alert" role="alert">
          {error}
          <button onClick={() => setError("")} aria-label="Dismiss error">
            <X size={14} />
          </button>
        </div>
      )}
      {notices.length > 0 && (
        <div className="upload-notices" role="status">
          {notices.map((notice, i) => (
            <p key={i}>{notice}</p>
          ))}
        </div>
      )}
      <div className="library-toolbar">
        <h2>
          All documents <span>{documents.length}</span>
        </h2>
        <div className="search-field">
          <Search size={15} />
          <input
            placeholder="Find a document…"
            aria-label="Search documents"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
      </div>
      {loading ? (
        <div className="empty-state">
          <LoaderCircle className="spin" size={24} />
          <p>Loading your library…</p>
        </div>
      ) : filtered.length ? (
        <div className="document-grid">
          {filtered.map((document) => (
            <div className="document-card" key={document.id}>
              <div className="document-card-top">
                <div
                  className={`file-icon ${document.name.endsWith(".pdf") ? "pdf" : ""}`}
                >
                  <FileText size={23} />
                </div>
                <span className={`document-status ${document.status}`}>
                  <span />
                  {document.status === "ready" ? "Indexed" : document.status}
                </span>
              </div>
              <button
                className="document-title"
                onClick={() => void act(document, "preview")}
                disabled={!!busy}
              >
                {document.name}
              </button>
              <p>
                {formatSize(document.size)}
                <span>·</span>
                {document.chunk_count} chunks
              </p>
              {document.error && (
                <small className="document-error">{document.error}</small>
              )}
              <div className="document-card-bottom">
                <span>
                  {new Date(document.created_at).toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                  })}
                </span>
                <div>
                  {busy === document.id ? (
                    <LoaderCircle className="spin" size={15} />
                  ) : (
                    <>
                      <Button
                        variant="ghost"
                        size="icon"
                        disabled={!!busy || uploading || streaming}
                        onClick={() => void act(document, "reindex")}
                        aria-label={`Reindex ${document.name}`}
                      >
                        <RefreshCw size={14} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        disabled={!!busy || uploading || streaming}
                        onClick={() => setDeleting(document)}
                        aria-label={`Delete ${document.name}`}
                      >
                        <Trash2 size={14} />
                      </Button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-icon">
            <FileText size={25} />
          </div>
          <h3>
            {search
              ? "No matching documents"
              : "Every good answer starts with a source"}
          </h3>
          <p>
            {search
              ? "Try another filename."
              : "Upload your first document to give Crebry a knowledge base to synthesize."}
          </p>
          <span>PDF, DOCX, text, and Markdown files</span>
        </div>
      )}
      <Dialog
        open={!!preview}
        onOpenChange={(open) => {
          if (!open) setPreview(null);
        }}
      >
        <DialogContent className="document-preview">
          <DialogTitle>{preview?.name}</DialogTitle>
          <DialogDescription>
            Extracted text · {preview?.chunk_count} indexed chunks
            {preview?.preview_truncated
              ? " · preview limited to 100,000 characters"
              : ""}
          </DialogDescription>
          <pre>{preview?.text}</pre>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!deleting}
        onOpenChange={(open) => {
          if (!open) setDeleting(null);
        }}
      >
        <DialogContent>
          <DialogTitle>Delete this document?</DialogTitle>
          <DialogDescription>
            “{deleting?.name}” and its indexed chunks will be removed. Existing
            answers retain their saved source excerpts.
          </DialogDescription>
          <div className="dialog-actions">
            <Button variant="ghost" onClick={() => setDeleting(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={!!busy}
              onClick={() => deleting && void act(deleting, "delete")}
            >
              <Trash2 size={14} /> Delete document
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
