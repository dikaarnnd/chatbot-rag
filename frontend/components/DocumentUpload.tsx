"use client";

import { useCallback, useState } from "react";
import { uploadDocument, type IngestResponse } from "@/lib/api";

type UploadState = "idle" | "uploading" | "success" | "error";

interface DocumentUploadProps {
  /** Dipanggil setelah ingestion sukses -- dipakai parent (page.tsx) untuk
   * mengaktifkan ChatPanel, karena chat cuma valid kalau sudah ada dokumen
   * ter-index. */
  onUploadSuccess?: (result: IngestResponse) => void;
}

export default function DocumentUpload({ onUploadSuccess }: DocumentUploadProps) {
  const [state, setState] = useState<UploadState>("idle");
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setState("error");
        setError("File harus berformat .pdf");
        return;
      }
      setFileName(file.name);
      setState("uploading");
      setError(null);

      try {
        const res = await uploadDocument(file);
        setResult(res);
        setState("success");
        onUploadSuccess?.(res);
      } catch (err) {
        setState("error");
        setError(err instanceof Error ? err.message : "Gagal mengunggah dokumen.");
      }
    },
    [onUploadSuccess],
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div className="mx-auto w-full max-w-xl">
      <label
        htmlFor="pdf-upload"
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        className={`block cursor-pointer rounded-md border-2 border-dashed px-8 py-10 text-center transition-colors ${
          isDragging
            ? "border-(--color-accent) bg-(--color-accent-soft)"
            : "border-(--color-paper-line-strong) bg-(--color-paper-soft)"
        } ${state === "uploading" ? "pointer-events-none opacity-60" : ""}`}
      >
        <input
          id="pdf-upload"
          type="file"
          accept=".pdf"
          className="sr-only"
          onChange={onFileInput}
          disabled={state === "uploading"}
        />
        <p className="font-serif text-lg text-(--color-ink)">
          {state === "uploading" ? "Memproses dokumen..." : "Unggah dokumen"}
        </p>
        <p className="mt-1 text-sm text-(--color-ink-soft)">
          Seret PDF ke sini, atau klik untuk pilih file
        </p>
        {fileName && state !== "idle" && (
          <p className="mt-3 font-mono text-xs text-(--color-accent)">{fileName}</p>
        )}
      </label>

      {state === "uploading" && (
        <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-(--color-paper-line)">
          <div className="h-full w-full animate-pulse bg-(--color-accent) motion-reduce:animate-none" />
        </div>
      )}

      {state === "error" && error && (
        <p className="mt-3 text-sm text-(--color-danger)">{error}</p>
      )}

      {state === "success" && result && (
        <div className="mt-4 rounded-md border border-(--color-paper-line) bg-white p-5">
          <p className="font-serif text-base text-(--color-ink)">
            Dokumen siap ditanyai — <span className="font-mono text-sm">{result.file_name}</span>
          </p>
          <dl className="mt-3 grid grid-cols-3 gap-4 border-t border-(--color-paper-line) pt-3 font-mono text-sm">
            <div>
              <dt className="text-xs uppercase tracking-wide text-(--color-ink-soft)">Halaman</dt>
              <dd className="text-(--color-ink)">{result.pages_loaded}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-(--color-ink-soft)">Chunk</dt>
              <dd className="text-(--color-ink)">{result.chunks_created}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-(--color-ink-soft)">Waktu</dt>
              <dd className="text-(--color-ink)">{result.duration_seconds}s</dd>
            </div>
          </dl>
          <p className="mt-3 text-xs text-(--color-ink-soft)">
            Mengunggah dokumen baru akan menggantikan dokumen ini (satu dokumen aktif per sesi).
          </p>
        </div>
      )}
    </div>
  );
}