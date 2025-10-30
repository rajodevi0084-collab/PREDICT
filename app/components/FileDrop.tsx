"use client";

import { useCallback, useRef, useState } from "react";
import { uploadFile, type UploadResponse } from "../lib/api";

export interface FileDropProps {
  onUploadComplete?: (response: UploadResponse) => void;
}

type Status = "idle" | "uploading" | "success" | "error";

export function FileDrop({ onUploadComplete }: FileDropProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<UploadResponse | null>(null);

  const reset = useCallback(() => {
    setStatus("idle");
    setError(null);
    setResponse(null);
  }, []);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) {
        return;
      }
      const file = files[0];
      reset();
      setStatus("uploading");

      if (!/(csv|parquet|pq)$/i.test(file.name)) {
        setError("Only CSV and Parquet files are supported.");
        setStatus("error");
        return;
      }

      try {
        const result = await uploadFile(file);
        setResponse(result);
        setStatus("success");
        onUploadComplete?.(result);
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err.message : "Upload failed");
      }
    },
    [onUploadComplete, reset]
  );

  const openFileDialog = useCallback(() => {
    inputRef.current?.click();
  }, []);

  return (
    <section className="file-drop">
      <div
        className="drop-zone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          handleFiles(event.dataTransfer.files);
        }}
        role="button"
        tabIndex={0}
        onClick={openFileDialog}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            openFileDialog();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.parquet"
          className="sr-only"
          onChange={(event) => handleFiles(event.target.files)}
        />
        <p>Drag & drop a CSV or Parquet file here, or click to browse.</p>
        <p className="hint">Files are normalised to UTC timestamps and uppercase symbols.</p>
      </div>

      {status === "uploading" && <p className="info">Uploading and parsing dataset…</p>}
      {status === "error" && error && <p className="error">{error}</p>}

      {status === "success" && response && (
        <div className="upload-summary">
          <h3>Upload complete</h3>
          <dl>
            <div>
              <dt>Dataset ID</dt>
              <dd>{response.file_id}</dd>
            </div>
            <div>
              <dt>Rows</dt>
              <dd>{response.rows.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Symbols</dt>
              <dd>{response.symbols.join(", ") || "—"}</dd>
            </div>
            <div>
              <dt>Date range</dt>
              <dd>
                {response.date_min && response.date_max
                  ? `${new Date(response.date_min).toLocaleString()} → ${new Date(response.date_max).toLocaleString()}`
                  : "—"}
              </dd>
            </div>
          </dl>
          <div className="preview-table">
            <table>
              <thead>
                <tr>
                  {Object.keys(response.preview[0] ?? {}).map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {response.preview.map((row, index) => (
                  <tr key={`${response.file_id}-${index}`}>
                    {Object.entries(row).map(([key, value]) => (
                      <td key={key}>{renderValue(value)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}

function renderValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  return String(value);
}
