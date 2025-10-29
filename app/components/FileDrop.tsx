"use client";

import { useCallback, useRef, useState } from "react";
import { uploadDataset, type UploadResponse } from "../lib/api";

type UploadState = "idle" | "uploading" | "success" | "error";

export interface FileDropProps {
  onUploadComplete?: (response: UploadResponse) => void;
}

export function FileDrop({ onUploadComplete }: FileDropProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [state, setState] = useState<UploadState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<UploadResponse | null>(null);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) {
        return;
      }

      const file = files[0];
      setState("uploading");
      setError(null);

      try {
        const response = await uploadDataset(file);
        setState("success");
        setLastResponse(response);
        onUploadComplete?.(response);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Upload failed";
        setState("error");
        setError(message);
      }
    },
    [onUploadComplete]
  );

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      handleFiles(event.dataTransfer.files);
    },
    [handleFiles]
  );

  const onSelectFile = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      handleFiles(event.target.files);
    },
    [handleFiles]
  );

  const onBrowse = useCallback(() => {
    inputRef.current?.click();
  }, []);

  return (
    <div>
      <div
        className="drop-zone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            onBrowse();
          }
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.json,.parquet"
          className="drop-input"
          onChange={onSelectFile}
        />
        <p className="drop-message">
          Drag & drop your dataset file here, or <span className="drop-link">browse</span> to upload.
        </p>
        <p className="drop-hint">Supported formats: CSV, JSON, Parquet</p>
      </div>
      {state === "uploading" && <p className="info">Uploading dataset…</p>}
      {state === "success" && lastResponse && (
        <div className="success">
          <strong>Upload complete!</strong>
          <p>Dataset ID: {lastResponse.datasetId}</p>
          {lastResponse.message && <p>{lastResponse.message}</p>}
        </div>
      )}
      {state === "error" && error && (
        <div className="error">
          <strong>Upload failed:</strong>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}
