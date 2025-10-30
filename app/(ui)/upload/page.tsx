"use client";

import { useState } from "react";
import { FileDrop } from "../../components/FileDrop";
import type { UploadResponse } from "../../lib/api";

export default function UploadPage() {
  const [result, setResult] = useState<UploadResponse | null>(null);

  return (
    <div className="container">
      <h2>Upload datasets</h2>
      <p className="hint">Upload CSV or Parquet files. Columns are normalised to UTC timestamps and uppercase symbols.</p>
      <FileDrop onUploadComplete={setResult} />
      {result && (
        <aside className="callout">
          <h3>Next steps</h3>
          <p>
            Dataset <strong>{result.file_id}</strong> is ready. Head to the training page to start a model run with this file.
          </p>
        </aside>
      )}
    </div>
  );
}
