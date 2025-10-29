"use client";

import { useCallback, useEffect, useState } from "react";
import {
  downloadPrediction,
  listRuns,
  triggerPrediction,
  type PredictionResponse,
  type RunSummary
} from "../../lib/api";

interface DownloadablePrediction extends PredictionResponse {
  blobUrl?: string;
}

export default function PredictPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [prediction, setPrediction] = useState<DownloadablePrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRuns()
      .then((data) => {
        setRuns(data);
        if (data.length > 0) {
          setSelectedRun(data[0].id);
        }
      })
      .catch((err) => {
        const message = err instanceof Error ? err.message : "Failed to load runs";
        setError(message);
      });
  }, []);

  const onPredict = useCallback(async () => {
    if (!selectedRun) {
      setError("Select a trained run before requesting predictions.");
      return;
    }
    setState("loading");
    setError(null);
    try {
      const response = await triggerPrediction(selectedRun);
      let blobUrl: string | undefined;
      if (response.downloadUrl) {
        blobUrl = response.downloadUrl;
      } else {
        const blob = await downloadPrediction(response.predictionId);
        blobUrl = URL.createObjectURL(blob);
      }
      setPrediction({ ...response, blobUrl });
      setState("ready");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Prediction request failed";
      setError(message);
      setState("error");
    }
  }, [selectedRun]);

  const onDownload = useCallback(() => {
    if (!prediction?.blobUrl) {
      return;
    }
    const link = document.createElement("a");
    link.href = prediction.blobUrl;
    link.download = `${prediction.predictionId}.json`;
    link.click();
  }, [prediction]);

  return (
    <section className="card">
      <h2>Generate Predictions</h2>
      <p>
        Select a trained run below and request predictions from the backend. When the prediction is
        ready you can download the generated output.
      </p>
      <div className="form-row">
        <label className="field">
          <span>Run ID</span>
          <select value={selectedRun} onChange={(event) => setSelectedRun(event.target.value)}>
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {run.id} — {run.status}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="primary"
          onClick={onPredict}
          disabled={state === "loading"}
        >
          Request Prediction
        </button>
      </div>
      {error && (
        <div className="error">
          <strong>Unable to fetch predictions.</strong>
          <p>{error}</p>
        </div>
      )}
      {state === "loading" && <p className="info">Requesting prediction…</p>}
      {state === "ready" && prediction && (
        <div className="success">
          <p>Prediction is ready! ID: {prediction.predictionId}</p>
          <button type="button" className="secondary" onClick={onDownload}>
            Download Prediction
          </button>
        </div>
      )}
    </section>
  );
}
