"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getCatalog,
  listRuns,
  runPredict,
  type CatalogItem,
  type PredictRequestPayload,
  type PredictResponse,
  type TrainRun,
} from "../../lib/api";
import { PredictionChart } from "../../components/PredictionChart";

interface RunsState {
  runs: TrainRun[];
  active: string | null;
}

export default function PredictPage() {
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [runsState, setRunsState] = useState<RunsState>({ runs: [], active: null });
  const [files, setFiles] = useState<string[]>([]);
  const [runId, setRunId] = useState<string>("active");
  const [symbols, setSymbols] = useState<string>("");
  const [start, setStart] = useState<string>("");
  const [end, setEnd] = useState<string>("");
  const [coverageTarget, setCoverageTarget] = useState<number | undefined>(undefined);
  const [tau, setTau] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("ALL");

  useEffect(() => {
    void (async () => {
      try {
        const [catalogResponse, runsResponse] = await Promise.all([getCatalog(), listRuns()]);
        setCatalog(catalogResponse.items);
        setRunsState(runsResponse);
      } catch (err) {
        setError("Failed to initialise prediction form.");
      }
    })();
  }, []);

  const handleSubmit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (files.length === 0) {
        setError("Select at least one dataset to run inference.");
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const payload: PredictRequestPayload = {
          files,
          run_id: runId === "active" ? undefined : runId,
        };
        if (symbols.trim()) {
          payload.symbols = symbols
            .split(/[,\n]/)
            .map((item) => item.trim().toUpperCase())
            .filter(Boolean);
        }
        if (start) {
          payload.start = start;
        }
        if (end) {
          payload.end = end;
        }
        if (coverageTarget !== undefined) {
          payload.coverage_target = coverageTarget;
        }
        if (tau !== undefined) {
          payload.tau = tau;
        }
        const response = await runPredict(payload);
        setResult(response);
        const symbolsInPreview = Array.from(new Set(response.preview.map((row) => row.symbol))).sort();
        setSelectedSymbol(symbolsInPreview[0] ?? "ALL");
      } catch (err) {
        const message = err instanceof Error ? err.message : "Prediction request failed";
        setError(message);
      } finally {
        setLoading(false);
      }
    },
    [coverageTarget, end, files, runId, symbols, start, tau]
  );

  const chartData = useMemo(() => {
    if (!result) {
      return [];
    }
    return result.preview
      .filter((row) => selectedSymbol === "ALL" || row.symbol === selectedSymbol)
      .map((row) => ({
        timestamp: row.timestamp,
        y_true: typeof row.y_true === "number" ? row.y_true : null,
        y_pred_price: row.y_pred_price,
        abstain: Boolean(row.abstain),
      }));
  }, [result, selectedSymbol]);

  const availableSymbols = useMemo(() => {
    if (!result) {
      return [];
    }
    return Array.from(new Set(result.preview.map((row) => row.symbol))).sort();
  }, [result]);

  return (
    <div className="container">
      <h2>Generate predictions</h2>
      <form className="form-grid" onSubmit={handleSubmit}>
        <label className="field">
          <span>Datasets</span>
          <select multiple value={files} onChange={(event) => setFiles(Array.from(event.target.selectedOptions, (option) => option.value))}>
            {catalog.map((item) => (
              <option key={item.file_id} value={item.file_id}>
                {item.filename} ({item.rows.toLocaleString()} rows)
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Model run</span>
          <select value={runId} onChange={(event) => setRunId(event.target.value)}>
            <option value="active">Active run</option>
            {runsState.runs.map((run) => (
              <option key={run.id} value={run.id}>
                {run.id} ({run.status ?? "unknown"})
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Symbols (optional)</span>
          <textarea value={symbols} onChange={(event) => setSymbols(event.target.value)} placeholder="Comma separated" />
        </label>
        <label className="field">
          <span>Start (UTC)</span>
          <input type="datetime-local" value={start} onChange={(event) => setStart(event.target.value)} />
        </label>
        <label className="field">
          <span>End (UTC)</span>
          <input type="datetime-local" value={end} onChange={(event) => setEnd(event.target.value)} />
        </label>
        <label className="field">
          <span>Coverage target</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={coverageTarget ?? ""}
            onChange={(event) => setCoverageTarget(event.target.value === "" ? undefined : Number(event.target.value))}
          />
        </label>
        <label className="field">
          <span>τ threshold</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={tau ?? ""}
            onChange={(event) => setTau(event.target.value === "" ? undefined : Number(event.target.value))}
          />
        </label>
        <div className="form-actions">
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Running…" : "Run prediction"}
          </button>
        </div>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="prediction-results">
          <header className="results-header">
            <h3>Results</h3>
            <div className="artifacts">
              {Object.entries(result.artifacts).map(([key, href]) => (
                <a key={key} href={href} className="secondary" target="_blank" rel="noreferrer">
                  Download {key}
                </a>
              ))}
            </div>
          </header>
          <p className="hint">
            τ = {result.tau.toFixed(3)}, coverage target = {result.coverage_target.toFixed(2)}, temperature = {result.temperature.toFixed(2)}
          </p>
          {availableSymbols.length > 0 && (
            <label className="field inline">
              <span>Chart symbol</span>
              <select value={selectedSymbol} onChange={(event) => setSelectedSymbol(event.target.value)}>
                <option value="ALL">All symbols</option>
                {availableSymbols.map((symbol) => (
                  <option key={symbol} value={symbol}>
                    {symbol}
                  </option>
                ))}
              </select>
            </label>
          )}
          <PredictionChart data={chartData} />
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  {Object.keys(result.preview[0] ?? {}).map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.preview.slice(0, 100).map((row, index) => (
                  <tr key={`${row.timestamp}-${index}`}>
                    {Object.entries(row).map(([key, value]) => (
                      <td key={key}>{renderCell(value)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

function renderCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }
  return String(value);
}
