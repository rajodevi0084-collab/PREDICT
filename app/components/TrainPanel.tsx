"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getCatalog,
  listRuns,
  openWS,
  promoteRun,
  startTrain,
  type CatalogItem,
  type TrainRun,
  type TrainStartRequest,
} from "../lib/api";

interface RunsState {
  runs: TrainRun[];
  active: string | null;
}

interface LogMessage {
  timestamp: string;
  event: string;
  payload: Record<string, unknown>;
}

const ENFORCED_HORIZON = 1;

const DEFAULT_FORM: TrainStartRequest = {
  files: [],
  symbols: [],
  horizon: ENFORCED_HORIZON,
  epochs: 25,
  feature_budget: 128,
  coverage: 0.75,
};

export function TrainPanel() {
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [runsState, setRunsState] = useState<RunsState>({ runs: [], active: null });
  const [form, setForm] = useState<TrainStartRequest>(DEFAULT_FORM);
  const [symbolsInput, setSymbolsInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const refreshCatalog = useCallback(async () => {
    try {
      const response = await getCatalog();
      setCatalog(response.items);
    } catch (err) {
      console.error("Failed to load catalog", err);
      setError("Unable to fetch dataset catalog.");
    }
  }, []);

  const refreshRuns = useCallback(async () => {
    try {
      const response = await listRuns();
      setRunsState(response);
    } catch (err) {
      console.error("Failed to load runs", err);
      setError("Unable to fetch training runs.");
    }
  }, []);

  useEffect(() => {
    void refreshCatalog();
    void refreshRuns();
    return () => {
      wsRef.current?.close();
    };
  }, [refreshCatalog, refreshRuns]);

  const handleFileSelection = useCallback((event: React.ChangeEvent<HTMLSelectElement>) => {
    const selected = Array.from(event.target.selectedOptions).map((option) => option.value);
    setForm((prev) => ({ ...prev, files: selected }));
  }, []);

  const handleNumberChange = useCallback(
    (key: keyof TrainStartRequest) =>
      (event: React.ChangeEvent<HTMLInputElement>) => {
        const value = event.target.valueAsNumber;
        setForm((prev) => ({ ...prev, [key]: Number.isFinite(value) ? value : prev[key] }));
      },
    []
  );

  const connectLogs = useCallback((runId: string) => {
    wsRef.current?.close();
    const socket = openWS(runId);
    wsRef.current = socket;
    setLogs([]);

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as { timestamp?: string; event?: string; payload?: Record<string, unknown> };
        setLogs((prev) => [
          {
            timestamp: data.timestamp ?? new Date().toISOString(),
            event: data.event ?? "log",
            payload: data.payload ?? {},
          },
          ...prev,
        ]);
      } catch (err) {
        console.error("Failed to parse log message", err);
      }
      void refreshRuns();
    };

    socket.onerror = (event) => {
      console.error("WebSocket error", event);
    };

    socket.onclose = () => {
      wsRef.current = null;
    };
  }, [refreshRuns]);

  const handleStart = useCallback(async () => {
    if (form.files.length === 0) {
      setError("Select at least one dataset to start training.");
      return;
    }
    const symbols = symbolsInput
      .split(/[,\n]/)
      .map((item) => item.trim().toUpperCase())
      .filter(Boolean);

    setLoading(true);
    setError(null);
    try {
      const payload: TrainStartRequest = {
        ...form,
        symbols: symbols.length ? symbols : undefined,
      };
      const response = await startTrain(payload);
      connectLogs(response.run_id);
      await refreshRuns();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start training";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [connectLogs, form, refreshRuns, symbolsInput]);

  const handlePromote = useCallback(
    async (runId: string) => {
      setLoading(true);
      setError(null);
      try {
        await promoteRun(runId);
        await refreshRuns();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to promote run";
        setError(message);
      } finally {
        setLoading(false);
      }
    },
    [refreshRuns]
  );

  const sortedRuns = useMemo(() => {
    return runsState.runs
      .slice()
      .sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  }, [runsState.runs]);

  const primaryCoverageKey = useMemo(() => {
    const first = sortedRuns.find((run) => run.metrics && Object.keys(run.metrics).some((key) => key.startsWith("acc@")));
    if (!first) {
      return "acc@75";
    }
    const candidate = Object.keys(first.metrics ?? {}).find((key) => key.startsWith("acc@"));
    return candidate ?? "acc@75";
  }, [sortedRuns]);

  return (
    <section className="train-panel">
      <header className="panel-header">
        <h2>Start a training run</h2>
        <div className="actions">
          <button type="button" onClick={() => void refreshCatalog()} disabled={loading}>
            Refresh datasets
          </button>
          <button type="button" onClick={() => void refreshRuns()} disabled={loading}>
            Refresh runs
          </button>
        </div>
      </header>

      <div className="form-grid">
        <label className="field">
          <span>Datasets</span>
          <select multiple value={form.files} onChange={handleFileSelection}>
            {catalog.map((item) => (
              <option key={item.file_id} value={item.file_id}>
                {item.filename} ({item.rows.toLocaleString()} rows)
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Symbols (comma separated)</span>
          <textarea
            value={symbolsInput}
            onChange={(event) => setSymbolsInput(event.target.value)}
            placeholder="Leave empty for all symbols"
          />
        </label>
        <label className="field">
          <span>Horizon</span>
          <input
            type="number"
            min={1}
            value={ENFORCED_HORIZON}
            disabled
            aria-readonly
          />
          <small style={{ color: "#555" }}>H=1 enforced for next-bar OHLCV.</small>
        </label>
        <label className="field">
          <span>Epochs</span>
          <input type="number" min={1} value={form.epochs} onChange={handleNumberChange("epochs")} />
        </label>
        <label className="field">
          <span>Feature budget</span>
          <input type="number" min={32} step={16} value={form.feature_budget} onChange={handleNumberChange("feature_budget")} />
        </label>
        <label className="field">
          <span>Coverage target</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={form.coverage}
            onChange={handleNumberChange("coverage")}
          />
        </label>
      </div>

      <div className="form-actions">
        <button type="button" className="primary" onClick={handleStart} disabled={loading}>
          Start training
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      <section className="logs">
        <h3>Live logs</h3>
        {logs.length === 0 ? (
          <p className="hint">Run training to stream log messages here.</p>
        ) : (
          <ul>
            {logs.slice(0, 50).map((log, index) => (
              <li key={`${log.timestamp}-${index}`}>
                <span className="timestamp">{new Date(log.timestamp).toLocaleTimeString()}</span>
                <span className="event">{log.event}</span>
                <code>{JSON.stringify(log.payload)}</code>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="runs">
        <h3>Training runs</h3>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Acc@C</th>
                <th>Coverage</th>
                <th>Brier</th>
                <th>Updated</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {sortedRuns.map((run) => {
                const metrics = run.metrics ?? {};
                const accuracy = metrics[primaryCoverageKey];
                const coverageKey = primaryCoverageKey.replace("acc@", "coverage@");
                const coverage = metrics[coverageKey];
                return (
                  <tr key={run.id} className={run.id === runsState.active ? "active-run" : undefined}>
                    <td>{run.id}</td>
                    <td>{run.status ?? "unknown"}</td>
                    <td>{formatMetric(accuracy)}</td>
                    <td>{formatMetric(coverage)}</td>
                    <td>{formatMetric(metrics.brier)}</td>
                    <td>{run.updated_at ? new Date(run.updated_at).toLocaleString() : "—"}</td>
                    <td>
                      <button type="button" onClick={() => handlePromote(run.id)} disabled={loading}>
                        Promote
                      </button>
                    </td>
                  </tr>
                );
              })}
              {sortedRuns.length === 0 && (
                <tr>
                  <td colSpan={7} className="empty">
                    No runs yet. Start a training job to see results.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}

function formatMetric(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(3);
}
