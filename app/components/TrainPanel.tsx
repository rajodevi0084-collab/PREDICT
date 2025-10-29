"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createRunWebSocket,
  fetchRunMetrics,
  listRuns,
  promoteRun,
  startTraining,
  type RunSummary
} from "../lib/api";

type PanelState = "idle" | "loading" | "updating";

interface RunWithMetrics extends RunSummary {
  metrics?: Record<string, number>;
}

export function TrainPanel() {
  const [runs, setRuns] = useState<RunWithMetrics[]>([]);
  const [panelState, setPanelState] = useState<PanelState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>("");

  const loadRuns = useCallback(async () => {
    setPanelState("loading");
    try {
      const data = await listRuns();
      setRuns(data);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load runs";
      setError(message);
    } finally {
      setPanelState("idle");
    }
  }, []);

  useEffect(() => {
    loadRuns().catch(() => undefined);
  }, [loadRuns]);

  const subscribeToRun = useCallback((runId: string) => {
    const socket = createRunWebSocket(runId);
    socket.onmessage = async () => {
      try {
        const metrics = await fetchRunMetrics(runId);
        setRuns((prev) =>
          prev.map((run) => (run.id === runId ? { ...run, metrics } : run))
        );
      } catch (err) {
        console.error("Failed to refresh metrics", err);
      }
    };
    socket.onerror = (event) => {
      console.error("Run socket error", event);
    };
    socket.onclose = () => {
      // nothing for now
    };
    return () => socket.close();
  }, []);

  const onStartTraining = useCallback(async () => {
    if (!selectedDataset) {
      setError("Select a dataset identifier before starting training.");
      return;
    }
    setPanelState("updating");
    try {
      const response = await startTraining(selectedDataset);
      setError(null);
      await loadRuns();
      subscribeToRun(response.runId);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to start training";
      setError(message);
    } finally {
      setPanelState("idle");
    }
  }, [loadRuns, selectedDataset, subscribeToRun]);

  const onPromote = useCallback(
    async (runId: string) => {
      setPanelState("updating");
      try {
        await promoteRun(runId);
        await loadRuns();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to promote run";
        setError(message);
      } finally {
        setPanelState("idle");
      }
    },
    [loadRuns]
  );

  const decoratedRuns = useMemo(
    () =>
      runs
        .slice()
        .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1)),
    [runs]
  );

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Training Runs</h2>
        <button
          type="button"
          className="refresh-button"
          onClick={() => loadRuns().catch(() => undefined)}
          disabled={panelState !== "idle"}
        >
          Refresh
        </button>
      </div>
      <div className="form-row">
        <label className="field">
          <span>Dataset ID</span>
          <input
            type="text"
            value={selectedDataset}
            onChange={(event) => setSelectedDataset(event.target.value)}
            placeholder="Enter dataset identifier"
          />
        </label>
        <button
          type="button"
          className="primary"
          onClick={onStartTraining}
          disabled={panelState === "updating"}
        >
          Start Training
        </button>
      </div>
      {error && (
        <div className="error">
          <strong>Something went wrong.</strong>
          <p>{error}</p>
        </div>
      )}
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Status</th>
              <th>Accuracy</th>
              <th>Loss</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {decoratedRuns.map((run) => (
              <tr key={run.id}>
                <td>{run.id}</td>
                <td>{run.status}</td>
                <td>{run.accuracy ?? run.metrics?.accuracy ?? "—"}</td>
                <td>{run.loss ?? run.metrics?.loss ?? "—"}</td>
                <td>{new Date(run.createdAt).toLocaleString()}</td>
                <td>
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => onPromote(run.id)}
                    disabled={panelState === "updating"}
                  >
                    Promote
                  </button>
                </td>
              </tr>
            ))}
            {decoratedRuns.length === 0 && (
              <tr>
                <td colSpan={6} className="empty">
                  No runs found. Upload a dataset and start training to see runs here.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
