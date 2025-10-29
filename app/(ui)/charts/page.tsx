"use client";

import { useEffect, useState } from "react";
import { fetchRunMetrics, listRuns, type RunSummary } from "../../lib/api";
import {
  PredictionChart,
  type PredictionChartPoint
} from "../../components/PredictionChart";

export default function ChartsPage() {
  const [runs, setRuns] = useState<PredictionChartPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const runSummaries = await listRuns();
        const withMetrics = await Promise.all(
          runSummaries.map(async (run: RunSummary) => {
            try {
              const metrics = await fetchRunMetrics(run.id);
              return {
                runId: run.id,
                createdAt: run.createdAt,
                accuracy: metrics.accuracy ?? (run as RunSummary).accuracy,
                loss: metrics.loss ?? (run as RunSummary).loss
              } satisfies PredictionChartPoint;
            } catch (err) {
              console.warn(`Unable to fetch metrics for run ${run.id}`, err);
              return {
                runId: run.id,
                createdAt: run.createdAt,
                accuracy: run.accuracy,
                loss: run.loss
              } satisfies PredictionChartPoint;
            }
          })
        );
        setRuns(withMetrics);
        setError(null);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to load run metrics";
        setError(message);
      } finally {
        setLoading(false);
      }
    }

    load().catch(() => undefined);
  }, []);

  return (
    <section className="card">
      <h2>Run Metrics</h2>
      <p>
        Compare accuracy and loss across training runs. The chart updates automatically whenever new
        run metrics are available.
      </p>
      {loading && <p className="info">Loading metrics…</p>}
      {error && (
        <div className="error">
          <strong>Unable to load metrics.</strong>
          <p>{error}</p>
        </div>
      )}
      {!loading && !error && <PredictionChart runs={runs} />}
    </section>
  );
}
