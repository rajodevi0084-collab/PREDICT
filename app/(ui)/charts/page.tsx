"use client";

import { useEffect, useMemo, useState } from "react";
import { getLatestPrediction, type LatestPredictionResponse } from "../../lib/api";
import { PredictionChart } from "../../components/PredictionChart";

export default function ChartsPage() {
  const [latest, setLatest] = useState<LatestPredictionResponse | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("ALL");
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const response = await getLatestPrediction();
      setLatest(response);
      const symbols = Array.from(new Set(response.preview.map((row) => row.symbol))).sort();
      setSelectedSymbol(symbols[0] ?? "ALL");
      setError(null);
    } catch (err) {
      setError("Failed to load latest predictions");
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const chartData = useMemo(() => {
    if (!latest) {
      return [];
    }
    return latest.preview
      .filter((row) => selectedSymbol === "ALL" || row.symbol === selectedSymbol)
      .map((row) => ({
        timestamp: row.timestamp,
        y_true: typeof row.y_true === "number" ? row.y_true : null,
        y_pred_price: row.y_pred_price,
        abstain: Boolean(row.abstain),
      }));
  }, [latest, selectedSymbol]);

  const symbols = useMemo(() => {
    if (!latest) {
      return [];
    }
    return Array.from(new Set(latest.preview.map((row) => row.symbol))).sort();
  }, [latest]);

  return (
    <div className="container">
      <header className="panel-header">
        <h2>Prediction charts</h2>
        <button type="button" onClick={() => void refresh()}>
          Refresh
        </button>
      </header>
      {error && <p className="error">{error}</p>}
      {!latest && !error && <p className="hint">Run a prediction to see charts.</p>}
      {latest && (
        <section className="charts-section">
          <div className="metadata">
            <p>
              Run <strong>{latest.run_id}</strong> · τ={latest.tau.toFixed(3)} · coverage target={latest.coverage_target.toFixed(2)}
            </p>
            <p>Generated at {latest.generated_at ? new Date(latest.generated_at).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }) : "unknown"}</p>
          </div>
          {symbols.length > 0 && (
            <label className="field inline">
              <span>Symbol</span>
              <select value={selectedSymbol} onChange={(event) => setSelectedSymbol(event.target.value)}>
                <option value="ALL">All symbols</option>
                {symbols.map((symbol) => (
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
                  {Object.keys(latest.preview[0] ?? {}).map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {latest.preview.slice(0, 100).map((row, index) => (
                  <tr key={`${row.timestamp}-${index}`}>
                    {Object.entries(row).map(([key, value]) => (
                      <td key={key}>{formatCell(value)}</td>
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

function formatCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(4);
  }
  return String(value);
}
