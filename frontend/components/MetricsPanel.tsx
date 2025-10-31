"use client";

import React, { useEffect, useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

type MetricsResponse = {
  hit_rate_costed: number;
  directional_accuracy_by_h: Record<string, number>;
  sharpe: number;
  sortino: number;
  profit_factor: number;
  mdd: number;
  turnover: number;
  coverage: number;
  avg_slippage_bp: number;
  coverage_vs_return: Array<{ coverage: number; return: number }>;
  window_json_paths: string[];
  trades_jsonl_path: string;
};

const formatPct = (value: number | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "–";

const formatNumber = (value: number | undefined, digits = 2) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "–";

const MetricsPanel: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/report")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to load metrics: ${res.status}`);
        }
        return res.json();
      })
      .then((data: MetricsResponse) => {
        if (!cancelled) {
          setMetrics(data);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const coverageCurve = useMemo(() => metrics?.coverage_vs_return ?? [], [metrics]);

  if (error) {
    return <div className="text-sm text-red-600">{error}</div>;
  }

  if (!metrics) {
    return <div className="text-sm text-slate-500">Loading metrics…</div>;
  }

  return (
    <div className="space-y-4 rounded border border-slate-200 bg-white p-4 shadow-sm">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div>
          <div className="text-xs uppercase text-slate-500">Hit-rate after costs</div>
          <div className="text-lg font-semibold">{formatPct(metrics.hit_rate_costed)}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-slate-500">Sharpe</div>
          <div className="text-lg font-semibold">{formatNumber(metrics.sharpe)}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-slate-500">Sortino</div>
          <div className="text-lg font-semibold">{formatNumber(metrics.sortino)}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-slate-500">Profit Factor</div>
          <div className="text-lg font-semibold">{formatNumber(metrics.profit_factor)}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-slate-500">Max Drawdown</div>
          <div className="text-lg font-semibold">{formatPct(metrics.mdd)}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-slate-500">Turnover</div>
          <div className="text-lg font-semibold">{formatNumber(metrics.turnover)}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-slate-500">Coverage</div>
          <div className="text-lg font-semibold">{formatPct(metrics.coverage)}</div>
        </div>
        <div>
          <div className="text-xs uppercase text-slate-500">Avg Slippage (bp)</div>
          <div className="text-lg font-semibold">{formatNumber(metrics.avg_slippage_bp)}</div>
        </div>
      </div>

      <div>
        <div className="text-sm font-semibold">Directional accuracy by horizon</div>
        <div className="mt-2 grid grid-cols-2 gap-2 md:grid-cols-4">
          {Object.entries(metrics.directional_accuracy_by_h).map(([h, value]) => (
            <div key={h} className="rounded border border-slate-200 p-2 text-sm">
              <div className="text-xs uppercase text-slate-500">h={h}</div>
              <div className="font-semibold">{formatPct(value)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="h-64">
        <div className="mb-2 text-sm font-semibold">Coverage vs Return</div>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={coverageCurve} margin={{ top: 8, right: 16, bottom: 8, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="coverage" tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
            <YAxis tickFormatter={(value) => `${(value * 100).toFixed(1)}%`} />
            <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} labelFormatter={(label: number) => `Coverage ${(label * 100).toFixed(1)}%`} />
            <Line type="monotone" dataKey="return" stroke="#2563eb" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {metrics.window_json_paths.map((path) => (
          <a key={path} href={path} className="rounded bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700 hover:bg-slate-200" download>
            Download {path.split("/").pop()}
          </a>
        ))}
        <a
          href={metrics.trades_jsonl_path}
          className="rounded bg-blue-100 px-3 py-1 text-xs font-medium text-blue-700 hover:bg-blue-200"
          download
        >
          Download trades JSONL
        </a>
      </div>
    </div>
  );
};

export default MetricsPanel;
