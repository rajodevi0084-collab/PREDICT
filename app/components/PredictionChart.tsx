"use client";

import { useMemo } from "react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend
} from "recharts";

export interface PredictionChartPoint {
  runId: string;
  accuracy?: number;
  loss?: number;
  createdAt: string;
}

export interface PredictionChartProps {
  runs: PredictionChartPoint[];
}

export function PredictionChart({ runs }: PredictionChartProps) {
  const data = useMemo(
    () =>
      runs.map((run) => ({
        runId: run.runId,
        createdAt: new Date(run.createdAt).toLocaleString(),
        accuracy: run.accuracy ?? null,
        loss: run.loss ?? null
      })),
    [runs]
  );

  if (data.length === 0) {
    return <p className="empty">No run metrics available for charting.</p>;
  }

  return (
    <div className="chart-card">
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={data} margin={{ top: 24, right: 32, left: 16, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
          <XAxis dataKey="createdAt" stroke="#a0aec0" tick={{ fontSize: 12 }} angle={-25} textAnchor="end" height={80} />
          <YAxis stroke="#a0aec0" tick={{ fontSize: 12 }} domain={[0, 1]} allowDecimals />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
            itemStyle={{ color: "#f9fafb" }}
          />
          <Legend wrapperStyle={{ color: "#f9fafb" }} />
          <Line type="monotone" dataKey="accuracy" stroke="#34d399" strokeWidth={3} dot={false} name="Accuracy" />
          <Line type="monotone" dataKey="loss" stroke="#60a5fa" strokeWidth={3} dot={false} name="Loss" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
