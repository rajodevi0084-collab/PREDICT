"use client";

import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts";

export type ActualPoint = {
  timestamp: string;
  close: number;
};

export type ActualSeries = {
  timebase: string;
  points: ActualPoint[];
};

export type PredictionPoint = {
  made_at: string;
  valid_at: string;
  predicted: number;
  probabilities?: Record<string, number>;
  actual_return?: number;
  timebase?: string;
};

export interface PriceChartProps {
  actualSeries: ActualSeries;
  predictionSeriesByHorizon: Record<number, PredictionPoint[]>;
  defaultHorizon?: number;
}

const COLORS = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"];

type TooltipMeta = {
  horizon: number;
  made_at: string;
  valid_at: string;
  predicted: number;
  probabilities?: Record<string, number>;
  actual_return?: number;
};

const PredictionTooltip: React.FC<any> = ({ label, payload, meta }: { label?: string; payload?: any[]; meta: Map<string, TooltipMeta[]> }) => {
  if (!label || !payload || payload.length === 0) {
    return null;
  }
  const entries = meta.get(label) ?? [];
  return (
    <div className="rounded border border-slate-300 bg-white p-2 text-xs shadow">
      <div className="font-semibold">{label}</div>
      {entries.map((entry) => {
        const probs = entry.probabilities ?? {};
        const up = probs.up ?? probs["1"] ?? 0;
        const flat = probs.flat ?? probs["0"] ?? 0;
        const down = probs.down ?? probs["-1"] ?? 0;
        return (
          <div key={`${entry.horizon}-${entry.valid_at}`} className="mt-1">
            <div className="font-medium">Horizon {entry.horizon}</div>
            <div>Made at: {entry.made_at}</div>
            <div>Valid at: {entry.valid_at}</div>
            <div>Predicted return: {(entry.predicted * 100).toFixed(2)}%</div>
            <div>
              Probabilities: ↑ {(up * 100).toFixed(1)}% | → {(flat * 100).toFixed(1)}% | ↓ {(down * 100).toFixed(1)}%
            </div>
            {typeof entry.actual_return === "number" && (
              <div>Actual return: {(entry.actual_return * 100).toFixed(2)}%</div>
            )}
          </div>
        );
      })}
    </div>
  );
};

const PriceChart: React.FC<PriceChartProps> = ({
  actualSeries,
  predictionSeriesByHorizon,
  defaultHorizon,
}) => {
  const horizonKeys = useMemo(() => Object.keys(predictionSeriesByHorizon).map(Number).sort((a, b) => a - b), [
    predictionSeriesByHorizon,
  ]);

  const chosenDefault = defaultHorizon ?? horizonKeys[horizonKeys.length - 1] ?? 0;

  const { dataset, tooltipMeta, predictionTimebase } = useMemo(() => {
    const base = new Map<string, { timestamp: string; close?: number }>();
    actualSeries.points.forEach((pt) => {
      base.set(pt.timestamp, { timestamp: pt.timestamp, close: pt.close });
    });

    const meta = new Map<string, TooltipMeta[]>();
    let timebase: string | undefined;

    horizonKeys.forEach((h) => {
      const horizonPoints = predictionSeriesByHorizon[h] ?? [];
      horizonPoints.forEach((point) => {
        const bucket = base.get(point.valid_at) ?? { timestamp: point.valid_at };
        (bucket as Record<string, unknown>)[`pred_h${h}`] = point.predicted;
        base.set(point.valid_at, bucket);
        const existing = meta.get(point.valid_at) ?? [];
        existing.push({
          horizon: h,
          made_at: point.made_at,
          valid_at: point.valid_at,
          predicted: point.predicted,
          probabilities: point.probabilities,
          actual_return: point.actual_return,
        });
        meta.set(point.valid_at, existing);
        if (!timebase && point.timebase) {
          timebase = point.timebase;
        }
      });
    });

    const combined = Array.from(base.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    return { dataset: combined, tooltipMeta: meta, predictionTimebase: timebase };
  }, [actualSeries.points, horizonKeys, predictionSeriesByHorizon]);

  const showWarning = predictionTimebase && predictionTimebase !== actualSeries.timebase;

  if (dataset.length === 0) {
    return <div className="text-sm text-slate-500">No data to display.</div>;
  }

  return (
    <div className="space-y-2">
      {showWarning && (
        <div className="inline-flex items-center rounded bg-yellow-100 px-2 py-1 text-xs text-yellow-800">
          Prediction timebase ({predictionTimebase}) differs from actual series ({actualSeries.timebase}).
        </div>
      )}
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={dataset} margin={{ top: 16, right: 24, left: 16, bottom: 16 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" minTickGap={32} />
          <YAxis yAxisId="left" stroke="#1f77b4" domain={["dataMin", "dataMax"]} />
          <Tooltip content={<PredictionTooltip meta={tooltipMeta} />} />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="close"
            stroke="#1f77b4"
            dot={false}
            name="Actual close"
          />
          {horizonKeys.map((h, idx) => (
            <Line
              key={h}
              yAxisId="left"
              type="linear"
              dataKey={`pred_h${h}`}
              stroke={COLORS[idx % COLORS.length]}
              dot={false}
              strokeWidth={h === chosenDefault ? 2 : 1}
              name={`Prediction h=${h}`}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;
