"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface PredictionChartDatum {
  timestamp: string;
  y_true: number | null;
  y_pred_price: number;
  abstain?: boolean;
}

export interface PredictionChartProps {
  data: PredictionChartDatum[];
  timezone?: string;
}

export function PredictionChart({ data, timezone = "Asia/Kolkata" }: PredictionChartProps) {
  const sorted = useMemo(() => {
    return data
      .slice()
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .map((point) => ({
        ...point,
        label: new Intl.DateTimeFormat("en-IN", {
          timeZone: timezone,
          hour: "2-digit",
          minute: "2-digit",
          month: "short",
          day: "2-digit",
        }).format(new Date(point.timestamp)),
      }));
  }, [data, timezone]);

  const abstainAreas = useMemo(() => {
    const areas: Array<{ start: string; end: string }> = [];
    let startIndex: number | null = null;
    sorted.forEach((point, index) => {
      if (point.abstain) {
        if (startIndex === null) {
          startIndex = index;
        }
      } else if (startIndex !== null) {
        areas.push({ start: sorted[startIndex].label, end: sorted[index - 1].label });
        startIndex = null;
      }
    });
    if (startIndex !== null) {
      areas.push({ start: sorted[startIndex].label, end: sorted[sorted.length - 1].label });
    }
    return areas;
  }, [sorted]);

  if (sorted.length === 0) {
    return <p className="hint">No prediction data available.</p>;
  }

  return (
    <div className="chart-card">
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={sorted} margin={{ top: 24, right: 32, left: 16, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
          <XAxis dataKey="label" stroke="#a0aec0" tick={{ fontSize: 12 }} angle={-25} textAnchor="end" height={80} />
          <YAxis stroke="#a0aec0" tick={{ fontSize: 12 }} domain={["auto", "auto"]} allowDecimals />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
            itemStyle={{ color: "#f9fafb" }}
            labelFormatter={(value) => value}
          />
          {abstainAreas.map((area) => (
            <ReferenceArea
              key={`${area.start}-${area.end}`}
              x1={area.start}
              x2={area.end}
              y1={Number.MIN_SAFE_INTEGER}
              y2={Number.MAX_SAFE_INTEGER}
              fill="rgba(252, 211, 77, 0.12)"
              ifOverflow="extendDomain"
            />
          ))}
          <Line type="monotone" dataKey="y_true" stroke="#60a5fa" strokeWidth={3} dot={false} name="Actual" />
          <Line type="monotone" dataKey="y_pred_price" stroke="#f97316" strokeWidth={3} dot={false} name="Prediction" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
