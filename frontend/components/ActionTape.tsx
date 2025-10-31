"use client";

import React, { useMemo, useState } from "react";

type Action = {
  decision_at: string;
  valid_at: string;
  horizon: number;
  action: "Flat" | "Long1" | "Long2" | "Short1" | "Short2";
  bucket?: string;
  conf: number;
  pnl_after_costs: number;
  price?: number;
  qty?: number;
  costs?: Record<string, number>;
};

export interface ActionTapeProps {
  actions: Action[];
}

const COLOR_MAP: Record<Action["action"], string> = {
  Flat: "#9ca3af",
  Long1: "#4ade80",
  Long2: "#15803d",
  Short1: "#f87171",
  Short2: "#b91c1c",
};

const formatBreakdown = (action: Action) => {
  const parts: string[] = [];
  if (typeof action.price === "number") {
    parts.push(`Price: ${action.price.toFixed(2)}`);
  }
  if (typeof action.qty === "number") {
    parts.push(`Qty: ${action.qty}`);
  }
  if (action.costs) {
    Object.entries(action.costs).forEach(([k, v]) => {
      parts.push(`${k}: ${v.toFixed(2)}`);
    });
  }
  parts.push(`Confidence: ${(action.conf * 100).toFixed(1)}%`);
  parts.push(`PnL after costs: ${action.pnl_after_costs.toFixed(2)}`);
  return parts.join("\n");
};

const ActionTape: React.FC<ActionTapeProps> = ({ actions }) => {
  const [hovered, setHovered] = useState<Action | null>(null);
  const timeline = useMemo(() => {
    if (actions.length === 0) {
      return null;
    }
    const parsed = actions.map((action) => ({
      ...action,
      decisionTs: new Date(action.decision_at).getTime(),
      validTs: new Date(action.valid_at).getTime(),
    }));
    const minTs = Math.min(...parsed.map((p) => p.decisionTs));
    const maxTs = Math.max(...parsed.map((p) => p.validTs));
    const span = Math.max(1, maxTs - minTs);
    return parsed.map((p) => ({
      action: p,
      left: ((p.decisionTs - minTs) / span) * 100,
      width: ((p.validTs - p.decisionTs) / span) * 100,
    }));
  }, [actions]);

  if (!timeline) {
    return <div className="text-xs text-slate-500">No actions executed.</div>;
  }

  return (
    <div className="space-y-2">
      <div className="relative h-12 w-full rounded border border-slate-200 bg-slate-50">
        {timeline.map(({ action, left, width }) => (
          <div
            key={`${action.decision_at}-${action.action}-${action.valid_at}`}
            className="absolute top-1/4 h-1/2 rounded"
            style={{
              left: `${left}%`,
              width: `${Math.max(width, 0.5)}%`,
              backgroundColor: COLOR_MAP[action.action],
            }}
            onMouseEnter={() => setHovered(action)}
            onMouseLeave={() => setHovered(null)}
            title={formatBreakdown(action)}
          />
        ))}
      </div>
      {hovered && (
        <div className="rounded border border-slate-200 bg-white p-2 text-xs shadow">
          <div className="font-semibold">
            {hovered.action} · Horizon {hovered.horizon} · {hovered.decision_at} → {hovered.valid_at}
          </div>
          <div>Confidence: {(hovered.conf * 100).toFixed(1)}%</div>
          <div>PnL after costs: {hovered.pnl_after_costs.toFixed(2)}</div>
          {hovered.bucket && <div>Bucket: {hovered.bucket}</div>}
          {hovered.costs && (
            <div className="mt-1 space-y-0.5">
              {Object.entries(hovered.costs).map(([k, v]) => (
                <div key={k}>
                  {k}: {v.toFixed(2)}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ActionTape;
