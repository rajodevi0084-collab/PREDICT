import { useEffect, useMemo, useState } from "react";
import type { NextBarHistoryPoint, NextBarPrediction } from "../state/prediction";
import { getNextBar } from "../state/prediction";

type Props = {
  symbol: string;
};

function formatPct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatTime(value: string | null) {
  return value ?? "—";
}

export function NextBarPanel({ symbol }: Props) {
  const [prediction, setPrediction] = useState<NextBarPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [smooth, setSmooth] = useState(false);
  const [hovered, setHovered] = useState<NextBarHistoryPoint | null>(null);

  useEffect(() => {
    let mounted = true;
    getNextBar(symbol)
      .then((data) => {
        if (mounted) {
          setPrediction(data);
          setError(null);
        }
      })
      .catch((err) => {
        if (mounted) {
          setError(err.message);
        }
      });
    return () => {
      mounted = false;
    };
  }, [symbol]);

  const history = useMemo<NextBarHistoryPoint[]>(() => {
    if (!prediction) {
      return [];
    }
    if (prediction.history && prediction.history.length > 0) {
      return prediction.history;
    }
    return [
      {
        obs_time: prediction.obs_time,
        target_time: prediction.target_time,
        next_close_hat: prediction.next_close_hat,
        p_down: prediction.p_down,
        p_flat: prediction.p_flat,
        p_up: prediction.p_up,
      },
    ];
  }, [prediction]);

  const displaySeries = useMemo(() => {
    if (!smooth || history.length < 2) {
      return history;
    }
    return history.map((point, idx) => {
      const start = Math.max(0, idx - 2);
      const window = history.slice(start, idx + 1);
      const average = window.reduce((acc, item) => acc + item.next_close_hat, 0) / window.length;
      return { ...point, next_close_hat: average };
    });
  }, [history, smooth]);

  const chartPoints = useMemo(() => {
    if (displaySeries.length === 0) {
      return [];
    }
    const width = 360;
    const height = 160;
    const padding = 16;
    const closes = displaySeries.map((point) => point.next_close_hat);
    const minClose = Math.min(...closes);
    const maxClose = Math.max(...closes);
    const span = maxClose - minClose || 1;
    return displaySeries.map((point, idx) => {
      const fraction = displaySeries.length > 1 ? idx / (displaySeries.length - 1) : 0.5;
      const x = padding + fraction * (width - padding * 2);
      const y = height - padding - ((point.next_close_hat - minClose) / span) * (height - padding * 2);
      return { ...point, x, y };
    });
  }, [displaySeries]);

  if (error) {
    return <div role="alert">{error}</div>;
  }

  if (!prediction || !prediction.bands) {
    return <div>Loading…</div>;
  }

  return (
    <section aria-label="next-bar">
      <h3>Next-Bar (OHLCV)</h3>
      <p>
        Observed at <strong>{formatTime(prediction.obs_time)}</strong> → target <strong>{formatTime(prediction.target_time)}</strong>
      </p>
      <div>
        <strong>Probabilities:</strong>
        <ul>
          <li>Down: {formatPct(prediction.p_down)}</li>
          <li>Flat: {formatPct(prediction.p_flat)}</li>
          <li>Up: {formatPct(prediction.p_up)}</li>
        </ul>
      </div>
      <label style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem" }}>
        <input
          type="checkbox"
          checked={smooth}
          onChange={(event) => setSmooth(event.target.checked)}
        />
        Smoothing (visual only)
      </label>
      <svg
        role="img"
        aria-label="Next close forecast"
        width={360}
        height={160}
        viewBox="0 0 360 160"
        onMouseLeave={() => setHovered(null)}
        style={{ display: "block", marginTop: "1rem", border: "1px solid #ccc" }}
      >
        <polyline
          fill="none"
          stroke="#0055aa"
          strokeWidth={2}
          points={chartPoints.map((point) => `${point.x},${point.y}`).join(" ")}
        />
        {chartPoints.map((point, idx) => (
          <circle
            key={`${point.target_time ?? "pt"}-${idx}`}
            cx={point.x}
            cy={point.y}
            r={4}
            fill="#ff6600"
            onMouseEnter={() => setHovered(point)}
          />
        ))}
      </svg>
      {hovered && (
        <div role="tooltip" style={{ marginTop: "0.5rem" }}>
          <div>
            <strong>
              {formatTime(hovered.obs_time)} → {formatTime(hovered.target_time)}
            </strong>
          </div>
          <div>Next close: {hovered.next_close_hat.toFixed(4)}</div>
          <div>
            p↓ {formatPct(hovered.p_down)}, p↔ {formatPct(hovered.p_flat)}, p↑ {formatPct(hovered.p_up)}
          </div>
        </div>
      )}
      <p>Expected next close: {prediction.next_close_hat.toFixed(4)}</p>
      <p>Log-return estimate: {prediction.y_reg_hat.toFixed(6)}</p>
      <p>
        Conformal band: {prediction.bands.lo.toFixed(4)} – {prediction.bands.hi.toFixed(4)} (mid {prediction.bands.med.toFixed(4)})
      </p>
    </section>
  );
}
