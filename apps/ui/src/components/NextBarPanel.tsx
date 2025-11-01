import { useEffect, useMemo, useState } from "react";
import type { NextBarPrediction } from "../state/prediction";
import { getNextBar } from "../state/prediction";

type Props = {
  symbol: string;
};

function formatPct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function NextBarPanel({ symbol }: Props) {
  const [prediction, setPrediction] = useState<NextBarPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  const bands = useMemo(() => {
    if (!prediction) {
      return null;
    }
    const [lo, mid, hi] = prediction.bands;
    return { lo, mid, hi };
  }, [prediction]);

  if (error) {
    return <div role="alert">{error}</div>;
  }

  if (!prediction || !bands) {
    return <div>Loading…</div>;
  }

  return (
    <section aria-label="next-bar">
      <h3>Next-Bar (OHLCV)</h3>
      <div>
        <strong>Probabilities:</strong>
        <ul>
          <li>Down: {formatPct(prediction.p_down)}</li>
          <li>Flat: {formatPct(prediction.p_flat)}</li>
          <li>Up: {formatPct(prediction.p_up)}</li>
        </ul>
      </div>
      <p>Expected next close: {prediction.next_close_hat.toFixed(4)}</p>
      <p>Log-return estimate: {prediction.y_reg_hat.toFixed(6)}</p>
      <p>
        Conformal band: {bands.lo.toFixed(4)} – {bands.hi.toFixed(4)} (mid {bands.mid.toFixed(4)})
      </p>
    </section>
  );
}
