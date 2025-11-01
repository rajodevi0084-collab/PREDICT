import { useEffect, useState } from "react";
import type { NextTickPrediction } from "../state/prediction";
import { getNextTick } from "../state/prediction";

type Props = {
  symbol: string;
};

export function NextTickPanel({ symbol }: Props) {
  const [prediction, setPrediction] = useState<NextTickPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    getNextTick(symbol)
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

  if (error) {
    return <div role="alert">{error}</div>;
  }

  if (!prediction) {
    return <div>Loading…</div>;
  }

  return (
    <div>
      <h3>Next-Tick Prediction</h3>
      <p>
        Probabilities — Down: {prediction.p_down.toFixed(2)}, Flat: {prediction.p_flat.toFixed(2)}, Up: {prediction.p_up.toFixed(2)}
      </p>
      <p>Next mid price: {prediction.m_next_hat.toFixed(4)}</p>
      <p>Regression head: {prediction.y_reg_hat.toFixed(6)}</p>
    </div>
  );
}
