export interface NextTickBands {
  lo: number;
  med: number;
  hi: number;
}

export interface NextTickPrediction {
  p_down: number;
  p_flat: number;
  p_up: number;
  y_reg_hat: number;
  m_next_hat: number;
  bands: number[];
}

export async function getNextTick(symbol: string): Promise<NextTickPrediction> {
  const response = await fetch("/prediction/next-tick", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol }),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch next-tick prediction: ${response.status}`);
  }
  return (await response.json()) as NextTickPrediction;
}
