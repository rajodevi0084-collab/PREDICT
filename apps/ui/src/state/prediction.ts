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

export type NextBarBands = {
  lo: number;
  med: number;
  hi: number;
};

export type NextBarPoint = {
  symbol: string;
  horizon: number;
  obs_time: string;
  target_time: string;
  c_t: number;
  y_reg_hat: number;
  next_close_hat: number;
  p_down: number;
  p_flat: number;
  p_up: number;
  bands: NextBarBands;
};

export type NextBarPrediction = NextBarPoint & {
  history?: NextBarPoint[];
};

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

export async function getNextBar(symbol: string): Promise<NextBarPrediction> {
  const response = await fetch("/prediction/next-bar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol }),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch next-bar prediction: ${response.status}`);
  }
  return (await response.json()) as NextBarPrediction;
}
