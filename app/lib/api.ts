export interface UploadResponse {
  file_id: string;
  filename: string;
  stored_as: string;
  rows: number;
  symbols: string[];
  date_min?: string;
  date_max?: string;
  dtypes: Record<string, string>;
  preview: Record<string, unknown>[];
}

export interface CatalogItem {
  file_id: string;
  filename: string;
  rows: number;
  symbols: string[];
  date_min?: string;
  date_max?: string;
  dtypes: Record<string, string>;
}

export interface CatalogResponse {
  items: CatalogItem[];
}

export interface TrainRun {
  id: string;
  status?: string;
  created_at: string;
  updated_at?: string;
  promoted_at?: string;
  metrics?: Record<string, number>;
  metadata?: Record<string, unknown>;
}

export interface TrainRunsResponse {
  runs: TrainRun[];
  active: string | null;
}

export interface TrainStartRequest {
  files: string[];
  symbols?: string[];
  horizon: number;
  epochs: number;
  feature_budget: number;
  coverage: number;
}

export interface TrainStartResponse {
  run_id: string;
}

export interface PromoteResponse {
  run_id: string;
  status: string;
}

export interface PredictPreviewRow {
  timestamp: string;
  symbol: string;
  y_true?: number | null;
  y_pred_price: number;
  p_up: number;
  p_down: number;
  margin: number;
  abstain: boolean;
  tau: number;
  coverage_target: number;
  run_id: string;
  [key: string]: unknown;
}

export interface PredictResponse {
  run_id: string;
  num_rows: number;
  tau: number;
  coverage_target: number;
  temperature: number;
  feature_columns: string[];
  predictions_path: string;
  manifest_path: string;
  artifacts: Record<string, string>;
  dataset: {
    files: string[];
    rows: number;
    symbols?: string[];
    timestamp?: { min: string; max: string };
  };
  preview: PredictPreviewRow[];
}

export interface LatestPredictionResponse extends PredictResponse {
  generated_at?: string;
  path?: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_BASE_URL ?? API_BASE_URL.replace(/^http/, "ws");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request to ${path} failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE_URL}/data/upload`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Upload failed with status ${response.status}`);
  }

  return (await response.json()) as UploadResponse;
}

export async function getCatalog(): Promise<CatalogResponse> {
  return request<CatalogResponse>("/data/catalog");
}

export async function previewDataset(fileId: string): Promise<UploadResponse> {
  return request<UploadResponse>(`/data/preview/${fileId}`);
}

export async function startTrain(payload: TrainStartRequest): Promise<TrainStartResponse> {
  return request<TrainStartResponse>("/train/start", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function listRuns(): Promise<TrainRunsResponse> {
  return request<TrainRunsResponse>("/train/runs");
}

export async function promoteRun(runId: string): Promise<PromoteResponse> {
  return request<PromoteResponse>("/train/promote", {
    method: "POST",
    body: JSON.stringify({ run_id: runId })
  });
}

export interface PredictRequestPayload {
  files: string[];
  run_id?: string;
  symbols?: string[];
  start?: string;
  end?: string;
  coverage_target?: number;
  tau?: number;
}

export async function runPredict(payload: PredictRequestPayload): Promise<PredictResponse> {
  return request<PredictResponse>("/predict/run", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function getLatestPrediction(): Promise<LatestPredictionResponse> {
  return request<LatestPredictionResponse>("/predict/latest");
}

export function openWS(runId: string): WebSocket {
  const base = WS_BASE_URL.endsWith("/") ? WS_BASE_URL.slice(0, -1) : WS_BASE_URL;
  return new WebSocket(`${base}/ws/train/${runId}`);
}
