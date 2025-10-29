export interface UploadResponse {
  datasetId: string;
  message: string;
}

export interface RunSummary {
  id: string;
  status: string;
  createdAt: string;
  updatedAt?: string;
  accuracy?: number;
  loss?: number;
  [key: string]: unknown;
}

export interface TrainingResponse {
  runId: string;
  status: string;
  message?: string;
}

export interface PredictionResponse {
  predictionId: string;
  status: string;
  downloadUrl?: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_BASE_URL ?? API_BASE_URL.replace(/^http/, "ws");

type FetchOptions = RequestInit & { parseJson?: boolean };

async function request<T>(path: string, options: FetchOptions = {}): Promise<T> {
  const { parseJson = true, headers, ...rest } = options;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...rest,
    headers: {
      "Content-Type": "application/json",
      ...(headers ?? {})
    }
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  if (!parseJson) {
    // @ts-expect-error caller knows the type when parseJson is false
    return response as unknown as T;
  }

  return (await response.json()) as T;
}

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Upload failed with status ${response.status}`);
  }

  return (await response.json()) as UploadResponse;
}

export async function listRuns(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/runs", { method: "GET" });
}

export async function startTraining(datasetId: string): Promise<TrainingResponse> {
  return request<TrainingResponse>("/train", {
    method: "POST",
    body: JSON.stringify({ datasetId })
  });
}

export async function promoteRun(runId: string): Promise<TrainingResponse> {
  return request<TrainingResponse>(`/runs/${runId}/promote`, {
    method: "POST"
  });
}

export async function fetchRunMetrics(runId: string): Promise<Record<string, number>> {
  return request<Record<string, number>>(`/runs/${runId}/metrics`, { method: "GET" });
}

export async function triggerPrediction(runId: string): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({ runId })
  });
}

export async function downloadPrediction(predictionId: string): Promise<Blob> {
  const response = await request<Response>(`/predictions/${predictionId}/download`, {
    method: "GET",
    parseJson: false
  });
  return response.blob();
}

export function createRunWebSocket(runId: string): WebSocket {
  const url = `${WS_BASE_URL}/ws/runs/${runId}`;
  return new WebSocket(url);
}
