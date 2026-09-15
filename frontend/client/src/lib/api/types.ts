/**
 * Types mirror backend/app/schemas/analysis.py and backend/app/schemas/common.py
 * exactly. Do not add fields the backend does not return.
 */

export type Verdict = "likely_ai_generated" | "likely_real";

export interface CueItem {
  type: string;
  description: string;
  confidence: number;
}

export interface ExplanationResponse {
  summary: string | null;
  cues: CueItem[];
  heatmap_url: string | null;
  heatmap_available: boolean;
}

export interface ImageMetadataResponse {
  format: string;
  width: number;
  height: number;
  file_size: number;
  has_exif: boolean;
  exif_summary: Record<string, unknown> | null;
  c2pa_detected: boolean;
  c2pa_verified: boolean;
  has_c2pa: boolean;
  note: string;
}

export interface AnalysisResponse {
  analysis_id: string;
  verdict: Verdict;
  confidence: number;
  threshold: number;
  model_version: string;
  processing_time_ms: number;
  is_calibrated: boolean;
  calibration_method: string | null;
  explanation: ExplanationResponse;
  metadata: ImageMetadataResponse;
  created_at: string;
}

export interface AnalysisListItemResponse {
  analysis_id: string;
  filename: string;
  verdict: Verdict;
  confidence: number;
  model_version: string;
  processing_time_ms: number;
  created_at: string;
}

export interface AnalysisListResponse {
  items: AnalysisListItemResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface ModelInfoResponse {
  name: string;
  version: string;
  task: string;
  loaded: boolean;
  device: string | null;
  architecture: string | null;
  checkpoint_path: string | null;
}

/** backend/app/schemas/common.py error envelope */
export interface ErrorPayload {
  code: string;
  message: string;
  details: unknown;
}

export interface ErrorResponse {
  error: ErrorPayload;
}

/** Normalized shape used throughout the frontend, regardless of failure source. */
export interface ApiError {
  code: string;
  message: string;
  status: number | null;
  details: unknown;
}
