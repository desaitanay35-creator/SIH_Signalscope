import { apiClient, resolveApiUrl, toApiError } from "./client";
import type { AnalysisListResponse, AnalysisResponse } from "./types";

/** POST /api/v1/analyze - the only place a real forensic analysis is produced. */
export async function analyzeImage(
  file: File,
  caption?: string
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("image", file);
  if (caption) formData.append("caption", caption);

  try {
    // Let the browser set the multipart boundary; do not set Content-Type manually.
    const { data } = await apiClient.post<AnalysisResponse>(
      "/analyze",
      formData
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

/** GET /api/v1/analyses - paginated history, backed by the server, not localStorage. */
export async function listAnalyses(
  page = 1,
  pageSize = 20
): Promise<AnalysisListResponse> {
  try {
    const { data } = await apiClient.get<AnalysisListResponse>("/analyses", {
      params: { page, page_size: pageSize },
    });
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

/** GET /api/v1/analyses/{id} - a single past analysis, full detail. */
export async function getAnalysis(analysisId: string): Promise<AnalysisResponse> {
  try {
    const { data } = await apiClient.get<AnalysisResponse>(
      `/analyses/${encodeURIComponent(analysisId)}`
    );
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

/**
 * Resolves the absolute URL for an analysis's Grad-CAM heatmap JPEG.
 * Prefers the backend-provided `heatmap_url`; falls back to constructing it
 * from the known route only if the backend omitted it but reported the
 * heatmap as available.
 */
export function getHeatmapUrl(analysis: AnalysisResponse): string | null {
  if (!analysis.explanation.heatmap_available) return null;
  const relativeUrl =
    analysis.explanation.heatmap_url ??
    `/api/v1/analyses/${analysis.analysis_id}/heatmap`;
  return resolveApiUrl(relativeUrl);
}
