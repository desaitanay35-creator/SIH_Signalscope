import axios, { type AxiosError } from "axios";
import type { ApiError, ErrorResponse } from "./types";

/**
 * Single source of truth for the backend base URL. Never hardcode
 * localhost/127.0.0.1 in components - derive from this client instead.
 */
export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
  "http://127.0.0.1:8000";

export const API_V1_PREFIX = "/api/v1";

export const apiClient = axios.create({
  baseURL: `${API_BASE_URL}${API_V1_PREFIX}`,
});

/** Resolves a backend-relative URL (e.g. an explanation.heatmap_url) against the API host. */
export function resolveApiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? "" : "/"}${path}`;
}

/**
 * Normalizes any Axios failure - a backend error envelope, a non-JSON HTTP
 * error, or a network-level failure with no response at all - into one
 * consistent shape the UI can render without knowing which case occurred.
 */
export function toApiError(err: unknown): ApiError {
  if (axios.isAxiosError(err)) {
    const axiosErr = err as AxiosError<ErrorResponse>;
    const status = axiosErr.response?.status ?? null;
    const payload = axiosErr.response?.data?.error;

    if (payload?.code && payload?.message) {
      return {
        code: payload.code,
        message: payload.message,
        status,
        details: payload.details ?? null,
      };
    }

    if (!axiosErr.response) {
      return {
        code: "NETWORK_ERROR",
        message:
          "Could not reach the SignalScope server. Check your connection and try again.",
        status: null,
        details: null,
      };
    }

    return {
      code: `HTTP_${status}`,
      message: "An unexpected server error occurred.",
      status,
      details: null,
    };
  }

  return {
    code: "UNKNOWN_ERROR",
    message: "An unexpected error occurred.",
    status: null,
    details: null,
  };
}
