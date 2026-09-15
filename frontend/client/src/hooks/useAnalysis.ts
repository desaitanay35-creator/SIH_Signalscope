import { useCallback, useEffect, useRef, useState } from "react";
import { analyzeImage } from "../lib/api/analysis";
import type { ApiError, AnalysisResponse } from "../lib/api/types";

export type AnalysisStatus = "empty" | "ready" | "scanning" | "result" | "error";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

const SCAN_STAGES = [
  "Uploading image",
  "Preprocessing (RGB + frequency)",
  "Running fusion model",
  "Applying calibration",
  "Generating explanation",
];

/**
 * Shared real-analysis flow for Home and Analyze. Owns the upload -> POST
 * /api/v1/analyze -> result lifecycle. The progress bar/stage list are a
 * cosmetic loading indicator only (the backend is a single request/response,
 * not a streaming pipeline) - they never influence or fabricate the result.
 */
export function useAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [status, setStatus] = useState<AnalysisStatus>("empty");
  const [error, setError] = useState<ApiError | null>(null);
  const [validationError, setValidationError] = useState("");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState(0);

  const progressTimer = useRef<number | null>(null);
  const stageTimer = useRef<number | null>(null);
  const previewUrlRef = useRef<string>("");

  const clearTimers = () => {
    if (progressTimer.current) window.clearInterval(progressTimer.current);
    if (stageTimer.current) window.clearInterval(stageTimer.current);
    progressTimer.current = null;
    stageTimer.current = null;
  };

  useEffect(() => clearTimers, []);

  const revokePreview = () => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = "";
    }
  };

  const chooseFile = useCallback((selected?: File) => {
    setValidationError("");
    if (!selected) return;

    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setValidationError("Please choose a JPG, PNG, or WEBP image.");
      return;
    }
    if (selected.size > MAX_UPLOAD_BYTES) {
      setValidationError("Images must be smaller than 10 MB.");
      return;
    }

    revokePreview();
    const url = URL.createObjectURL(selected);
    previewUrlRef.current = url;

    setFile(selected);
    setPreview(url);
    setResult(null);
    setError(null);
    setStatus("ready");
  }, []);

  const analyze = useCallback(
    async (caption?: string) => {
      if (!file) return;

      setStatus("scanning");
      setError(null);
      setProgress(0);
      setStage(0);

      // Cosmetic-only progress: creeps toward 90% while the real request is
      // in flight, then snaps to 100% once the response actually arrives.
      progressTimer.current = window.setInterval(() => {
        setProgress(p => (p < 90 ? p + 3 : p));
      }, 200);
      stageTimer.current = window.setInterval(() => {
        setStage(s => (s < SCAN_STAGES.length - 1 ? s + 1 : s));
      }, 900);

      try {
        const response = await analyzeImage(file, caption);
        clearTimers();
        setProgress(100);
        setStage(SCAN_STAGES.length - 1);
        setResult(response);
        setStatus("result");
      } catch (err) {
        clearTimers();
        setError(err as ApiError);
        setStatus("error");
      }
    },
    [file]
  );

  const reset = useCallback(() => {
    clearTimers();
    revokePreview();
    setFile(null);
    setPreview("");
    setResult(null);
    setError(null);
    setValidationError("");
    setProgress(0);
    setStage(0);
    setStatus("empty");
  }, []);

  return {
    file,
    preview,
    status,
    error,
    validationError,
    result,
    progress,
    stage,
    stages: SCAN_STAGES,
    chooseFile,
    analyze,
    reset,
  };
}
