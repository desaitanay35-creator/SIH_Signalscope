import { BarChart3, FileSearch, Info, RotateCcw, ShieldCheck } from "lucide-react";
import { getHeatmapUrl } from "../lib/api/analysis";
import type { AnalysisResponse } from "../lib/api/types";

function ConfidenceRing({ value }: { value: number }) {
  return (
    <div
      className="confidence-ring"
      style={{ "--ring-progress": `${value * 3.6}deg` } as React.CSSProperties}
    >
      <div>
        <strong>{value}%</strong>
        <span>confidence</span>
      </div>
    </div>
  );
}

function verdictLabel(verdict: AnalysisResponse["verdict"]): string {
  return verdict === "likely_ai_generated" ? "Likely AI-generated" : "Likely authentic";
}

function formatBytes(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

/**
 * Renders a real /api/v1/analyze response. Every value shown here comes
 * directly from the backend - nothing here is fabricated, and fields the
 * backend does not provide (per-signal scores, robustness, a two-sided
 * AI/real probability split) are intentionally absent.
 */
export default function AnalysisResult({
  result,
  fileName,
  onReset,
}: {
  result: AnalysisResponse;
  fileName?: string;
  onReset: () => void;
}) {
  const heatmapUrl = getHeatmapUrl(result);
  const confidencePercent = Math.round(result.confidence * 100);
  const { metadata, explanation } = result;

  return (
    <div className="results-wrap" id="results">
      <div className="result-toolbar">
        <div>
          <span className="eyebrow">ANALYSIS COMPLETE</span>
          <h3>Assessment for {fileName || "uploaded image"}</h3>
        </div>
        <button className="secondary-button" onClick={onReset}>
          <RotateCcw size={15} /> Analyze another
        </button>
      </div>

      <div className="result-grid">
        <div className="evidence-card">
          <div className="card-heading">
            <div>
              <span className="eyebrow">VISUAL EVIDENCE</span>
              <h3>Grad-CAM heatmap</h3>
            </div>
            <span className="mini-tag">XAI</span>
          </div>
          <div className="heatmap-viewer">
            {heatmapUrl ? (
              <img src={heatmapUrl} alt="Grad-CAM explanation heatmap" />
            ) : (
              <div className="heatmap-unavailable">
                <Info size={18} />
                <span>Explanation heatmap unavailable for this analysis.</span>
              </div>
            )}
          </div>
          <p className="disclaimer">
            <Info size={14} /> The heatmap is rendered at the model's input
            resolution and shows where the model's visual (RGB) analysis
            concentrated - it is not pixel-aligned to the original upload and
            is not proof of a specific forensic artifact.
          </p>
        </div>

        <div className="verdict-card">
          <div className="verdict-top">
            <div>
              <span className="eyebrow">LIKELIHOOD ASSESSMENT</span>
              <div className="verdict-label">
                <span className="verdict-dot" /> {verdictLabel(result.verdict)}
              </div>
            </div>
            <ConfidenceRing value={confidencePercent} />
          </div>
          <div className="probabilities">
            <div>
              <span>Model confidence in this verdict</span>
              <strong>{confidencePercent}%</strong>
            </div>
            <div className="prob-bar">
              <i style={{ width: `${confidencePercent}%` }} />
            </div>
          </div>
          <p className="calibration">
            {result.is_calibrated
              ? `Calibrated confidence (${result.calibration_method ?? "temperature-scaled"}). `
              : "Uncalibrated raw model confidence. "}
            Interpret this as a likelihood, not certainty.
          </p>
          <p className="small-note">
            Model {result.model_version} · {result.processing_time_ms}ms ·
            decision threshold {Math.round(result.threshold * 100)}%
          </p>
        </div>
      </div>

      <div className="lower-grid">
        <div className="explanation-card">
          <div className="card-heading">
            <div>
              <span className="eyebrow">EXPLAINABILITY LAYER</span>
              <h3>Why SignalScope reached this assessment</h3>
            </div>
            <BarChart3 size={18} />
          </div>
          <p>
            {explanation.summary ??
              "Grounded visual explanation was unavailable for this analysis."}
          </p>
        </div>

        <div className="metadata-card">
          <div className="card-heading">
            <div>
              <span className="eyebrow">PROVENANCE SNAPSHOT</span>
              <h3>Metadata & provenance</h3>
            </div>
            <FileSearch size={18} />
          </div>
          <div className="metadata-list">
            <div>
              <span>Format</span>
              <strong>{metadata.format}</strong>
            </div>
            <div>
              <span>Dimensions</span>
              <strong>
                {metadata.width}×{metadata.height}px
              </strong>
            </div>
            <div>
              <span>File size</span>
              <strong>{formatBytes(metadata.file_size)}</strong>
            </div>
            <div>
              <span>EXIF data</span>
              <strong>{metadata.has_exif ? "Present" : "Not present"}</strong>
            </div>
            <div>
              <span>C2PA / Content Credentials</span>
              <strong>
                {metadata.c2pa_verified
                  ? "Verified"
                  : metadata.c2pa_detected
                    ? "Marker detected, not verified"
                    : "Not detected"}
              </strong>
            </div>
          </div>
          <p className="small-note">
            <ShieldCheck size={14} /> {metadata.note}
          </p>
        </div>
      </div>
    </div>
  );
}
