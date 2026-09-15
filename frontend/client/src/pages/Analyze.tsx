import { useRef } from "react";
import {
  ArrowRight,
  Check,
  CircleHelp,
  CloudUpload,
  FileSearch,
  ScanLine,
  ShieldCheck,
  Upload,
  X,
  Zap,
} from "lucide-react";
import { motion } from "framer-motion";
import PageLayout, { PageSection } from "../components/PageLayout";
import AnalysisResult from "../components/AnalysisResult";
import { useAnalysis } from "../hooks/useAnalysis";

export default function Analyze() {
  const inputRef = useRef<HTMLInputElement>(null);
  const {
    file,
    preview,
    status,
    error,
    validationError,
    result,
    progress,
    stage,
    stages,
    chooseFile,
    analyze,
    reset,
  } = useAnalysis();

  return (
    <PageLayout
      eyebrow="ANALYSIS WORKSPACE"
      title="Analyze an image"
      copy="Upload an image and SignalScope's RGB + frequency fusion model will assess whether it is likely AI-generated or likely authentic."
    >
      <PageSection title="Forensic scan workspace">
        {status === "scanning" && (
          <motion.div
            className="scanner-card"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <div className="scanner-image">
              <img src={preview} alt="Image being scanned" />
              <div className="scanner-line" />
              <div className="scanner-overlay">
                <ScanLine size={24} />
                <span>Forensic scan in progress</span>
              </div>
            </div>
            <div className="scanner-meta">
              <div>
                <span className="eyebrow">SIGNAL ENGINE / RUNNING</span>
                <h3>Reading the image's visual fingerprint...</h3>
              </div>
              <div className="progress-number">
                {progress}
                <small>%</small>
              </div>
            </div>
            <div className="stage-list">
              {stages.map((item, i) => (
                <div
                  className={
                    i < stage
                      ? "stage done"
                      : i === stage
                        ? "stage current"
                        : "stage"
                  }
                  key={item}
                >
                  <span>{i < stage ? <Check size={13} /> : `0${i + 1}`}</span>
                  <label>{item}</label>
                </div>
              ))}
            </div>
            <div className="progress-track">
              <span style={{ width: `${progress}%` }} />
            </div>
          </motion.div>
        )}

        {status === "error" && (
          <div className="upload-card">
            <div className="error-message">
              <CircleHelp size={15} />
              {error?.message || "Something went wrong while analyzing this image."}
            </div>
            <button className="secondary-button" onClick={reset}>
              Try again
            </button>
          </div>
        )}

        {status === "result" && result && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <AnalysisResult result={result} fileName={file?.name} onReset={reset} />
          </motion.div>
        )}

        {(status === "empty" || status === "ready") && (
          <div className="upload-card">
            <div
              className="drop-zone"
              onClick={() => inputRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => {
                e.preventDefault();
                chooseFile(e.dataTransfer.files?.[0]);
              }}
              role="button"
              tabIndex={0}
            >
              <input
                ref={inputRef}
                type="file"
                hidden
                accept="image/png,image/jpeg,image/webp"
                onChange={e => chooseFile(e.target.files?.[0])}
              />
              <div className="upload-icon">
                <CloudUpload size={24} />
              </div>
              <h3>Drop your image here</h3>
              <p>
                or <span>browse from your device</span>
              </p>
              <small>
                PNG, JPG, JPEG, WEBP <i /> Max 10 MB
              </small>
            </div>
            {validationError && (
              <p className="error-message">{validationError}</p>
            )}
            {status === "ready" && file && (
              <div className="selected-file">
                <img src={preview} alt="Selected preview" />
                <div className="file-info">
                  <strong>{file.name}</strong>
                  <span>Image ready to scan</span>
                </div>
                <button onClick={reset} aria-label="Remove image">
                  <X size={17} />
                </button>
                <button className="analyze-button" onClick={() => analyze()}>
                  <Zap size={16} /> Analyze image
                </button>
              </div>
            )}
            <div className="upload-foot">
              <span>
                <ShieldCheck size={14} /> Sent directly to the SignalScope model
              </span>
              <span>
                <Upload size={14} /> Responsible AI
              </span>
            </div>
          </div>
        )}
      </PageSection>
      <PageSection title="What you get">
        <div className="feature-grid">
          <div className="feature-card">
            <ShieldCheck size={18} />
            <h3>Calibrated likelihood</h3>
            <p>
              Responsible labels designed to communicate uncertainty without
              accusations.
            </p>
          </div>
          <div className="feature-card">
            <ScanLine size={18} />
            <h3>Visual explanation</h3>
            <p>
              A real Grad-CAM heatmap and grounded summary help reviewers
              understand what influenced the output.
            </p>
          </div>
          <div className="feature-card">
            <ArrowRight size={18} />
            <h3>Provenance context</h3>
            <p>
              EXIF and C2PA metadata add supporting context without
              overstating what it proves.
            </p>
          </div>
          <div className="feature-card">
            <FileSearch size={18} />
            <h3>No login required</h3>
            <p>Run an analysis without creating an account.</p>
          </div>
        </div>
      </PageSection>
    </PageLayout>
  );
}
