import { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  Check,
  CircleHelp,
  ChevronRight,
  CloudUpload,
  Code2,
  Cpu,
  FileImage,
  FileSearch,
  LogOut,
  Menu,
  Moon,
  ScanLine,
  ShieldCheck,
  Sun,
  Upload,
  User as UserIcon,
  X,
  Zap,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "../contexts/AuthContext";
import { useAnalysis } from "../hooks/useAnalysis";
import AnalysisResult from "../components/AnalysisResult";

type Theme = "dark" | "light";

const stages = [
  "Uploading image",
  "Preprocessing (RGB + frequency)",
  "Running fusion model",
  "Applying calibration",
  "Generating explanation",
];

function SectionLabel({
  eyebrow,
  title,
  copy,
}: {
  eyebrow: string;
  title: string;
  copy: string;
}) {
  return (
    <div className="section-heading">
      <span className="eyebrow">{eyebrow}</span>
      <h2>{title}</h2>
      <p>{copy}</p>
    </div>
  );
}

function Brand() {
  return (
    <a className="brand" href="#top" aria-label="SignalScope home">
      <span className="brand-mark">
        <ScanLine size={18} />
      </span>
      <span>
        Signal<span>Scope</span>
      </span>
    </a>
  );
}

function ThemeToggle({
  theme,
  setTheme,
}: {
  theme: Theme;
  setTheme: (t: Theme) => void;
}) {
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      className="icon-button"
      onClick={() => setTheme(next)}
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
    >
      {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
    </button>
  );
}

function ForensicVisual() {
  return (
    <div
      className="forensic-visual"
      aria-label="Illustration of AI signal analysis"
    >
      <div className="visual-top">
        <span>
          <span className="status-dot" /> live signal map
        </span>
      </div>
      <div className="visual-canvas">
        <div className="grid-lines" />
        <div className="scan-beam" />
        <div className="subject-shape">
          <div className="subject-ring" />
          <div className="subject-core" />
        </div>
        <span className="marker marker-a">
          A7 <i />
        </span>
        <span className="marker marker-b">
          C2 <i />
        </span>
        <span className="marker marker-c">
          D4 <i />
        </span>
        <div className="waveform">
          <i />
          <i />
          <i />
          <i />
          <i />
          <i />
          <i />
          <i />
          <i />
          <i />
          <i />
          <i />
        </div>
      </div>
      <div className="visual-bottom">
        <div>
          <span className="micro-label">RGB + FREQUENCY FUSION</span>
          <strong>EfficientNet-B4</strong>
        </div>
        <div className="signal-pill">
          <span className="status-dot" /> model ready
        </div>
      </div>
    </div>
  );
}

function formatBytes(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

export default function Home() {
  const { user, logout } = useAuth();
  const [theme, setThemeState] = useState<Theme>(
    () => (localStorage.getItem("signalscope-theme") as Theme) || "dark"
  );
  const [dragging, setDragging] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);

  const {
    file,
    preview,
    status,
    error,
    validationError,
    result,
    progress,
    stage,
    chooseFile,
    analyze,
    reset,
  } = useAnalysis();

  const setTheme = (value: Theme) => {
    setThemeState(value);
    localStorage.setItem("signalscope-theme", value);
  };
  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  return (
    <div id="top" className="app-shell">
      <header className="navbar">
        <div className="nav-inner">
          <Brand />
          <nav className={mobileMenu ? "nav-links open" : "nav-links"}>
            <a href="/analyze" onClick={() => setMobileMenu(false)}>
              Analyze
            </a>
            <a href="/how-it-works" onClick={() => setMobileMenu(false)}>
              How it works
            </a>
            <a href="/technology" onClick={() => setMobileMenu(false)}>
              Technology
            </a>
            <a href="/trust" onClick={() => setMobileMenu(false)}>
              Trust & ethics
            </a>
            <a href="/history" onClick={() => setMobileMenu(false)}>
              History
            </a>
          </nav>
          <div className="nav-actions">
            <ThemeToggle theme={theme} setTheme={setTheme} />
            {user ? (
              <div className="user-nav-badge">
                <span className="user-avatar-pill">
                  <UserIcon size={14} />
                  <span>{user.name}</span>
                </span>
                <button
                  className="icon-button logout-btn"
                  onClick={logout}
                  title="Sign Out"
                  aria-label="Sign Out"
                >
                  <LogOut size={16} />
                </button>
              </div>
            ) : (
              <div className="nav-auth-buttons">
                <a className="nav-login-btn" href="/login">
                  Log in
                </a>
                <a className="nav-signup-btn" href="/signup">
                  Sign up
                </a>
              </div>
            )}
            <button
              className="mobile-menu"
              onClick={() => setMobileMenu(!mobileMenu)}
              aria-label="Toggle navigation"
            >
              <Menu size={20} />
            </button>
          </div>
        </div>
      </header>

      <main>
        <section className="hero container">
          <div className="hero-copy">
            <div className="kicker">
              <span className="status-dot" /> MEDIA FORENSICS
            </div>
            <h1>
              Telling real from <em>synthetic.</em>
            </h1>
            <p className="hero-lede">
              Analyze visual signals, detect synthetic imagery, and understand{" "}
              <strong>why</strong> an image looks AI-generated.
            </p>
            <div className="hero-actions">
              <a className="primary-button" href="/analyze">
                <Upload size={18} /> Analyze an image <ArrowRight size={16} />
              </a>
              <a className="text-button" href="/how-it-works">
                How it works <ChevronRight size={16} />
              </a>
            </div>
            <div className="hero-proof">
              <div>
                <span className="proof-number">01</span>
                <span>
                  Responsible
                  <br />
                  likelihoods
                </span>
              </div>
              <div>
                <span className="proof-number">02</span>
                <span>
                  Explainable
                  <br />
                  signals
                </span>
              </div>
              <div>
                <span className="proof-number">03</span>
                <span>
                  Human-led
                  <br />
                  review
                </span>
              </div>
            </div>
          </div>
          <ForensicVisual />
        </section>

        <section id="analyze" className="analyze-section container">
          <div className="section-intro">
            <div>
              <span className="eyebrow">ANALYSIS WORKSPACE</span>
              <h2>Analyze an image</h2>
              <p>
                Upload an image and SignalScope's fusion model will assess it
                for signs of AI generation.
              </p>
            </div>
          </div>
          <AnimatePresence mode="wait">
            {status === "scanning" && (
              <div className="scanner-card">
                <div className="scanner-image">
                  <img src={preview} alt="Image being analyzed" />
                  <div className="scanner-line" />
                  <div className="scanner-overlay">
                    <ScanLine size={23} />
                    <span>Forensic scan in progress</span>
                  </div>
                </div>
                <div className="scanner-meta">
                  <div className="scanner-title">
                    <span className="eyebrow">SIGNAL ENGINE / RUNNING</span>
                    <h3>
                      Reading the image's visual fingerprint
                      <span className="typing">...</span>
                    </h3>
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
                      {i === stage && <i className="stage-pulse" />}
                    </div>
                  ))}
                </div>
                <div className="progress-track">
                  <span style={{ width: `${progress}%` }} />
                </div>
              </div>
            )}

            {status === "error" && (
              <div className="upload-card">
                <div className="error-message">
                  <CircleHelp size={15} />{" "}
                  {error?.message ||
                    "Something went wrong while analyzing this image."}
                </div>
                <button className="secondary-button" onClick={reset}>
                  Try again
                </button>
              </div>
            )}

            {status === "result" && result && (
              <AnalysisResult result={result} fileName={file?.name} onReset={reset} />
            )}

            {(status === "empty" || status === "ready") && (
              <div className="upload-card">
                <div
                  className={dragging ? "drop-zone dragging" : "drop-zone"}
                  onClick={() =>
                    document.getElementById("home-file-input")?.click()
                  }
                  onDragOver={e => {
                    e.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={e => {
                    e.preventDefault();
                    setDragging(false);
                    chooseFile(e.dataTransfer.files[0]);
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <input
                    id="home-file-input"
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
                  <div className="error-message">
                    <CircleHelp size={15} /> {validationError}
                  </div>
                )}
                {status === "ready" && file && (
                  <div className="selected-file">
                    <img src={preview} alt="Selected preview" />
                    <div className="file-info">
                      <strong>{file.name}</strong>
                      <span>{formatBytes(file.size)} · Image ready to scan</span>
                    </div>
                    <button
                      onClick={e => {
                        e.stopPropagation();
                        reset();
                      }}
                      aria-label="Remove image"
                    >
                      <X size={17} />
                    </button>
                    <button
                      className="analyze-button"
                      onClick={e => {
                        e.stopPropagation();
                        analyze();
                      }}
                    >
                      <Zap size={16} /> Analyze image
                    </button>
                  </div>
                )}
                <div className="upload-foot">
                  <span>
                    <ShieldCheck size={14} /> Sent directly to the SignalScope model
                  </span>
                  <span>Responsible AI · No identity claims</span>
                </div>
              </div>
            )}
          </AnimatePresence>
        </section>

        <section id="how" className="how-section container">
          <SectionLabel
            eyebrow="02 / THE METHOD"
            title="A signal, not a verdict."
            copy="SignalScope makes complex media forensics legible. It surfaces the visual evidence so people can make better-informed decisions."
          />
          <div className="method-grid">
            {[
              [
                "01",
                "Upload",
                "Bring a visual into the workspace. Your file is sent directly to the SignalScope model for analysis.",
                <CloudUpload />,
              ],
              [
                "02",
                "Inspect",
                "An EfficientNet-B4 RGB branch and an FFT-based frequency branch each examine the image, then fuse their signals.",
                <Cpu />,
              ],
              [
                "03",
                "Understand",
                "See a calibrated likelihood with context, limitations, and a grounded Grad-CAM explanation.",
                <FileSearch />,
              ],
            ].map(([num, title, copy, icon]) => (
              <div className="method-card" key={String(num)}>
                <span className="method-num">{num}</span>
                <div className="method-icon">{icon}</div>
                <h3>{String(title)}</h3>
                <p>{String(copy)}</p>
                <ArrowRight size={16} />
              </div>
            ))}
          </div>
        </section>

        <section id="technology" className="technology-section">
          <div className="container">
            <SectionLabel
              eyebrow="03 / TECHNOLOGY"
              title="Designed to show its work."
              copy="SignalScope brings several media-forensics ideas together in one clear, responsible workflow."
            />
            <div className="tech-grid">
              <div className="tech-feature">
                <span className="eyebrow">CORE LAYER</span>
                <h3>Computer vision for the synthetic era.</h3>
                <p>
                  Real-vs-synthetic classification paired with saliency-based
                  localization and provenance signals.
                </p>
                <div className="tech-lines">
                  <div>
                    <Code2 size={17} />
                    <span>
                      Explainable AI
                      <small>Grad-CAM localization</small>
                    </span>
                  </div>
                  <div>
                    <Zap size={17} />
                    <span>
                      Calibration<small>Temperature-scaled confidence</small>
                    </span>
                  </div>
                  <div>
                    <FileImage size={17} />
                    <span>
                      Provenance<small>EXIF / C2PA when available</small>
                    </span>
                  </div>
                </div>
              </div>
              <div className="architecture-card">
                <div className="arch-node primary">
                  <ScanLine size={18} /> Image input
                </div>
                <div className="arch-connector" />
                <div className="arch-row">
                  <div className="arch-node">
                    <Activity size={16} /> RGB + frequency fusion
                  </div>
                  <div className="arch-node">
                    <ShieldCheck size={16} /> Provenance
                  </div>
                </div>
                <div className="arch-connector short" />
                <div className="arch-node output">
                  <FileSearch size={17} /> Calibrated assessment
                </div>
                <span className="arch-note">model output + human context</span>
              </div>
            </div>
          </div>
        </section>

        <section id="trust" className="trust-section container">
          <div className="trust-copy">
            <span className="eyebrow">TRUST & ETHICS</span>
            <h2>Built for responsible media verification.</h2>
            <p>
              SignalScope is intentionally designed to support judgment—not
              replace it. The interface makes uncertainty visible and keeps the
              human in the loop.
            </p>
            <a href="/analyze" className="text-button">
              Try the workspace <ArrowRight size={16} />
            </a>
          </div>
          <div className="trust-list">
            {[
              [
                "Likelihood, not accusation",
                "Outputs use calibrated language rather than absolute claims.",
              ],
              [
                "No identity claims",
                "SignalScope is not a face-swap detector and does not identify people.",
              ],
              [
                "Explainability over black boxes",
                "Every assessment is paired with the signals that influenced it.",
              ],
              [
                "Human judgment remains important",
                "Context, provenance, and expert review still matter.",
              ],
            ].map(([title, copy]) => (
              <div className="trust-item" key={title}>
                <span>
                  <Check size={15} />
                </span>
                <div>
                  <strong>{title}</strong>
                  <p>{copy}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
      <footer className="footer">
        <div className="container footer-inner">
          <div>
            <Brand />
            <p>
              Telling Real From Synthetic
              <br />
              in the Age of Generative Media.
            </p>
          </div>
          <div className="footer-links">
            <a href="/analyze">Analyze</a>
            <a href="/how-it-works">How it works</a>
            <a href="/technology">Technology</a>
            <a href="/trust">Trust & ethics</a>
          </div>
          <div className="footer-meta">
            <span>Built for SIH 2026</span>
            <span>v0.1</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
