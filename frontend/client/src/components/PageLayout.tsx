import { LogOut, Moon, ScanLine, Sun, User as UserIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { useAuth } from "../contexts/AuthContext";

export default function PageLayout({
  children,
  eyebrow,
  title,
  copy,
}: {
  children: React.ReactNode;
  eyebrow: string;
  title: string;
  copy: string;
}) {
  const { user, logout } = useAuth();
  const [theme, setTheme] = useState<"dark" | "light">(
    () =>
      (localStorage.getItem("signalscope-theme") as "dark" | "light") || "dark"
  );
  const [open, setOpen] = useState(false);
  const currentPath = window.location.pathname;

  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("signalscope-theme", theme);
  }, [theme]);

  const toggle = () => setTheme(theme === "dark" ? "light" : "dark");

  return (
    <div className="app-shell">
      <header className="navbar">
        <div className="nav-inner">
          <a className="brand" href="/">
            <span className="brand-mark">
              <ScanLine size={18} />
            </span>
            <span>
              Signal<span>Scope</span>
            </span>
          </a>
          <nav className={open ? "nav-links open" : "nav-links"}>
            <a
              href="/analyze"
              className={currentPath === "/analyze" ? "active" : ""}
            >
              Analyze
            </a>
            <a
              href="/how-it-works"
              className={currentPath === "/how-it-works" ? "active" : ""}
            >
              How it works
            </a>
            <a
              href="/technology"
              className={currentPath === "/technology" ? "active" : ""}
            >
              Technology
            </a>
            <a
              href="/trust"
              className={currentPath === "/trust" ? "active" : ""}
            >
              Trust & ethics
            </a>
            <a
              href="/history"
              className={currentPath === "/history" ? "active" : ""}
            >
              History
            </a>
          </nav>
          <div className="nav-actions">
            <button
              className="icon-button"
              onClick={toggle}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>

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
              onClick={() => setOpen(!open)}
              aria-label="Toggle navigation"
            >
              ☰
            </button>
          </div>
        </div>
      </header>
      <main>
        <section className="page-hero container">
          <span className="eyebrow">{eyebrow}</span>
          <h1>{title}</h1>
          <p>{copy}</p>
        </section>
        {children}
      </main>
      <footer className="footer">
        <div className="container footer-inner">
          <div>
            <a className="brand" href="/">
              <span className="brand-mark">
                <ScanLine size={18} />
              </span>
              <span>
                Signal<span>Scope</span>
              </span>
            </a>
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
            <a href="/history">History</a>
            <a href="/login">Sign in</a>
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

export function PageSection({
  eyebrow,
  title,
  children,
}: {
  eyebrow?: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="inner-page-section container">
      <div className="section-heading">
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

export function FeatureCard({
  number,
  title,
  copy,
  accent = "cyan",
}: {
  number: string;
  title: string;
  copy: string;
  accent?: string;
}) {
  return (
    <article className="feature-card">
      <span className={`feature-number ${accent}`}>{number}</span>
      <h3>{title}</h3>
      <p>{copy}</p>
    </article>
  );
}
