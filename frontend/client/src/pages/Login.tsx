import { useState } from "react";
import {
  ArrowRight,
  Eye,
  EyeOff,
  Lock,
  Mail,
  ShieldCheck,
  Sparkles,
  UserCheck,
} from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import PageLayout from "../components/PageLayout";

export default function Login() {
  const { login, user, logout } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(true);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!email || !email.includes("@") || !email.includes(".")) {
      setError(
        "Please enter a valid email address (e.g. name@organization.com)."
      );
      return;
    }
    if (!password || password.length < 6) {
      setError("Password must be at least 6 characters long.");
      return;
    }

    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 600));
      await login(email, password);
      setSuccess(true);
      setTimeout(() => {
        window.location.href = "/analyze";
      }, 700);
    } catch {
      setError("Failed to sign in. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  const handleDemoFill = () => {
    setEmail("researcher@signalscope.io");
    setPassword("demoPass2026!");
    setError("");
  };

  return (
    <PageLayout
      eyebrow="AUTHENTICATION"
      title="Welcome back to SignalScope"
      copy="Sign in to access your media verification history, custom signal thresholds, and saved forensic assessments."
    >
      <div className="auth-container">
        <div className="auth-card">
          {user ? (
            <div className="auth-signed-in">
              <div className="signed-in-avatar">
                <UserCheck size={32} />
              </div>
              <h3>Currently signed in</h3>
              <p>
                You are logged in as <strong>{user.email}</strong>.
              </p>
              <div className="signed-in-actions">
                <a className="primary-button" href="/analyze">
                  Go to Workspace
                </a>
                <button className="secondary-button" onClick={logout}>
                  Sign Out
                </button>
              </div>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="auth-form">
              <div className="auth-header">
                <h3>Sign in to your account</h3>
                <button
                  type="button"
                  className="demo-fill-btn"
                  onClick={handleDemoFill}
                >
                  <Sparkles size={13} />
                  <span>Use Demo Account</span>
                </button>
              </div>
              <p className="small-note">
                Demo account only — stored in this browser, not connected to
                the SignalScope backend. Image analysis does not require
                signing in.
              </p>

              {error && (
                <div className="auth-error">
                  <span>{error}</span>
                </div>
              )}

              {success && (
                <div className="auth-success">
                  <ShieldCheck size={16} />
                  <span>Successfully logged in! Redirecting...</span>
                </div>
              )}

              <div className="form-group">
                <label htmlFor="email">Email Address</label>
                <div className="input-wrapper">
                  <Mail size={17} className="input-icon" />
                  <input
                    id="email"
                    type="email"
                    placeholder="name@organization.com"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <div className="label-row">
                  <label htmlFor="password">Password</label>
                  <a
                    href="#forgot"
                    className="forgot-link"
                    onClick={e => {
                      e.preventDefault();
                      alert(
                        "Demo mode: Click 'Use Demo Account' or enter any valid email and 6+ character password."
                      );
                    }}
                  >
                    Forgot password?
                  </a>
                </div>
                <div className="input-wrapper">
                  <Lock size={17} className="input-icon" />
                  <input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                  />
                  <button
                    type="button"
                    className="toggle-password"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={
                      showPassword ? "Hide password" : "Show password"
                    }
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              <div className="form-options">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={remember}
                    onChange={e => setRemember(e.target.checked)}
                  />
                  <span>Remember me on this device</span>
                </label>
              </div>

              <button
                type="submit"
                className="primary-button auth-submit-btn"
                disabled={loading || success}
              >
                {loading ? (
                  <span>Signing in...</span>
                ) : (
                  <>
                    <span>Sign In</span>
                    <ArrowRight size={16} />
                  </>
                )}
              </button>

              <div className="auth-footer">
                <span>Don't have an account?</span>
                <a href="/signup" className="auth-link">
                  Create an account
                </a>
              </div>
            </form>
          )}
        </div>
      </div>
    </PageLayout>
  );
}
