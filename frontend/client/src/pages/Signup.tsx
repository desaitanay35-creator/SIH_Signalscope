import { useState } from "react";
import {
  ArrowRight,
  Check,
  CheckCircle2,
  Eye,
  EyeOff,
  Lock,
  Mail,
  ShieldCheck,
  Sparkles,
  User,
  X,
} from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import PageLayout from "../components/PageLayout";

export default function Signup() {
  const { signup, user } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [agreedTerms, setAgreedTerms] = useState(true);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  // Detailed password validation rules
  const passwordRules = {
    minLength: password.length >= 8,
    hasUpper: /[A-Z]/.test(password),
    hasLower: /[a-z]/.test(password),
    hasNumber: /[0-9]/.test(password),
    hasSpecial: /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(password),
  };

  const isPasswordValid =
    passwordRules.minLength &&
    passwordRules.hasUpper &&
    passwordRules.hasLower &&
    passwordRules.hasNumber &&
    passwordRules.hasSpecial;

  const getPasswordStrength = () => {
    const count = Object.values(passwordRules).filter(Boolean).length;
    if (count === 0) return { score: 0, label: "", color: "" };
    if (count <= 2) return { score: 1, label: "Weak", color: "danger" };
    if (count <= 4) return { score: 2, label: "Medium", color: "amber" };
    return { score: 3, label: "Strong", color: "green" };
  };

  const strength = getPasswordStrength();
  const passwordsMatch =
    confirmPassword.length > 0 && password === confirmPassword;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!name.trim()) {
      setError("Please enter your full name.");
      return;
    }
    if (!email || !email.includes("@")) {
      setError("Please enter a valid email address.");
      return;
    }
    if (!isPasswordValid) {
      setError(
        "Password does not meet all security requirements listed below."
      );
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (!agreedTerms) {
      setError("You must agree to the Terms of Service & Privacy Policy.");
      return;
    }

    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 600));
      await signup(name, email, password);
      setSuccess(true);
      setTimeout(() => {
        window.location.href = "/analyze";
      }, 700);
    } catch {
      setError("Failed to create account. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleFillDemo = () => {
    setName("Elena Rostova");
    setEmail("elena.rostova@forensics-lab.org");
    setPassword("SignalScope2026!");
    setConfirmPassword("SignalScope2026!");
    setAgreedTerms(true);
    setError("");
  };

  return (
    <PageLayout
      eyebrow="AUTHENTICATION"
      title="Create your SignalScope account"
      copy="Get started with synthetic media detection, calibrated confidence scoring, and visual explainability tools."
    >
      <div className="auth-container">
        <div className="auth-card">
          {user ? (
            <div className="auth-signed-in">
              <div className="signed-in-avatar">
                <CheckCircle2 size={32} />
              </div>
              <h3>Account Active</h3>
              <p>
                You are currently signed in as <strong>{user.name}</strong> (
                {user.email}).
              </p>
              <div className="signed-in-actions">
                <a className="primary-button" href="/analyze">
                  Go to Workspace
                </a>
              </div>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="auth-form">
              <div className="auth-header">
                <h3>Sign up for an account</h3>
                <button
                  type="button"
                  className="demo-fill-btn"
                  onClick={handleFillDemo}
                >
                  <Sparkles size={13} />
                  <span>Auto-fill Form</span>
                </button>
              </div>
              <p className="small-note">
                Demo account only — stored in this browser, not connected to
                the SignalScope backend. Image analysis does not require an
                account.
              </p>

              {error && (
                <div className="auth-error">
                  <span>{error}</span>
                </div>
              )}

              {success && (
                <div className="auth-success">
                  <ShieldCheck size={16} />
                  <span>Account created successfully! Redirecting...</span>
                </div>
              )}

              <div className="form-group">
                <label htmlFor="name">Full Name</label>
                <div className="input-wrapper">
                  <User size={17} className="input-icon" />
                  <input
                    id="name"
                    type="text"
                    placeholder="e.g. Dr. Alex Morgan"
                    value={name}
                    onChange={e => setName(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="email">Work Email Address</label>
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
                <label htmlFor="password">Password</label>
                <div className="input-wrapper">
                  <Lock size={17} className="input-icon" />
                  <input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Create a strong password"
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

                {password && (
                  <div className="password-strength">
                    <div className="strength-bars">
                      <span
                        className={strength.score >= 1 ? strength.color : ""}
                      />
                      <span
                        className={strength.score >= 2 ? strength.color : ""}
                      />
                      <span
                        className={strength.score >= 3 ? strength.color : ""}
                      />
                    </div>
                    <span className="strength-label">
                      Strength: {strength.label}
                    </span>
                  </div>
                )}

                {/* Password Requirements Checklist */}
                <div className="password-checklist-container">
                  <span className="checklist-title">
                    Password must contain:
                  </span>
                  <ul className="password-checklist">
                    <li className={passwordRules.minLength ? "met" : ""}>
                      {passwordRules.minLength ? (
                        <Check size={12} />
                      ) : (
                        <X size={12} />
                      )}
                      <span>At least 8 characters</span>
                    </li>
                    <li className={passwordRules.hasUpper ? "met" : ""}>
                      {passwordRules.hasUpper ? (
                        <Check size={12} />
                      ) : (
                        <X size={12} />
                      )}
                      <span>One uppercase letter (A-Z)</span>
                    </li>
                    <li className={passwordRules.hasLower ? "met" : ""}>
                      {passwordRules.hasLower ? (
                        <Check size={12} />
                      ) : (
                        <X size={12} />
                      )}
                      <span>One lowercase letter (a-z)</span>
                    </li>
                    <li className={passwordRules.hasNumber ? "met" : ""}>
                      {passwordRules.hasNumber ? (
                        <Check size={12} />
                      ) : (
                        <X size={12} />
                      )}
                      <span>One number (0-9)</span>
                    </li>
                    <li className={passwordRules.hasSpecial ? "met" : ""}>
                      {passwordRules.hasSpecial ? (
                        <Check size={12} />
                      ) : (
                        <X size={12} />
                      )}
                      <span>One special character (!@#$%^&*)</span>
                    </li>
                  </ul>
                </div>
              </div>

              <div className="form-group">
                <div className="label-row">
                  <label htmlFor="confirmPassword">Confirm Password</label>
                  {confirmPassword.length > 0 && (
                    <span
                      className={
                        passwordsMatch
                          ? "match-badge valid"
                          : "match-badge invalid"
                      }
                    >
                      {passwordsMatch ? "Passwords match" : "Does not match"}
                    </span>
                  )}
                </div>
                <div className="input-wrapper">
                  <Lock size={17} className="input-icon" />
                  <input
                    id="confirmPassword"
                    type={showPassword ? "text" : "password"}
                    placeholder="Re-enter your password"
                    value={confirmPassword}
                    onChange={e => setConfirmPassword(e.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="form-options">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={agreedTerms}
                    onChange={e => setAgreedTerms(e.target.checked)}
                    required
                  />
                  <span>
                    I agree to the <a href="/trust">Terms of Service</a> and{" "}
                    <a href="/trust">Privacy Policy</a>
                  </span>
                </label>
              </div>

              <button
                type="submit"
                className="primary-button auth-submit-btn"
                disabled={loading || success}
              >
                {loading ? (
                  <span>Creating account...</span>
                ) : (
                  <>
                    <span>Create Account</span>
                    <ArrowRight size={16} />
                  </>
                )}
              </button>

              <div className="auth-footer">
                <span>Already have an account?</span>
                <a href="/login" className="auth-link">
                  Sign in
                </a>
              </div>
            </form>
          )}
        </div>
      </div>
    </PageLayout>
  );
}
