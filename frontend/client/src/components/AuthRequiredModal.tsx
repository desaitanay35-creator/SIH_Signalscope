import { motion, AnimatePresence } from "framer-motion";
import { LockKeyhole, LogIn, UserPlus, X, ShieldAlert } from "lucide-react";

interface AuthRequiredModalProps {
  isOpen: boolean;
  onClose: () => void;
  message?: string;
}

export default function AuthRequiredModal({
  isOpen,
  onClose,
  message = "You must be logged in to upload photos and run forensic analysis.",
}: AuthRequiredModalProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="auth-modal-overlay" onClick={onClose}>
          <motion.div
            className="auth-modal-card"
            initial={{ opacity: 0, scale: 0.94, y: 15 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.94, y: 15 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="auth-modal-close"
              onClick={onClose}
              aria-label="Close dialog"
            >
              <X size={18} />
            </button>

            <div className="auth-modal-icon-wrap">
              <div className="auth-modal-icon-glow" />
              <LockKeyhole size={28} className="auth-modal-icon" />
            </div>

            <div className="auth-modal-content">
              <span className="eyebrow">ACCESS RESTRICTED</span>
              <h3>Authentication Required</h3>
              <p>{message}</p>
            </div>

            <div className="auth-modal-benefits">
              <div className="benefit-item">
                <ShieldAlert size={14} />
                <span>Upload & analyze high-res photos</span>
              </div>
              <div className="benefit-item">
                <ShieldAlert size={14} />
                <span>Save assessment history & provenance logs</span>
              </div>
            </div>

            <div className="auth-modal-actions">
              <a href="/login" className="primary-button auth-modal-btn">
                <LogIn size={16} />
                <span>Log In</span>
              </a>
              <a href="/signup" className="secondary-button auth-modal-btn">
                <UserPlus size={16} />
                <span>Sign Up</span>
              </a>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
