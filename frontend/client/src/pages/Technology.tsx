import {
  Activity,
  FileImage,
  Network,
  ShieldCheck,
  Zap,
} from "lucide-react";
import PageLayout, { FeatureCard, PageSection } from "../components/PageLayout";
export default function Technology() {
  return (
    <PageLayout
      eyebrow="TECHNOLOGY"
      title="Designed to show its work."
      copy="SignalScope combines computer vision, explainable AI, robustness testing, and provenance context into one responsible product surface."
    >
      <PageSection title="The signal stack">
        <div className="feature-grid three">
          <FeatureCard
            number="01"
            title="Computer vision"
            copy="An EfficientNet-B4 RGB branch fused with an FFT-based frequency branch, trained to classify real vs. AI-generated images."
          />
          <FeatureCard
            number="02"
            title="Explainable AI"
            copy="Grad-CAM localization on the RGB branch's final feature block, rendered as a real heatmap image for every analysis."
            accent="violet"
          />
          <FeatureCard
            number="03"
            title="Calibration"
            copy="Temperature-scaled confidence when calibration is configured, so reported confidence reflects the verdict, not certainty."
            accent="amber"
          />
        </div>
      </PageSection>
      <PageSection title="Architecture overview">
        <div className="tech-architecture">
          <div className="arch-node primary">
            <FileImage size={18} /> Image input
          </div>
          <div className="arch-connector" />
          <div className="arch-row">
            <div className="arch-node">
              <Activity size={16} /> RGB + frequency fusion
            </div>
            <div className="arch-node">
              <ShieldCheck size={16} /> Provenance
            </div>
            <div className="arch-node">
              <Zap size={16} /> Calibration
            </div>
          </div>
          <div className="arch-connector short" />
          <div className="arch-node output">
            <Network size={17} /> Calibrated assessment
          </div>
          <span className="arch-note">model output + human context</span>
        </div>
      </PageSection>
      <PageSection title="Integration status">
        <div className="integration-table">
          <div>
            <strong>POST /api/v1/analyze</strong>
            <span>multipart/form-data · image</span>
            <em className="table-status">live</em>
          </div>
          <div>
            <strong>Grad-CAM heatmap</strong>
            <span>explanation.heatmap_url</span>
            <em className="table-status">live</em>
          </div>
          <div>
            <strong>Metadata / provenance</strong>
            <span>EXIF · C2PA marker detection · dimensions</span>
            <em className="table-status">live</em>
          </div>
          <div>
            <strong>Robustness testing</strong>
            <span>compression · resize · screenshot stability</span>
            <em className="table-status pending">not yet implemented</em>
          </div>
        </div>
      </PageSection>
    </PageLayout>
  );
}
