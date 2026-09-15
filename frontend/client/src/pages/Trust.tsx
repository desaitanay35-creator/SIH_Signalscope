import {
  AlertTriangle,
  EyeOff,
  HandHeart,
  Scale,
  ShieldCheck,
  UserRoundCheck,
} from "lucide-react";
import PageLayout, { PageSection } from "../components/PageLayout";
export default function Trust() {
  return (
    <PageLayout
      eyebrow="TRUST & ETHICS"
      title="Built for responsible media verification."
      copy="SignalScope is intentionally designed to support judgment—not replace it. The interface makes uncertainty visible and keeps the human in the loop."
    >
      <PageSection title="The trust framework">
        <div className="ethics-grid">
          <div>
            <span className="ethics-icon">
              <Scale size={18} />
            </span>
            <h3>Likelihood, not accusation</h3>
            <p>
              Results use calibrated language such as “Likely AI-generated,”
              “Likely authentic,” or “Uncertain.” They are never presented as
              definitive proof.
            </p>
          </div>
          <div>
            <span className="ethics-icon violet">
              <EyeOff size={18} />
            </span>
            <h3>No identity claims</h3>
            <p>
              SignalScope is not a face-swap detector for identifiable people
              and must not make accusations about real people.
            </p>
          </div>
          <div>
            <span className="ethics-icon green">
              <HandHeart size={18} />
            </span>
            <h3>Human review matters</h3>
            <p>
              Source context, provenance, editorial judgment, and expert review
              remain important parts of responsible verification.
            </p>
          </div>
          <div>
            <span className="ethics-icon amber">
              <AlertTriangle size={18} />
            </span>
            <h3>Limitations are visible</h3>
            <p>
              Missing metadata does not mean an image is synthetic. Model
              confidence can be wrong, biased, or affected by transformations.
            </p>
          </div>
        </div>
      </PageSection>
      <PageSection title="Before you rely on an assessment">
        <div className="checklist">
          <div>
            <ShieldCheck size={17} />
            <strong>Check the source and chain of custody</strong>
            <p>
              Ask where the image came from, who shared it, and whether the
              original file is available.
            </p>
          </div>
          <div>
            <UserRoundCheck size={17} />
            <strong>Seek independent context</strong>
            <p>
              Compare with trusted reporting, reverse-image sources, and domain
              expertise.
            </p>
          </div>
          <div>
            <Scale size={17} />
            <strong>Do not overstate the output</strong>
            <p>
              A likelihood assessment is one supporting signal—not a declaration
              that content is true or false.
            </p>
          </div>
        </div>
      </PageSection>
      <section className="disclaimer-banner container">
        <ShieldCheck size={23} />
        <div>
          <strong>
            SignalScope provides an AI-based likelihood assessment.
          </strong>
          <p>It is not definitive proof that an image is real or synthetic.</p>
        </div>
      </section>
    </PageLayout>
  );
}
