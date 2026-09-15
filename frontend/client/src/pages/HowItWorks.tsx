import {
  CloudUpload,
  FileSearch,
  ShieldCheck,
  UserRoundCheck,
} from "lucide-react";
import PageLayout, { FeatureCard, PageSection } from "../components/PageLayout";
export default function HowItWorks() {
  return (
    <PageLayout
      eyebrow="THE METHOD"
      title="A signal, not a verdict."
      copy="SignalScope turns complex media forensics into a clear, inspectable workflow. The product helps people ask better questions about visual media."
    >
      <PageSection title="From image to understanding">
        <div className="feature-grid three">
          <FeatureCard
            number="01"
            title="Upload"
            copy="Bring a visual into the workspace. The image is sent directly to the SignalScope model for analysis and is not shared beyond that."
          />
          <FeatureCard
            number="02"
            title="Inspect"
            copy="An RGB branch and a frequency-domain branch each examine the image, then a fusion model combines both signals into one assessment."
            accent="violet"
          />
          <FeatureCard
            number="03"
            title="Understand"
            copy="See a calibrated likelihood alongside supporting signals, provenance context, and model limitations."
            accent="green"
          />
        </div>
      </PageSection>
      <PageSection
        eyebrow="THE REVIEW LOOP"
        title="Designed for human judgment"
      >
        <div className="timeline">
          <div>
            <span>1</span>
            <CloudUpload />
            <div>
              <h3>Bring context in</h3>
              <p>
                Start with the image and the context you already have.
                SignalScope does not attempt to identify people or verify
                political events.
              </p>
            </div>
          </div>
          <div>
            <span>2</span>
            <FileSearch />
            <div>
              <h3>Make evidence legible</h3>
              <p>
                A Grad-CAM heatmap, a grounded explanation, and image
                metadata are shown separately so no single signal is
                mistaken for proof.
              </p>
            </div>
          </div>
          <div>
            <span>3</span>
            <UserRoundCheck />
            <div>
              <h3>Keep a human in the loop</h3>
              <p>
                Use the output as one input into a careful review process.
                Expert judgment and source context remain important.
              </p>
            </div>
          </div>
        </div>
      </PageSection>
      <PageSection title="The principles">
        <div className="principles-grid">
          <div>
            <ShieldCheck size={19} />
            <strong>Likelihood language</strong>
            <p>
              Likely authentic, likely AI-generated, or uncertain—not absolute
              truth.
            </p>
          </div>
          <div>
            <FileSearch size={19} />
            <strong>Explainability first</strong>
            <p>
              Every assessment should be paired with the evidence that
              influenced it.
            </p>
          </div>
          <div>
            <UserRoundCheck size={19} />
            <strong>Responsible scope</strong>
            <p>
              No identity claims, no accusations, and no political-event
              verification.
            </p>
          </div>
        </div>
      </PageSection>
    </PageLayout>
  );
}
