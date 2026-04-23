import Image from "next/image";
import { Timeline } from "@/components/Timeline";

export const metadata = {
  title: "About",
  description:
    "ML Engineer II at Amazon (Prime Video) and Lead Software Engineer at MIT PSFC (DisruptionPy). Prior: Cerebro Sports, Microsoft M365 Security, AI.Reverie, Red Hat OpenShift. BU: MS AI + Deep Learning, BS CS + Astrophysics.",
  openGraph: {
    title: "About -- Sameer Chaturvedi",
    description:
      "Engineer at Amazon and MIT. Basketball lifer building the tools I wish we had.",
  },
};

export default function AboutPage() {
  return (
    <>
      {/* Hero with headshot */}
      <section className="relative overflow-hidden border-b hairline">
        <div className="hero-aura" aria-hidden />
        <div className="relative mx-auto max-w-6xl px-6 pt-20 pb-16 sm:pt-24">
          <div className="grid lg:grid-cols-12 gap-10 items-center">
            <div className="lg:col-span-7 order-2 lg:order-1">
              <p className="overline">About</p>
              <h1 className="mt-5 h-display">Sameer Chaturvedi</h1>
              <p className="mt-6 text-lg sm:text-xl text-muted-strong leading-relaxed max-w-2xl">
                Full-stack software engineer. Basketball lifer. I ship
                ML, mobile, and web systems end-to-end -- turning game
                film into numbers coaches, scouts, and front offices can
                actually use.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <a
                  href="mailto:sam.chaturvedi24@gmail.com"
                  className="btn btn-primary"
                >
                  sam.chaturvedi24@gmail.com
                </a>
                <a
                  href="/Sameer_Chaturvedi_Resume.pdf"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-secondary"
                >
                  Resume (PDF)
                </a>
                <a
                  href="https://linkedin.com/in/sameerc"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-secondary"
                >
                  LinkedIn
                </a>
                <a
                  href="https://github.com/samc24"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-secondary"
                >
                  GitHub / samc24
                </a>
              </div>
            </div>

            <div className="lg:col-span-5 order-1 lg:order-2">
              <div className="relative">
                <div
                  className="absolute -inset-4 rounded-[2rem] bg-gradient-to-br from-accent/15 via-transparent to-ft-blue/15 blur-2xl pointer-events-none"
                  aria-hidden
                />
                <div className="relative rounded-2xl overflow-hidden border hairline-strong bg-black">
                  <Image
                    src="/photos/sam_mit.jpg"
                    alt="Sameer Chaturvedi"
                    width={1400}
                    height={1000}
                    className="w-full h-auto"
                    priority
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Why I build */}
      <section className="mx-auto max-w-4xl px-6 py-20">
        <p className="overline">Purpose</p>
        <h2 className="mt-4 text-2xl sm:text-3xl font-semibold tracking-tight max-w-3xl">
          Using engineering to solve problems in the two things I love most:
          basketball and physics.
        </h2>

        <div className="mt-10 grid md:grid-cols-2 gap-5">
          <div className="card-raised rounded-2xl p-8">
            <p className="overline text-accent">Basketball</p>
            <p className="mt-4 text-lg text-foreground leading-snug font-medium">
              Give back to the game that has given so much.
            </p>
            <p className="mt-4 text-muted-strong leading-relaxed">
              Basketball has been the main constant in my life -- the
              driving motivator behind my mentality, perspectives, and
              success. I build AI for basketball -- from the models
              themselves to the iOS and web apps that put them in a
              coach&apos;s hands -- to bridge the eye-test and counting
              stats for coaching, scouting, and training. I play every
              day. The love of the art drives the work.
            </p>
          </div>

          <div
            className="card-raised rounded-2xl p-8"
            style={{ borderColor: "rgba(59,130,246,0.28)" }}
          >
            <p
              className="overline"
              style={{ color: "var(--color-ft-blue)" }}
            >
              Physics
            </p>
            <p className="mt-4 text-lg text-foreground leading-snug font-medium">
              Engineering with purpose -- build the tools that extend what
              we can perceive and what we can power.
            </p>
            <p className="mt-4 text-muted-strong leading-relaxed">
              I studied astrophysics as an undergrad, and today I write
              software for plasma-fusion research at MIT: two ends of the
              same obsession with how the universe actually works and how
              we harness it. Observational instruments stretched our senses
              for centuries. ML and software are the next stretch --
              letting us see further, model deeper, and extract energy
              from the phenomena that built us.
            </p>
          </div>
        </div>
      </section>

      {/* Timeline */}
      <section className="mx-auto max-w-4xl px-6 py-12">
        <p className="overline">Experience</p>
        <h2 className="mt-4 text-3xl sm:text-4xl font-semibold tracking-tight">
          Where I&apos;ve shipped.
        </h2>
        <p className="mt-4 text-muted-strong max-w-2xl">
          Recommendation systems at Amazon. Scientific software at MIT.
          Live-basketball analytics at Cerebro Sports. Petabyte-scale
          security ML at Microsoft. On-device computer vision at AI.Reverie.
          Upstream Kubernetes runtimes at Red Hat. The range is the point.
        </p>

        <div className="mt-14">
          <Timeline />
        </div>
      </section>

      {/* Education */}
      <section className="mx-auto max-w-4xl px-6 py-16 border-t hairline">
        <p className="overline">Education</p>
        <h2 className="mt-4 text-3xl font-semibold tracking-tight">
          Boston University
        </h2>
        <dl className="mt-8 grid sm:grid-cols-3 gap-8">
          <div>
            <dt className="overline">Master&apos;s</dt>
            <dd className="mt-2 text-foreground">AI + Deep Learning</dd>
          </div>
          <div>
            <dt className="overline">Bachelor&apos;s</dt>
            <dd className="mt-2 text-foreground">
              Computer Science + Astrophysics
            </dd>
          </div>
          <div>
            <dt className="overline">Honours</dt>
            <dd className="mt-2 text-foreground">
              Presidential Scholar -- GPA 3.75 / 4
            </dd>
          </div>
        </dl>
      </section>

      {/* Skills, resume-grounded */}
      <section className="mx-auto max-w-4xl px-6 py-16 border-t hairline">
        <p className="overline">Skills</p>
        <h2 className="mt-4 text-3xl font-semibold tracking-tight">
          What I&apos;ve shipped with.
        </h2>
        <p className="mt-4 text-muted-strong max-w-2xl">
          Everything below is from production work or research, tagged to
          where it was used.
        </p>
        <div className="mt-10 grid sm:grid-cols-2 gap-x-10 gap-y-8">
          <StackBucket
            label="Recommendation + ranking"
            items={[
              "Mixture-of-Retrieval ML models (Amazon)",
              "Logistic regression, stratified sampling, softmax tuning (Amazon)",
              "Cross-service content-filtering pipelines (Amazon)",
            ]}
          />
          <StackBucket
            label="Security + anomaly ML"
            items={[
              "Mail-anomaly detection for M365 (Microsoft)",
              "GPT-based security-detection assistant (Microsoft)",
              "Random-forest entity detections (Microsoft)",
              "Petabyte-scale event processing (Microsoft)",
            ]}
          />
          <StackBucket
            label="Distributed systems"
            items={[
              "Spark, PySpark, Azure Functions (Microsoft)",
              "CI/CD, real-time job monitoring (Microsoft)",
              "Auto-failover geo-replicated BCDR (Microsoft)",
              "CRI-O, Libpod, Kubernetes container runtimes (Red Hat)",
            ]}
          />
          <StackBucket
            label="Scientific software"
            items={[
              "DisruptionPy: anomaly models on Tokamak fusion data (MIT)",
              "Pluggable backends via abstract interfaces (MIT)",
              "MDSplus, Xarray/Zarr for multi-machine scalability (MIT)",
            ]}
          />
          <StackBucket
            label="Live sports + LLM apps"
            items={[
              "Semantic-text to SQL pipelines (Cerebro Sports)",
              "Basketball stats DB + edge-case handling (Cerebro Sports)",
              "LLM chat + realtime graphs + dashboards (Cerebro Sports)",
            ]}
          />
          <StackBucket
            label="Classical + deep computer vision"
            items={[
              "Kalman filters, Savitzky-Golay smoothing, Circle Hough Transform",
              "Motion energy, contour detection, frame differencing, template matching",
              "Single-shot object detection models",
              "Pose estimation for shot-mechanics analysis",
            ]}
          />
          <StackBucket
            label="Deep learning"
            items={[
              "Dropout, batch normalisation, hyperparameter tuning (AI.Reverie)",
              "Training on custom synthetic data (AI.Reverie)",
              "Custom PyTorch DataLoaders and dataset mappers (AI.Reverie)",
              "TorchScript model conversion for mobile (AI.Reverie)",
            ]}
          />
          <StackBucket
            label="On-device mobile ML"
            items={[
              "iOS object detection + instance segmentation (AI.Reverie)",
              "Swift + AVFoundation capture pipeline",
              "LibTorch via Objective-C++ bridge",
              "Device motion logging for trajectory analysis",
            ]}
          />
          <StackBucket
            label="Full-stack product"
            items={[
              "End-to-end app: UI, API, data pipeline, LLM (Cerebro Sports)",
              "Realtime graphs + dashboards over a stats DB (Cerebro Sports)",
              "Agile PySpark workspace with CI/CD + job monitoring (Microsoft)",
              "Next.js, React, TypeScript, Tailwind (this site)",
            ]}
          />
          <StackBucket
            label="Small-team engineering"
            items={[
              "Leading features from spec to ship -- at Amazon, MIT, Microsoft",
              "Working with scientists, analysts, and non-technical stakeholders",
              "Production release hygiene: CI/CD, BCDR, observability",
              "Framework choice as a design decision, not a reflex",
            ]}
          />
        </div>
      </section>
    </>
  );
}

function StackBucket({ label, items }: { label: string; items: string[] }) {
  return (
    <div>
      <p className="overline">{label}</p>
      <ul className="mt-4 space-y-2">
        {items.map((it) => (
          <li
            key={it}
            className="text-muted-strong pl-4 relative before:content-['-'] before:absolute before:left-0 before:text-muted"
          >
            {it}
          </li>
        ))}
      </ul>
    </div>
  );
}
