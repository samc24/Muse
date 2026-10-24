import Image from "next/image";
import { Timeline } from "@/components/Timeline";

export const metadata = {
  title: "About",
  description:
    "ML Engineer II at Amazon Prime Video and Lead Software Engineer at MIT PSFC (DisruptionPy). Prior: Cerebro Sports, Microsoft M365 Security, AI.Reverie, Red Hat OpenShift. BU: MS AI + Deep Learning, BS CS + Astrophysics.",
  openGraph: {
    title: "About -- Sameer Chaturvedi",
    description:
      "ML + software engineer at Amazon and MIT. Building AI for basketball to bridge the gap between the eye-test and counting stats.",
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
                I&apos;m an ML + software engineer. Basketball has been the
                main constant in my life, and the driving motivator behind
                my mentality, perspectives, and success. I build AI for the
                game, and I bring the same engineering to recommendation
                systems at Amazon and fusion research at MIT.
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
              Give back to the game that has given so much to us all.
            </p>
            <p className="mt-4 text-muted-strong leading-relaxed">
              The game found me after I moved from California to India
              before high school. No coach, rare courts, broken wooden
              backboards, and sometimes all I had was my ball. I taught
              myself, became an early member of Jr. NBA and NBA India, and
              got deep into the community working to grow the game there.
              The sheer lack of resources is what pushed me to build.
            </p>
            <p className="mt-4 text-muted-strong leading-relaxed">
              Follow Through started with me recording my own jumper,
              watching pros on TV, and tweaking micromovements. It became an
              app that gives anyone with a phone the coaching feedback I
              never had. I play every day, and I&apos;m not chasing a
              shortcut into the league. I&apos;m chasing mastery: seeing the
              game clearly, articulating insights simply, and helping
              players reach their ceiling.
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
              Engineering with purpose.
            </p>
            <p className="mt-4 text-muted-strong leading-relaxed">
              I studied astrophysics alongside computer science, and in
              college I faced a choice: pursue a physics Ph.D. in a niche
              academic subarea, or develop my engineering intuition to build
              tools with immediate, real-world impact. I chose engineering.
              We build instruments to stretch our perception, to glimpse
              truths inaccessible to our raw senses, and that recursive
              process, discovery leading to tools that unlock further
              discovery, is what I want to be part of.
            </p>
            <p className="mt-4 text-muted-strong leading-relaxed">
              Today that means software for plasma fusion research at MIT:
              gold-standard engineering applied to the framework code that
              is rarely in a scientist&apos;s time budget, so physicists can
              spend their limited time on the physics. I believe evolution
              is no longer strictly biological. It&apos;s technological.
              There are always more problems to solve, more tools to build,
              more boundaries to shatter.
            </p>
          </div>
        </div>
      </section>

      {/* Timeline */}
      <section className="mx-auto max-w-4xl px-6 py-12">
        <p className="overline">Experience</p>
        <h2 className="mt-4 text-3xl sm:text-4xl font-semibold tracking-tight">
          Where I&apos;ve worked.
        </h2>
        <p className="mt-4 text-muted-strong max-w-2xl">
          Currently at Amazon Prime Video and MIT&apos;s Plasma Science and
          Fusion Center. Before that: four years at Microsoft on M365
          security, a short contract at Cerebro Sports, on-device computer vision
          at AI.Reverie, and Kubernetes container runtimes at Red Hat.
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

      {/* Outside the day job, resume-grounded */}
      <section className="mx-auto max-w-4xl px-6 py-16 border-t hairline">
        <p className="overline">Also</p>
        <h2 className="mt-4 text-3xl font-semibold tracking-tight">
          Outside the day job.
        </h2>
        <ul className="mt-8 space-y-3">
          {[
            "Technical mentor at the Boston University Innovation Program, 2023 to 2025.",
            "Official speaker at DevConf.us 2019 in Boston: “Sports Science: ML for Skill Analysis”.",
            "hackNY technical fellow, 2020. BU Spark! technical fellow, 2018 to 2019.",
            "BostonHacks 2018 winner with BostonCracks: a sidewalk damage score from semantic segmentation of Street View frames, for equitable repair.",
            "Vice President of OpenWeb, BU’s coding and app development student org, 2020 to 2021.",
            "Hosted an NBA podcast.",
          ].map((it) => (
            <li
              key={it}
              className="text-muted-strong leading-relaxed pl-4 relative before:content-['-'] before:absolute before:left-0 before:text-muted"
            >
              {it}
            </li>
          ))}
        </ul>
      </section>

      {/* Skills, resume-grounded */}
      <section className="mx-auto max-w-4xl px-6 py-16 border-t hairline">
        <p className="overline">Skills</p>
        <h2 className="mt-4 text-3xl font-semibold tracking-tight">
          What I&apos;ve worked with.
        </h2>
        <p className="mt-4 text-muted-strong max-w-2xl">
          Everything below is from production work or research, tagged to
          where it was used.
        </p>
        <div className="mt-10 grid sm:grid-cols-2 gap-x-10 gap-y-8">
          <StackBucket
            label="Recommendation + ranking"
            items={[
              "Contextual multi-armed bandits in production across 6 regions (Amazon)",
              "Sequence-aware collaborative filtering + ANN retrieval (Amazon)",
              "Logistic regression, stratified sampling, softmax tuning (Amazon)",
              "Staged A/B dial-ups with latency + availability guardrails (Amazon)",
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
              "Airflow on EMR on EKS, AWS CDK, multi-region pipelines (Amazon)",
              "Tier-1 oncall, Sev2 root-causing, infra cost planning (Amazon)",
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
            label="LLM agents + apps"
            items={[
              "Agentic oncall platform: Rust tools + agent skills for Claude Code and Kiro (Amazon)",
              "AWS Bedrock AgentCore with per-user OAuth delegation, MCP servers (Amazon)",
              "GPT-based security-detection assistant (Microsoft)",
              "Semantic-text to SQL over a live stats DB (Cerebro Sports)",
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
              "Agile PySpark workspace with CI/CD + job monitoring (Microsoft)",
              "LLM chat demo with realtime graphs + dashboards (Cerebro Sports)",
              "Next.js, React, TypeScript, Tailwind (this site)",
            ]}
          />
          <StackBucket
            label="Small-team engineering"
            items={[
              "Leading features end to end at Amazon, MIT, and Microsoft",
              "Working with physicists, coaches, analysts, and non-technical stakeholders",
              "Production release hygiene: CI/CD, BCDR, observability",
              "Mentoring student founders at the BU Innovation Program",
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
