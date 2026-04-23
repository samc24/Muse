import Image from "next/image";
import { VideoClip } from "@/components/VideoClip";

export const metadata = {
  title: "Follow Through",
  description:
    "Grade a shooter's form against NBA pros. MediaPipe pose extraction, per-joint deviation, a 0-99 similarity score, and the three mechanics to fix first.",
  openGraph: {
    title: "Follow Through -- shot-form analysis for basketball",
    description:
      "A 0-99 similarity score and the three mechanics to fix first. Pose extracted frame-by-frame and aligned to an NBA reference.",
  },
};

export default function FollowThroughPage() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden border-b hairline">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse 900px 500px at 80% 0%, rgba(59,130,246,0.15), transparent 60%)," +
              "radial-gradient(ellipse 800px 400px at 10% 100%, rgba(225,29,72,0.12), transparent 60%)",
          }}
          aria-hidden
        />
        <div className="relative mx-auto max-w-6xl px-6 pt-20 pb-16 sm:pt-24">
          <div className="flex items-center gap-5 mb-8">
            <Image
              src="/branding/followthrough.png"
              alt="Follow Through"
              width={96}
              height={96}
              className="rounded-xl"
              priority
            />
            <div>
              <p className="text-3xl sm:text-4xl font-semibold tracking-tight text-foreground leading-none">
                Follow Through
              </p>
              <p className="mt-2 overline" style={{ color: "var(--color-ft-blue)" }}>
                Shot-form analysis
              </p>
            </div>
          </div>

          <h1 className="h-display max-w-4xl">
            Your shot,
            <br />
            <span className="text-muted">graded against the pros.</span>
          </h1>

          <p className="mt-8 text-lg sm:text-xl text-muted-strong max-w-3xl leading-relaxed">
            Record the shot. Follow Through extracts the shooter&apos;s pose,
            aligns every frame to a pro reference, and returns a 0-99
            similarity score plus the three mechanics to fix first. A coach
            can read it in thirty seconds.
          </p>
        </div>
      </section>

      {/* Reference + target clips */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <p
              className="overline mb-3"
              style={{ color: "var(--color-ft-blue)" }}
            >
              Reference -- Shaquille O&apos;Neal
            </p>
            <VideoClip src="/clips/shaq_overlay.mp4" />
          </div>
          <div>
            <p
              className="overline mb-3"
              style={{ color: "var(--color-ft-red)" }}
            >
              Target -- Steve Nash
            </p>
            <VideoClip src="/clips/nash_overlay.mp4" />
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-6xl px-6 py-16 border-t hairline">
        <p className="overline">Pipeline</p>
        <h2 className="mt-4 text-3xl sm:text-4xl font-semibold tracking-tight">
          How it works.
        </h2>
        <div className="mt-10 grid md:grid-cols-3 gap-5">
          <Card
            n="01"
            title="Pose extraction"
            body="MediaPipe lands 18 joints per frame. Savitzky-Golay smooths the jitter without flattening the snap at release."
          />
          <Card
            n="02"
            title="Normalisation"
            body="Centre on the hip midpoint; scale by the shoulder-to-hip distance. Camera angle and body size wash out, leaving only form."
          />
          <Card
            n="03"
            title="Deviation + scoring"
            body="Per-joint Euclidean distance to the reference, resampled to a common 60-frame axis. Mean deviation collapses into one number you can say out loud."
          />
        </div>
      </section>

      {/* Advice surface */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="relative rounded-3xl overflow-hidden border hairline bg-gradient-to-br from-surface-raised to-surface p-10 sm:p-14">
          <div
            className="absolute inset-0 pointer-events-none opacity-60"
            style={{
              background:
                "radial-gradient(circle at 85% 30%, rgba(225,29,72,0.15), transparent 40%)",
            }}
          />
          <div className="relative">
            <p className="overline" style={{ color: "var(--color-ft-red)" }}>
              The product layer
            </p>
            <h3 className="mt-4 text-3xl sm:text-4xl font-semibold tracking-tight max-w-3xl leading-tight">
              A score is the headline.
              <br />
              <span className="text-muted">
                Advice is what you do with it.
              </span>
            </h3>
            <p className="mt-6 text-muted-strong leading-relaxed max-w-3xl">
              The score answers <em>how close am I?</em> The coaching layer
              answers <em>what do I change?</em> For the three most-deviated
              joints, the system writes specific cues in plain English:
              vertical offsets (&quot;elbow drops 8% lower than Nash at
              release&quot;), lateral offsets (&quot;guide-hand wrist drifts
              across the body&quot;), and timing deltas. The whole analysis
              layer is pure functions -- no UI, no framework, ready to ship
              on a phone.
            </p>
            <dl className="mt-10 grid sm:grid-cols-3 gap-8">
              <Stat label="Joints tracked" value="18" />
              <Stat label="Score range" value="0 -- 99" />
              <Stat label="Unit tests on analysis" value="10" />
            </dl>
          </div>
        </div>
      </section>

      {/* Example output -- brings the live demo surface into the fold */}
      <section className="mx-auto max-w-6xl px-6 pb-16">
        <p className="overline">Example output -- Shaq vs. Nash</p>
        <h2 className="mt-4 text-3xl sm:text-4xl font-semibold tracking-tight">
          What the system actually returns.
        </h2>

        <div className="mt-10 grid lg:grid-cols-3 gap-5">
          {/* Score card -- real output from demo/analysis.py on shaq + nash keypoints */}
          <div
            className="card-raised rounded-2xl p-8 relative overflow-hidden"
            style={{ borderColor: "rgba(59,130,246,0.25)" }}
          >
            <p
              className="overline"
              style={{ color: "var(--color-ft-blue)" }}
            >
              Similarity score
            </p>
            <div className="mt-6 flex items-baseline gap-2">
              <span className="text-7xl sm:text-8xl font-semibold tracking-tight text-foreground leading-none">
                50
              </span>
              <span className="text-2xl font-semibold text-muted">/ 99</span>
            </div>
            <p className="mt-5 text-foreground font-medium">
              Noticeably different mechanics
            </p>
            <p className="mt-2 text-sm text-muted-strong leading-relaxed">
              Mean per-joint deviation of{" "}
              <span className="font-mono">0.251</span> across 18 shared
              joints over a 60-frame aligned window.
            </p>
          </div>

          {/* Advice card */}
          <div className="card-raised rounded-2xl p-8 lg:col-span-2">
            <p className="overline text-accent">Top-3 coaching cues</p>
            <ol className="mt-6 space-y-5">
              <AdviceLine
                n="01"
                joint="Guide-hand wrist"
                cue="Sits lower than Nash&rsquo;s. Lift it higher through the shot."
              />
              <AdviceLine
                n="02"
                joint="Shooting wrist"
                cue="Tucked too close vs. Nash&rsquo;s. Let it extend outward naturally."
              />
              <AdviceLine
                n="03"
                joint="Right heel"
                cue="Tucked too close vs. Nash&rsquo;s. Let the base open up on the release."
              />
            </ol>
          </div>
        </div>

        {/* Per-joint deviation preview -- real values from the analysis */}
        <div className="mt-6 card-raised rounded-2xl p-8">
          <p className="overline">Per-joint deviation</p>
          <p className="mt-3 text-muted-strong max-w-2xl leading-relaxed">
            Normalised Euclidean deviation per joint, descending. Top
            contributors drive the advice.
          </p>
          <div className="mt-8 space-y-3">
            <JointBar joint="Left wrist (guide)" pct={0.467} />
            <JointBar joint="Right wrist (shooting)" pct={0.432} />
            <JointBar joint="Right heel" pct={0.351} />
            <JointBar joint="Left heel" pct={0.326} />
            <JointBar joint="Left foot index" pct={0.316} />
            <JointBar joint="Right foot index" pct={0.310} />
            <JointBar joint="Left elbow" pct={0.300} />
            <JointBar joint="Right ankle" pct={0.295} />
          </div>
        </div>
      </section>

      {/* Synthetic data story */}
      <section className="mx-auto max-w-6xl px-6 py-12">
        <div className="rounded-3xl border hairline bg-surface p-8 sm:p-12">
          <p className="overline text-accent">Engineering note</p>
          <h2 className="mt-4 text-2xl sm:text-3xl font-semibold tracking-tight max-w-3xl">
            Training data was the hardest part. I got around it with{" "}
            <span className="text-accent">NBA 2K</span>.
          </h2>
          <p className="mt-5 text-muted-strong leading-relaxed max-w-3xl">
            Labelled pose data from real basketball film is expensive and
            scarce. Before MediaPipe made the custom-trained pose step
            unnecessary in production, I generated synthetic training data
            from NBA 2K -- a game engine that renders realistic shooting
            mechanics on demand, with programmatic control over camera,
            lighting, and player. Good enough as a proof of concept, good
            enough to test on, and a pattern that still works wherever
            training data is the bottleneck. (The production system now
            leans on MediaPipe&apos;s pretrained landmarks.)
          </p>
        </div>
      </section>

      {/* Stack */}
      <section className="mx-auto max-w-6xl px-6 pt-8 pb-20">
        <p className="overline">Under the hood</p>
        <h2 className="mt-4 text-2xl font-semibold tracking-tight">Stack.</h2>
        <ul className="mt-8 grid sm:grid-cols-2 gap-x-10 gap-y-3 text-muted-strong">
          <StackRow label="Pose">MediaPipe Pose</StackRow>
          <StackRow label="Smoothing">Savitzky-Golay, per-joint</StackRow>
          <StackRow label="Analysis">Pure Python: NumPy + Pandas, zero UI deps</StackRow>
          <StackRow label="Tests">pytest; contract tests on every public function</StackRow>
          <StackRow label="Mobile target">KMP shared + Compose / SwiftUI native UI</StackRow>
          <StackRow label="Video I/O">OpenCV + H.264 transcode for browser delivery</StackRow>
        </ul>
      </section>
    </>
  );
}

function Card({ n, title, body }: { n: string; title: string; body: string }) {
  return (
    <div className="card-raised rounded-2xl p-6">
      <p className="text-xs font-mono text-accent">{n}</p>
      <h3 className="mt-3 text-lg font-semibold text-foreground tracking-tight">
        {title}
      </h3>
      <p className="mt-3 text-sm text-muted-strong leading-relaxed">{body}</p>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="overline">{label}</dt>
      <dd className="mt-3 text-4xl font-semibold text-foreground tracking-tight">
        {value}
      </dd>
    </div>
  );
}

function StackRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <li className="flex gap-4 items-baseline border-b hairline pb-3">
      <span className="overline w-36 flex-shrink-0">{label}</span>
      <span className="text-foreground">{children}</span>
    </li>
  );
}

function AdviceLine({
  n,
  joint,
  cue,
}: {
  n: string;
  joint: string;
  cue: string;
}) {
  return (
    <li className="flex gap-4 items-start">
      <span className="text-xs font-mono text-accent pt-1 w-6 flex-shrink-0">
        {n}
      </span>
      <div>
        <p className="text-foreground font-semibold tracking-tight">
          {joint}
        </p>
        <p className="mt-1 text-muted-strong leading-relaxed">{cue}</p>
      </div>
    </li>
  );
}

function JointBar({ joint, pct }: { joint: string; pct: number }) {
  // pct is the raw mean deviation; the scoring function treats 0.5 as the
  // "maximally different" end of the scale, so we visualise width against
  // that ceiling while the label stays true to the raw value.
  const widthPct = Math.min(pct / 0.5, 1) * 100;
  return (
    <div className="flex items-center gap-3 sm:gap-4">
      <span className="w-28 sm:w-48 flex-shrink-0 text-xs sm:text-sm text-muted-strong">
        {joint}
      </span>
      <div className="flex-1 h-2 rounded-full bg-background overflow-hidden border hairline">
        <div
          className="h-full rounded-full"
          style={{
            width: `${widthPct}%`,
            background:
              "linear-gradient(to right, var(--color-ft-blue), var(--color-ft-red))",
          }}
        />
      </div>
      <span className="w-12 text-right text-xs font-mono text-muted">
        {pct.toFixed(3)}
      </span>
    </div>
  );
}
