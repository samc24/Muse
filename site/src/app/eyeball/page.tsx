import Image from "next/image";
import { VideoClip } from "@/components/VideoClip";

export const metadata = {
  title: "EyeBall",
  description:
    "Follow the ball across game film. Kalman-filtered trajectories, pass detection, and a virtual-panning broadcast camera for amateur footage.",
  openGraph: {
    title: "EyeBall -- ball tracking and virtual broadcast for basketball",
    description:
      "Trajectory, passes, and downstream broadcast from the ball signal alone. Classical CV + Kalman now; YOLO next.",
  },
};

export default function EyeBallPage() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden border-b hairline">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse 1000px 500px at 20% 0%, rgba(249,115,22,0.18), transparent 55%)," +
              "radial-gradient(ellipse 800px 400px at 90% 100%, rgba(249,115,22,0.08), transparent 60%)",
          }}
          aria-hidden
        />
        <div className="relative mx-auto max-w-6xl px-6 pt-20 pb-16 sm:pt-24">
          <div className="mb-10">
            <Image
              src="/branding/eyeball.png"
              alt="EyeBall"
              width={720}
              height={200}
              className="h-16 sm:h-20 w-auto invert"
              priority
            />
            <p className="mt-4 overline text-accent">Ball tracking</p>
          </div>

          <h1 className="h-display max-w-4xl">
            Follow the ball.
            <br />
            <span className="text-muted">Understand the play.</span>
          </h1>

          <p className="mt-8 text-lg sm:text-xl text-muted-strong max-w-3xl leading-relaxed">
            EyeBall follows the basketball across game film. A Kalman filter
            stitches noisy detections into a smooth trajectory; its shape
            tells you when passes happen. Layer in other signals -- dribbles,
            player position, clock state -- and you get actions, and from
            actions, plays: the coaching surface the box score never
            captures. The same primitive also drives a synthetic broadcast
            camera for amateur footage, no operator required.
          </p>
        </div>
      </section>

      {/* Case 01 */}
      <section className="mx-auto max-w-6xl px-6 pt-16 pb-10">
        <div className="grid md:grid-cols-5 gap-10">
          <div className="md:col-span-2">
            <p className="overline text-accent">Case 01 -- Tight tracking</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              Jordan, drive to the rim.
            </h2>
            <p className="mt-4 text-muted-strong leading-relaxed">
              A classic crossover-and-finish. The ball changes direction and
              speed twice per second and briefly disappears behind MJ&apos;s
              body at the rim. The system holds the trajectory through all of
              it -- the kind of possession the box score never explains.
              That&apos;s the starting point for any downstream analytics:
              if you can&apos;t follow the ball, you can&apos;t measure the
              play.
            </p>
          </div>
          <div className="md:col-span-3">
            <VideoClip src="/clips/jordan_tracked.mp4" />
          </div>
        </div>
      </section>

      {/* Case 02 */}
      <section className="mx-auto max-w-6xl px-6 pt-16 pb-10 border-t hairline">
        <div className="grid md:grid-cols-5 gap-10">
          <div className="md:col-span-2">
            <p className="overline text-accent">Case 02 -- Possession breakdown</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              Spurs at Houston, half-court.
            </h2>
            <p className="mt-4 text-muted-strong leading-relaxed">
              One half-court possession. The system watches the ball move
              between players and stamps a timestamp every time the
              trajectory changes direction sharply enough to register as
              a pass. It also over-counts here -- classical CV has no way
              to distinguish a real pass from a crisp dribble reversal.
              That&apos;s the natural limitation of a ball-only signal.
            </p>
            <p className="mt-4 text-muted-strong leading-relaxed">
              Passes are a starting point, not the answer. Combine them
              with dribble detection, player position, and shot-clock
              state and you get action detection -- pick-and-roll,
              hand-off, off-ball screen, isolation -- and from there,
              play detection. That&apos;s the coaching surface I&apos;m
              building toward: the box score never tells you which
              actions actually got run.
            </p>
            <p className="mt-4 text-xs text-muted leading-relaxed">
              Under the hood, no training data required: HSV + contour
              detection, Kalman smoothing, outlier rejection, and an
              angle-based pass detector with a cooldown gate.
            </p>
            <div className="mt-6 grid grid-cols-1 gap-3">
              <MiniStat k="Detection" v="HSV + contours" />
              <MiniStat k="Smoothing" v="Two-pass Savgol + outlier rejection" />
              <MiniStat k="Events" v="Angle-based pass detection" />
            </div>
          </div>
          <div className="md:col-span-3">
            <VideoClip src="/clips/houston_tracked.mp4" />
          </div>
        </div>

        {/* Real output from the Houston run */}
        <div className="mt-10 card-raised rounded-2xl p-8">
          <p className="overline text-accent">
            Output -- Houston half-court run
          </p>
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-6">
            <BigStat label="Passes detected" value="7" />
            <BigStat label="Detection rate" value="70%" sub="294 / 420 frames" />
            <BigStat label="Clip duration" value="14.0s" />
            <BigStat label="Trajectory" value="4,796px" sub="total path length" />
          </div>

          <div className="mt-8">
            <p className="overline">Pass timeline (seconds)</p>
            <div className="mt-4 relative h-10">
              <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-px bg-border-strong" />
              {[0.2, 1.733, 6.1, 6.867, 7.933, 9.6, 12.267].map((t, i) => (
                <div
                  key={i}
                  className="absolute top-0 flex flex-col items-center"
                  style={{ left: `${(t / 14) * 100}%`, transform: "translateX(-50%)" }}
                >
                  <span className="w-2 h-2 rounded-full bg-accent mt-[18px]" />
                  <span className="text-[10px] font-mono text-muted mt-1">
                    {t.toFixed(1)}s
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Broadcasting */}
      <section className="mx-auto max-w-6xl px-6 pt-20 pb-16 border-t hairline">
        <div className="max-w-3xl">
          <p className="overline text-accent">Downstream application</p>
          <h2 className="mt-4 text-4xl sm:text-5xl font-semibold tracking-tight leading-[1.05]">
            Virtual panning.
            <br />
            <span className="text-muted">No operator, still a broadcast.</span>
          </h2>
          <p className="mt-6 text-muted-strong leading-relaxed text-lg">
            Almost no amateur game has a camera operator. A stationary wide
            panorama turns into a real broadcast by tracking the action and
            rendering a synthetic 16:9 pan over it. Motion inside a court ROI
            drives the pan target; Savitzky-Golay keeps the crop from
            jittering.
          </p>
        </div>

        <div className="mt-10 grid md:grid-cols-2 gap-8">
          <div>
            <p className="overline text-accent mb-3">Source</p>
            <VideoClip src="/clips/broadcasting_source.mp4" />
          </div>
          <div>
            <p className="overline text-accent mb-3">Output</p>
            <VideoClip src="/clips/broadcasting_output.mp4" />
          </div>
        </div>

      </section>

      {/* Synthetic data story */}
      <section className="mx-auto max-w-6xl px-6 py-12 border-t hairline">
        <div className="rounded-3xl border hairline bg-surface p-8 sm:p-12">
          <p className="overline text-accent">Engineering note</p>
          <h2 className="mt-4 text-2xl sm:text-3xl font-semibold tracking-tight max-w-3xl">
            No training data? Generate it with{" "}
            <span className="text-accent">NBA 2K</span>.
          </h2>
          <p className="mt-5 text-muted-strong leading-relaxed max-w-3xl">
            Good labelled basketball film -- with the ball picked out on
            every frame -- is rare. Before classical CV got the current
            pipeline running, I trained the early ball-detection models on
            synthetic data generated from NBA 2K: realistic geometry,
            controllable camera, and a game engine I could produce labelled
            frames from on demand. Not a shortcut to production, but a
            legitimate proof of concept and a reliable test bed wherever a
            real labelled set doesn&apos;t exist yet.
          </p>
        </div>
      </section>

      {/* Stack */}
      <section className="mx-auto max-w-6xl px-6 pt-12 pb-20 border-t hairline">
        <p className="overline">Under the hood</p>
        <h2 className="mt-4 text-2xl font-semibold tracking-tight">Stack.</h2>
        <ul className="mt-8 grid sm:grid-cols-2 gap-x-10 gap-y-3 text-muted-strong">
          <StackRow label="Detection">HSV + contours (live); YOLO (production upgrade)</StackRow>
          <StackRow label="State estimation">Kalman filter with outlier rejection on a speed gate</StackRow>
          <StackRow label="Smoothing">Two-pass Savitzky-Golay with linear interpolation through gaps</StackRow>
          <StackRow label="Events">Angle-based pass detection with cooldown + min-motion gate</StackRow>
          <StackRow label="Broadcasting">Court-ROI motion envelope, Savgol-smoothed 16:9 crop</StackRow>
          <StackRow label="Stack">Python 3.12, OpenCV 4.10, NumPy, SciPy, imutils</StackRow>
        </ul>
      </section>
    </>
  );
}

function MiniStat({ k, v }: { k: string; v: string }) {
  return (
    <div className="card-raised rounded-xl px-5 py-3.5">
      <p className="text-[10px] uppercase tracking-[0.16em] text-muted font-mono">
        {k}
      </p>
      <p className="mt-1 text-sm text-foreground">{v}</p>
    </div>
  );
}

function BigStat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div>
      <p className="overline">{label}</p>
      <p className="mt-2 text-3xl sm:text-4xl font-semibold text-foreground tracking-tight">
        {value}
      </p>
      {sub && <p className="mt-1 text-xs text-muted">{sub}</p>}
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
