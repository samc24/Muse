import Link from "next/link";
import Image from "next/image";
import { ProjectCard } from "@/components/ProjectCard";
import { CredentialStrip } from "@/components/CredentialStrip";

export default function Home() {
  return (
    <>
      {/* Hero -- split layout, prominent Muse mark, no photo caption */}
      <section className="relative overflow-hidden border-b hairline">
        <div className="hero-aura" aria-hidden />
        <div className="absolute inset-0 grid-lines" aria-hidden />

        <div className="relative mx-auto max-w-6xl px-6 pt-20 pb-20 sm:pt-24 sm:pb-24">
          <div className="grid lg:grid-cols-12 gap-10 lg:gap-14 items-center">
            <div className="lg:col-span-7">
              <div className="flex items-center gap-4 sm:gap-6">
                <Image
                  src="/branding/muse.png"
                  alt="Muse"
                  width={256}
                  height={256}
                  className="w-28 h-28 sm:w-40 sm:h-40 md:w-52 md:h-52 object-contain flex-shrink-0"
                  priority
                />
                <div>
                  <p className="overline">
                    Computer vision for basketball
                  </p>
                  <p className="mt-2 text-sm text-muted">
                    Pose analysis. Ball tracking. Broadcast automation.
                  </p>
                </div>
              </div>

              <h1 className="mt-10 h-display text-foreground">
                Bridge the
                <br />
                <span className="text-accent">eye-test</span>
                <br />
                <span className="text-muted">and counting stats.</span>
              </h1>

              <p className="mt-8 max-w-xl text-lg sm:text-xl text-muted-strong leading-relaxed">
                Full-stack engineer -- ML, mobile, and web. I build
                end-to-end systems for basketball: film in; numbers,
                insights, and interfaces a coach can use, a scout can
                trust, and a front office can act on come out.
              </p>

              <div className="mt-10 flex flex-wrap gap-3">
                <Link href="/follow-through" className="btn btn-primary btn-lg">
                  Follow Through
                </Link>
                <Link href="/eyeball" className="btn btn-secondary btn-lg">
                  EyeBall
                </Link>
                <a
                  href="mailto:sam.chaturvedi24@gmail.com"
                  className="btn btn-secondary btn-lg"
                >
                  Get in touch
                </a>
              </div>
            </div>

            <div className="lg:col-span-5">
              <div className="relative">
                <div
                  className="absolute -inset-4 rounded-[2rem] bg-gradient-to-br from-accent/20 via-transparent to-ft-blue/15 blur-2xl pointer-events-none"
                  aria-hidden
                />
                <div className="relative rounded-2xl overflow-hidden border hairline-strong bg-black aspect-square max-w-md mx-auto">
                  <Image
                    src="/photos/medellin_hoop.jpg"
                    alt="Sameer playing basketball on a Medellín street court"
                    width={900}
                    height={900}
                    className="w-full h-full object-cover"
                    priority
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Thesis band -- pulled from your own words */}
      <section className="border-b hairline bg-surface/40">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <div className="grid md:grid-cols-5 gap-10 items-start">
            <div className="md:col-span-2">
              <p className="overline text-accent">Thesis</p>
            </div>
            <blockquote className="md:col-span-3 text-xl sm:text-2xl font-medium tracking-tight text-foreground leading-snug">
              Basketball has been the main constant in my life. I build AI for
              basketball to give back to the game that has given so much --
              for coaching, scouting, and training.
            </blockquote>
          </div>
        </div>
      </section>

      {/* Credentials */}
      <section className="border-b hairline">
        <div className="mx-auto max-w-6xl px-6 py-14">
          <CredentialStrip />
        </div>
      </section>

      {/* Bio strip */}
      <section className="border-b hairline bg-surface/30">
        <div className="mx-auto max-w-6xl px-6 py-16 grid md:grid-cols-5 gap-10">
          <div className="md:col-span-2">
            <p className="overline">Who</p>
            <p className="mt-4 text-2xl font-semibold tracking-tight text-foreground leading-snug">
              Full-stack engineer at Amazon and MIT. ML, mobile, web.
              Basketball lifer building the tools I wish we had.
            </p>
          </div>
          <div className="md:col-span-3 prose-muse">
            <p>
              Production ML at <strong>Amazon</strong> (Prime Video Browse
              and Discovery) and scientific software at <strong>MIT&apos;s
              Plasma Science and Fusion Center</strong> (DisruptionPy, a
              framework analysing plasma-fusion data across five Tokamak
              reactors).
            </p>
            <p>
              Earlier in 2025, Lead ML Engineer at{" "}
              <strong>Cerebro Sports</strong> -- a full-stack app for live
              basketball-tournament analytics: LLM chat over a stats
              database, realtime graphs, dashboards, text-to-SQL. Before
              that, four years on <strong>Microsoft&apos;s</strong> M365
              Security Detections team, where I led the first production
              mail-anomaly detection system (response to the nation-state
              email-breach incident) and designed the agile PySpark
              workspace the team ships on.
            </p>
            <p>
              Earlier still: <strong>AI.Reverie</strong>, where I led a
              Swift + AVFoundation iOS app running a PyTorch YOLO model
              on-device, and <strong>Red Hat OpenShift</strong>, upstream
              contributions to the Kubernetes container runtime (CRI-O,
              Libpod). Boston University: MS in AI + Deep Learning, BS in
              Computer Science + Astrophysics, Presidential Scholar.
            </p>
          </div>
        </div>
      </section>

      {/* Projects */}
      <section className="mx-auto max-w-6xl px-6 py-24">
        <div className="max-w-3xl">
          <p className="overline">The work</p>
          <h2 className="mt-4 text-4xl sm:text-5xl font-semibold tracking-tight leading-[1.05]">
            Two systems.
            <br />
            <span className="text-muted">One game.</span>
          </h2>
          <p className="mt-6 text-muted-strong text-lg leading-relaxed">
            Follow Through grades a shooter&apos;s form against NBA pros and
            returns the three mechanics to fix. EyeBall follows the ball
            across game film, derives passes and possessions, and turns a
            stationary wide shot into a broadcast that pans the action.
          </p>
        </div>

        <div className="mt-12 grid md:grid-cols-2 gap-6">
          <ProjectCard
            href="/follow-through"
            logo="/branding/followthrough.png"
            logoAlt="Follow Through"
            name="Follow Through"
            tagline="Shot-form analysis"
            summary="Eighteen joints, tracked frame-by-frame, aligned to a pro reference. A single 0-99 score for how close you are. The three mechanics to work on first."
            posterVideo="/clips/shaq_overlay.mp4"
            accent="blue"
          />
          <ProjectCard
            href="/eyeball"
            logo="/branding/eyeball.png"
            logoAlt="EyeBall"
            name="EyeBall"
            tagline="Ball tracking + broadcasting"
            summary="Follow the ball. Find passes and possession boundaries in its path. Then use that primitive to pan a synthetic broadcast camera across amateur footage -- no operator required."
            posterVideo="/clips/houston_tracked.mp4"
            accent="orange"
            invertLogo
          />
        </div>
      </section>

      {/* Closing CTA */}
      <section className="mx-auto max-w-6xl px-6 pb-24">
        <div className="relative rounded-3xl overflow-hidden border hairline-strong bg-gradient-to-br from-surface-raised via-surface to-ink px-8 py-14 sm:px-14 sm:py-20">
          <div className="absolute -top-20 -right-20 w-80 h-80 rounded-full bg-accent/15 blur-3xl pointer-events-none" />
          <div className="relative max-w-3xl">
            <p className="overline text-accent">Let&apos;s talk</p>
            <h3 className="mt-4 text-3xl sm:text-4xl font-semibold tracking-tight leading-tight">
              Work on basketball, together.
            </h3>
            <p className="mt-4 text-muted-strong leading-relaxed max-w-2xl">
              Open to conversations with teams, front offices, and product
              people -- on basketball analytics, on-device ML, iOS and web
              product work for small teams, broadcast automation, and the
              eye-test layer of scouting.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a
                href="mailto:sam.chaturvedi24@gmail.com"
                className="btn btn-primary btn-lg"
              >
                Email me
              </a>
              <a
                href="https://linkedin.com/in/sameerc"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-secondary btn-lg"
              >
                LinkedIn
              </a>
              <a
                href="https://github.com/samc24"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-secondary btn-lg"
              >
                GitHub
              </a>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
