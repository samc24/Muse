import Link from "next/link";
import Image from "next/image";
import { ProjectCard } from "@/components/ProjectCard";

export const metadata = {
  title: "Muse",
  description:
    "Muse: computer vision for basketball. Follow Through grades shot form against NBA pros. EyeBall tracks the ball, finds passes and possessions, and pans a synthetic broadcast camera.",
  openGraph: {
    title: "Muse -- computer vision for basketball",
    description:
      "Pose analysis. Ball tracking. Broadcast automation. Built by Sameer Chaturvedi.",
  },
};

export default function MusePage() {
  return (
    <>
      {/* Hero -- split layout, prominent Muse mark */}
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
                  <p className="overline">Computer vision for basketball</p>
                  <p className="mt-2 text-sm text-muted">
                    Pose analysis. Ball tracking. Broadcast automation.
                  </p>
                </div>
              </div>

              <h1 className="mt-10 h-display text-foreground">
                Bridging the gap between the{" "}
                <span className="text-accent">eye-test</span>
                <br />
                <span className="text-muted">and counting stats.</span>
              </h1>

              <p className="mt-8 max-w-xl text-lg sm:text-xl text-muted-strong leading-relaxed">
                There&apos;s a common debate in basketball analytics between
                the eye-test and counting stats. I think computer vision on
                game footage bridges it: meaningful insights from the plays
                themselves, for coaching, scouting, and training, without
                someone having to watch every minute of every game.
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

      {/* Thesis band -- Sameer's own words */}
      <section className="border-b hairline bg-surface/40">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <div className="grid md:grid-cols-5 gap-10 items-start">
            <div className="md:col-span-2">
              <p className="overline text-accent">Why Muse</p>
            </div>
            <blockquote className="md:col-span-3 text-xl sm:text-2xl font-medium tracking-tight text-foreground leading-snug">
              Basketball is my most consistent and biggest passion, and my
              muse for upskilling technologically, practicing discipline, and
              learning about life.
            </blockquote>
          </div>
        </div>
      </section>

      {/* Projects */}
      <section className="mx-auto max-w-6xl px-6 py-24">
        <div className="max-w-3xl">
          <p className="overline">The work</p>
          <h2 className="mt-4 text-4xl sm:text-5xl font-semibold tracking-tight leading-[1.05]">
            Turn players
            <br />
            <span className="text-muted">into ballers.</span>
          </h2>
          <p className="mt-6 text-muted-strong text-lg leading-relaxed">
            I started Muse while at Boston University, with the goal of
            democratizing access to sophisticated basketball coaching and
            analysis. Follow Through looks at the individual: your shot
            mechanics compared with NBA players. EyeBall looks at the group:
            following the ball through game film to find passes, possessions,
            and eventually plays, and to broadcast amateur games without a
            camera operator.
          </p>
        </div>

        <div className="mt-12 grid md:grid-cols-2 gap-6">
          <ProjectCard
            href="/follow-through"
            logo="/branding/followthrough.png"
            logoAlt="Follow Through"
            name="Follow Through"
            tagline="Shot-form analysis"
            summary="Record your shot and get immediate feedback: your mechanics compared frame by frame with an NBA player, a 0-99 similarity score, and the three things to work on first."
            posterVideo="/clips/shaq_overlay.mp4"
            accent="blue"
          />
          <ProjectCard
            href="/eyeball"
            logo="/branding/eyeball.png"
            logoAlt="EyeBall"
            name="EyeBall"
            tagline="Ball tracking + broadcasting"
            summary="Follows the ball through game film to pick out passes and possessions, and uses the same tracking to turn a fixed wide-angle camera into a broadcast that pans with the action, no operator needed."
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
            <p className="overline text-accent">Get in touch</p>
            <h3 className="mt-4 text-3xl sm:text-4xl font-semibold tracking-tight leading-tight">
              Let&apos;s talk about the game.
            </h3>
            <p className="mt-4 text-muted-strong leading-relaxed max-w-2xl">
              If you work with a team, a league, or a training program and
              have problems with game film analysis or player development,
              I&apos;d love to hear about them. I&apos;m also looking for
              guidance on securing multi-angle footage from pro leagues to
              train better models.
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
