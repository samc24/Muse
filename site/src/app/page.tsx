import Link from "next/link";
import Image from "next/image";
import { WORK, PROJECTS, CURRENT, hrefFor } from "@/content";
import { EntryCard } from "@/components/EntryCard";
import { CredentialStrip } from "@/components/CredentialStrip";

export default function Home() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden border-b hairline">
        <div className="hero-aura" aria-hidden />
        <div className="absolute inset-0 grid-lines" aria-hidden />

        <div className="relative mx-auto max-w-6xl px-6 pt-20 pb-20 sm:pt-24 sm:pb-24">
          <div className="grid lg:grid-cols-12 gap-10 lg:gap-14 items-center">
            <div className="lg:col-span-7">
              <p className="overline">Sameer Chaturvedi</p>
              <h1 className="mt-5 h-display text-foreground">
                ML + software
                <br />
                <span className="text-muted">engineer.</span>
              </h1>

              <p className="mt-8 max-w-xl text-lg sm:text-xl text-muted-strong leading-relaxed">
                Machine Learning Engineer at Amazon Prime Video and Lead
                Software Engineer at MIT&apos;s Plasma Science and Fusion
                Center. Before that, four years at Microsoft on
                petabyte-scale security ML.
              </p>

              <p className="mt-5 max-w-xl text-lg sm:text-xl text-muted-strong leading-relaxed">
                I studied computer science and astrophysics, and I&apos;ve
                spent my career where ML meets something I care about:
                recommendation systems, fusion research, cyber security, and
                computer vision for basketball.
              </p>

              <div className="mt-10 flex flex-wrap gap-3">
                <Link href="/work" className="btn btn-primary btn-lg">
                  Work
                </Link>
                <Link href="/muse" className="btn btn-secondary btn-lg">
                  Muse
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
                <div className="relative rounded-2xl overflow-hidden border hairline-strong bg-black max-w-md mx-auto">
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

      {/* Now */}
      <section className="border-b hairline bg-surface/40">
        <div className="mx-auto max-w-6xl px-6 py-14">
          <div className="grid md:grid-cols-5 gap-10 items-start">
            <div className="md:col-span-2">
              <p className="overline text-accent">Currently</p>
            </div>
            <div className="md:col-span-3 grid sm:grid-cols-2 gap-6">
              {CURRENT.map((w) => (
                <Link
                  key={w.slug}
                  href={hrefFor(w)}
                  className="group border-l hairline-strong pl-5 hover:border-accent transition-colors"
                >
                  <p className="text-xl font-semibold tracking-tight text-foreground group-hover:text-accent transition-colors">
                    {w.org}
                  </p>
                  <p className="mt-1 text-sm text-muted">{w.role}</p>
                  <p className="mt-3 text-muted-strong leading-relaxed text-[0.95rem]">
                    {w.summary}
                  </p>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Credentials strip */}
      <section className="border-b hairline">
        <div className="mx-auto max-w-6xl px-6 py-14">
          <CredentialStrip />
        </div>
      </section>

      {/* Work */}
      <section className="mx-auto max-w-6xl px-6 py-24">
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6">
          <div className="max-w-3xl">
            <p className="overline">Work</p>
            <h2 className="mt-4 text-4xl sm:text-5xl font-semibold tracking-tight leading-[1.05]">
              From container runtimes
              <br />
              <span className="text-muted">to recommendation models.</span>
            </h2>
            <p className="mt-6 text-muted-strong text-lg leading-relaxed">
              Production ML at Amazon, scientific software at MIT, four years
              of security engineering at Microsoft, and earlier stints in
              live basketball analytics, on-device computer vision, and
              Kubernetes internals.
            </p>
          </div>
          <Link href="/work" className="btn btn-secondary">
            All work
          </Link>
        </div>

        <div className="mt-12 grid md:grid-cols-2 gap-6">
          {WORK.map((w) => (
            <EntryCard key={w.slug} entry={w} />
          ))}
        </div>
      </section>

      {/* Projects */}
      <section className="mx-auto max-w-6xl px-6 pb-24">
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-6">
          <div className="max-w-3xl">
            <p className="overline">Side projects</p>
            <h2 className="mt-4 text-4xl sm:text-5xl font-semibold tracking-tight leading-[1.05]">
              Outside
              <br />
              <span className="text-muted">of work.</span>
            </h2>
            <p className="mt-6 text-muted-strong text-lg leading-relaxed">
              Muse is the long-running one: computer vision for basketball
              coaching, scouting, and training. The name comes from
              basketball being my muse for upskilling technologically and
              practicing discipline. Exploding Catan is the newer one: a
              board game I play with friends, rebuilt with a rules engine,
              bots, and online play.
            </p>
          </div>
          <Link href="/projects" className="btn btn-secondary">
            All projects
          </Link>
        </div>

        <div className="mt-12 grid md:grid-cols-2 gap-6">
          {PROJECTS.map((p) => (
            <EntryCard key={p.slug} entry={p} />
          ))}
        </div>
      </section>

      {/* Closing CTA */}
      <section className="mx-auto max-w-6xl px-6 pb-24">
        <div className="relative rounded-3xl overflow-hidden border hairline-strong bg-gradient-to-br from-surface-raised via-surface to-ink px-8 py-14 sm:px-14 sm:py-20">
          <div className="absolute -top-20 -right-20 w-80 h-80 rounded-full bg-accent/15 blur-3xl pointer-events-none" />
          <div className="relative max-w-3xl">
            <p className="overline text-accent">Get in touch</p>
            <h3 className="mt-4 text-3xl sm:text-4xl font-semibold tracking-tight leading-tight">
              Would love to connect.
            </h3>
            <p className="mt-4 text-muted-strong leading-relaxed max-w-2xl">
              Whether it&apos;s ML systems, scientific software, basketball
              tech, or a hard engineering problem, I&apos;d love to hear
              about it. Email is the fastest way to reach me.
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
              <a
                href="/Sameer_Chaturvedi_Resume.pdf"
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-secondary btn-lg"
              >
                Resume
              </a>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
