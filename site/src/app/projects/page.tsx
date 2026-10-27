import { PROJECTS } from "@/content";
import { EntryCard } from "@/components/EntryCard";

export const metadata = {
  title: "Projects",
  description: "Ongoing side projects by Sameer Chaturvedi: Muse, computer vision for basketball, and Exploding Catan, a browser board game with bots and online play.",
};

export default function ProjectsIndexPage() {
  return (
    <>
      <section className="relative overflow-hidden border-b hairline">
        <div className="hero-aura" aria-hidden />
        <div className="relative mx-auto max-w-6xl px-6 pt-20 pb-16 sm:pt-24">
          <p className="overline">Ongoing projects</p>
          <h1 className="mt-5 h-display max-w-4xl">
            Engineering is how
            <br />
            <span className="text-muted">I realize my curiosity.</span>
          </h1>
          <p className="mt-8 text-lg sm:text-xl text-muted-strong max-w-3xl leading-relaxed">
            There are always more problems to solve and more tools to build.
            These are the ones I&apos;m actively working on outside of work, and
            they were passion projects long before any of it was a
            professional opportunity. Muse is the long-running one: computer
            vision for basketball coaching, scouting, and training, and my
            muse for upskilling technologically and practicing discipline.
            Exploding Catan is the newer one: a friend&apos;s browser board game
            rebuilt with a real rules engine, bots, and online play, so game
            night works with a room code. Projects that have gone quiet live
            on GitHub.
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid md:grid-cols-2 gap-6">
          {PROJECTS.map((p) => (
            <EntryCard key={p.slug} entry={p} />
          ))}
        </div>
      </section>
    </>
  );
}
