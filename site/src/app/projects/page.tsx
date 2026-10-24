import { PROJECTS } from "@/content";
import { EntryCard } from "@/components/EntryCard";

export const metadata = {
  title: "Projects",
  description: "Side projects by Sameer Chaturvedi. Muse: computer vision for basketball.",
};

export default function ProjectsIndexPage() {
  return (
    <>
      <section className="relative overflow-hidden border-b hairline">
        <div className="hero-aura" aria-hidden />
        <div className="relative mx-auto max-w-6xl px-6 pt-20 pb-16 sm:pt-24">
          <p className="overline">Projects</p>
          <h1 className="mt-5 h-display max-w-4xl">
            Basketball
            <br />
            <span className="text-muted">is my muse.</span>
          </h1>
          <p className="mt-8 text-lg sm:text-xl text-muted-strong max-w-3xl leading-relaxed">
            Projects I work on because I care about them, outside of work.
            Muse is the long-running one: computer vision for basketball
            coaching, scouting, and training, from the models to the phone
            app I&apos;m building around them.
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
