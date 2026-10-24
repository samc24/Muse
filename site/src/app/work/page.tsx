import { WORK } from "@/content";
import { EntryCard } from "@/components/EntryCard";

export const metadata = {
  title: "Work",
  description:
    "Where Sameer Chaturvedi has worked: Amazon Prime Video, MIT PSFC, Cerebro Sports, Microsoft, AI.Reverie, Red Hat.",
};

export default function WorkIndexPage() {
  return (
    <>
      <section className="relative overflow-hidden border-b hairline">
        <div className="hero-aura" aria-hidden />
        <div className="relative mx-auto max-w-6xl px-6 pt-20 pb-16 sm:pt-24">
          <p className="overline">Work</p>
          <h1 className="mt-5 h-display max-w-4xl">
            Where I&apos;ve
            <br />
            <span className="text-muted">worked.</span>
          </h1>
          <p className="mt-8 text-lg sm:text-xl text-muted-strong max-w-3xl leading-relaxed">
            Currently at Amazon Prime Video and MIT&apos;s Plasma Science and
            Fusion Center. Before that: four years at Microsoft on M365
            security, a short contract at Cerebro Sports, on-device computer vision
            at AI.Reverie, and Kubernetes container runtimes at Red Hat.
          </p>
          <div className="mt-8">
            <a
              href="/Sameer_Chaturvedi_Resume.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary"
            >
              Resume (PDF)
            </a>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid md:grid-cols-2 gap-6">
          {WORK.map((w) => (
            <EntryCard key={w.slug} entry={w} />
          ))}
        </div>
      </section>
    </>
  );
}
