import Link from "next/link";

export const metadata = {
  title: "Not found",
};

export default function NotFound() {
  return (
    <section className="relative overflow-hidden">
      <div className="hero-aura" aria-hidden />
      <div className="relative mx-auto max-w-3xl px-6 py-24 sm:py-32 flex flex-col items-center text-center">
        <p className="overline text-accent">404</p>
        <h1 className="mt-4 h-display">Airball.</h1>
        <p className="mt-6 text-lg text-muted-strong max-w-xl leading-relaxed">
          That page doesn&apos;t exist. Head back home, or check out the work
          or Muse.
        </p>

        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link href="/" className="btn btn-primary btn-lg">
            Back to home
          </Link>
          <Link href="/work" className="btn btn-secondary btn-lg">
            Work
          </Link>
          <Link href="/muse" className="btn btn-secondary btn-lg">
            Muse
          </Link>
        </div>
      </div>
    </section>
  );
}
