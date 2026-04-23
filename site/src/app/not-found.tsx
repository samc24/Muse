import Image from "next/image";
import Link from "next/link";

export const metadata = {
  title: "Not found",
};

export default function NotFound() {
  return (
    <section className="relative overflow-hidden">
      <div className="hero-aura" aria-hidden />
      <div className="relative mx-auto max-w-3xl px-6 py-24 sm:py-32 flex flex-col items-center text-center">
        <Image
          src="/branding/muse.png"
          alt="Muse"
          width={256}
          height={256}
          className="w-28 h-28 sm:w-36 sm:h-36 object-contain"
          priority
        />

        <p className="mt-10 overline text-accent">404 -- off the court</p>
        <h1 className="mt-4 h-display">
          This page
          <br />
          <span className="text-muted">wasn&apos;t drafted.</span>
        </h1>
        <p className="mt-6 text-lg text-muted-strong max-w-xl leading-relaxed">
          Nothing lives at this URL yet. Head back to the home page, or jump
          straight into one of the projects.
        </p>

        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <Link href="/" className="btn btn-primary btn-lg">
            Back to home
          </Link>
          <Link href="/follow-through" className="btn btn-secondary btn-lg">
            Follow Through
          </Link>
          <Link href="/eyeball" className="btn btn-secondary btn-lg">
            EyeBall
          </Link>
        </div>
      </div>
    </section>
  );
}
