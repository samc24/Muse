import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t hairline mt-24">
      <div className="mx-auto max-w-6xl px-6 py-10 flex flex-col sm:flex-row gap-6 sm:items-end sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-muted">
            Sameer Chaturvedi
          </p>
          <p className="mt-2 text-muted-strong max-w-md">
            ML + software engineer. Ball is life.
          </p>
        </div>

        <div className="flex flex-col sm:items-end text-sm">
          <div className="flex flex-wrap gap-5">
            <a
              href="mailto:sam.chaturvedi24@gmail.com"
              className="text-muted-strong hover:text-accent transition-colors"
            >
              Email
            </a>
            <a
              href="https://github.com/samc24"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-strong hover:text-accent transition-colors"
            >
              GitHub
            </a>
            <a
              href="https://linkedin.com/in/sameerc"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-strong hover:text-accent transition-colors"
            >
              LinkedIn
            </a>
            <Link
              href="/work"
              className="text-muted-strong hover:text-accent transition-colors"
            >
              Work
            </Link>
            <Link
              href="/projects"
              className="text-muted-strong hover:text-accent transition-colors"
            >
              Projects
            </Link>
            <Link
              href="/muse"
              className="text-muted-strong hover:text-accent transition-colors"
            >
              Muse
            </Link>
            <Link
              href="/qr"
              className="text-muted-strong hover:text-accent transition-colors"
            >
              QR
            </Link>
          </div>
          <p className="mt-4 text-xs text-muted">
            &copy; {new Date().getFullYear()} Sameer Chaturvedi.
          </p>
        </div>
      </div>
    </footer>
  );
}
