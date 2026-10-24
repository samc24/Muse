import Link from "next/link";

const LINKS = [
  { href: "/work", label: "Work", short: "Work" },
  { href: "/muse", label: "Muse", short: "Muse" },
  { href: "/about", label: "About", short: "About" },
];

export function Nav() {
  return (
    <header className="sticky top-0 z-40 border-b hairline bg-background/70 backdrop-blur-xl">
      <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-baseline gap-2 group">
          <span className="font-semibold tracking-tight text-foreground group-hover:text-accent transition-colors">
            Sameer Chaturvedi
          </span>
          <span className="hidden sm:inline-block text-xs text-muted font-mono">
            / ML + software
          </span>
        </Link>

        <nav className="flex items-center gap-0.5 sm:gap-3 text-sm">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="text-muted hover:text-foreground transition-colors px-2 sm:px-3 py-1.5"
              aria-label={l.label}
            >
              <span className="sm:hidden">{l.short}</span>
              <span className="hidden sm:inline">{l.label}</span>
            </Link>
          ))}
          <a
            href="mailto:sam.chaturvedi24@gmail.com"
            className="btn btn-primary ml-1 sm:ml-2 !py-1.5 !px-3 sm:!px-4 !text-xs sm:!text-sm"
          >
            Contact
          </a>
        </nav>
      </div>
    </header>
  );
}
