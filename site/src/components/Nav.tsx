import Link from "next/link";
import Image from "next/image";

export function Nav() {
  return (
    <header className="sticky top-0 z-40 border-b hairline bg-background/70 backdrop-blur-xl">
      <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5 group">
          <Image
            src="/branding/muse.png"
            alt="Muse"
            width={26}
            height={26}
            className="rounded-sm"
            priority
          />
          <span className="font-semibold tracking-tight text-foreground group-hover:text-accent transition-colors">
            Muse
          </span>
          <span className="hidden sm:inline-block ml-2 text-xs text-muted font-mono">
            / Sameer Chaturvedi
          </span>
        </Link>

        <nav className="flex items-center gap-0.5 sm:gap-3 text-sm">
          <Link
            href="/follow-through"
            className="text-muted hover:text-foreground transition-colors px-2 sm:px-3 py-1.5"
            aria-label="Follow Through"
          >
            <span className="sm:hidden">FT</span>
            <span className="hidden sm:inline">Follow Through</span>
          </Link>
          <Link
            href="/eyeball"
            className="text-muted hover:text-foreground transition-colors px-2 sm:px-3 py-1.5"
            aria-label="EyeBall"
          >
            <span className="sm:hidden">EB</span>
            <span className="hidden sm:inline">EyeBall</span>
          </Link>
          <Link
            href="/about"
            className="text-muted hover:text-foreground transition-colors px-2 sm:px-3 py-1.5"
          >
            About
          </Link>
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
