import Link from "next/link";
import { WORK, hrefFor } from "@/content";

/** Compact "worked at" strip. Derived from the WORK content registry. */
export function CredentialStrip() {
  return (
    <div className="relative">
      <p className="overline">Worked at</p>
      <div className="mt-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-x-8 gap-y-6">
        {WORK.map((c) => (
          <Link
            key={c.slug}
            href={hrefFor(c)}
            className="group flex flex-col border-l hairline-strong pl-4 hover:border-accent transition-colors"
          >
            <span className="text-sm font-semibold tracking-tight text-foreground group-hover:text-accent transition-colors">
              {c.name}
            </span>
            <span className="mt-1 text-xs text-muted leading-snug">
              {c.role}
            </span>
            <span className="mt-2 text-[10px] uppercase tracking-[0.15em] text-muted/70 font-mono">
              {c.period}
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}
