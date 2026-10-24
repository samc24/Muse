import Link from "next/link";
import { WORK, hrefFor } from "@/content";

/** Resume timeline. Derived from the WORK content registry; each entry links to its page. */
export function Timeline() {
  return (
    <ol className="relative">
      {WORK.map((item) => (
        <li
          key={item.slug}
          className="relative pl-8 pb-10 last:pb-0 border-l hairline-strong ml-3"
        >
          <span className="absolute -left-[7px] top-1 w-3 h-3 rounded-full bg-accent ring-4 ring-background" />
          <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-1">
            <div>
              <span className="text-lg font-semibold tracking-tight text-foreground">
                {item.role}
              </span>
              <span className="text-muted"> at </span>
              <Link
                href={hrefFor(item)}
                className="text-lg font-semibold tracking-tight text-foreground hover:text-accent transition-colors underline decoration-border-strong underline-offset-4 hover:decoration-accent"
              >
                {item.name}
              </Link>
            </div>
            <span className="text-xs font-mono uppercase tracking-[0.15em] text-muted whitespace-nowrap">
              {item.period}
            </span>
          </div>
          {item.team && <p className="mt-1 text-sm text-muted">{item.team}</p>}
          <ul className="mt-4 space-y-2">
            {item.highlights.map((p) => (
              <li
                key={p}
                className="text-muted-strong leading-relaxed text-[0.975rem] pl-4 relative before:content-['-'] before:absolute before:left-0 before:text-muted"
              >
                {p}
              </li>
            ))}
          </ul>
          <Link
            href={hrefFor(item)}
            className="mt-3 inline-flex items-center gap-2 text-sm text-muted hover:text-foreground transition-colors"
          >
            More on {item.name}
            <span aria-hidden>-&gt;</span>
          </Link>
        </li>
      ))}
    </ol>
  );
}
