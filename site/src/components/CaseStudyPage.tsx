import Link from "next/link";
import type { CaseStudy } from "@/content/types";
import { hrefFor } from "@/content";
import { accentVars } from "@/lib/accent";

type Props = {
  entry: CaseStudy;
  /** Siblings in display order, used for prev/next navigation. */
  siblings: CaseStudy[];
};

/** Single template for every /work and /projects page. */
export function CaseStudyPage({ entry, siblings }: Props) {
  const idx = siblings.findIndex((s) => s.slug === entry.slug);
  const prev = idx > 0 ? siblings[idx - 1] : undefined;
  const next = idx >= 0 && idx < siblings.length - 1 ? siblings[idx + 1] : undefined;
  const indexHref = entry.kind === "work" ? "/work" : "/projects";
  const indexLabel = entry.kind === "work" ? "Work" : "Projects";

  return (
    <div style={accentVars(entry.accent)}>
      {/* Hero */}
      <section className="relative overflow-hidden border-b hairline">
        <div className="entry-aura absolute inset-0 pointer-events-none" aria-hidden />
        <div className="relative mx-auto max-w-6xl px-6 pt-16 pb-16 sm:pt-20">
          <Link
            href={indexHref}
            className="inline-flex items-center gap-2 text-sm text-muted hover:text-foreground transition-colors"
          >
            <span aria-hidden>&lt;-</span> {indexLabel}
          </Link>

          <p className="entry-accent mt-8 overline">{entry.tagline}</p>
          <h1 className="mt-4 h-display max-w-4xl">{entry.org}</h1>

          <dl className="mt-8 flex flex-wrap gap-x-10 gap-y-4 text-sm">
            {entry.role && (
              <div>
                <dt className="overline">Role</dt>
                <dd className="mt-1 text-foreground">{entry.role}</dd>
              </div>
            )}
            {entry.team && (
              <div>
                <dt className="overline">Team</dt>
                <dd className="mt-1 text-foreground">{entry.team}</dd>
              </div>
            )}
            <div>
              <dt className="overline">When</dt>
              <dd className="mt-1 text-foreground font-mono text-xs uppercase tracking-[0.15em] pt-1">
                {entry.period}
              </dd>
            </div>
          </dl>

          <p className="mt-8 text-lg sm:text-xl text-muted-strong max-w-3xl leading-relaxed">
            {entry.summary}
          </p>

          {entry.links.length > 0 && (
            <div className="mt-8 flex flex-wrap gap-3">
              {entry.links.map((l) => (
                <a
                  key={l.href}
                  href={l.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-secondary"
                >
                  {l.label}
                </a>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Highlights */}
      {entry.highlights.length > 0 && (
        <section className="mx-auto max-w-4xl px-6 py-16">
          <p className="overline">What I did</p>
          <ul className="mt-6 space-y-4">
            {entry.highlights.map((h) => (
              <li
                key={h}
                className="text-muted-strong leading-relaxed text-lg pl-5 relative before:content-['-'] before:absolute before:left-0 before:text-muted"
              >
                {h}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Long-form sections */}
      {entry.sections.map((s, i) => (
        <section
          key={s.heading}
          className={`mx-auto max-w-4xl px-6 py-16 ${i === 0 ? "border-t hairline" : ""}`}
        >
          {s.overline && <p className="entry-accent overline">{s.overline}</p>}
          <h2 className="mt-4 text-2xl sm:text-3xl font-semibold tracking-tight max-w-3xl">
            {s.heading}
          </h2>
          {s.paragraphs && (
            <div className="mt-6 prose-muse max-w-3xl">
              {s.paragraphs.map((p) => (
                <p key={p}>{p}</p>
              ))}
            </div>
          )}
          {s.bullets && (
            <ul className="mt-6 space-y-3 max-w-3xl">
              {s.bullets.map((b) => (
                <li
                  key={b}
                  className="text-muted-strong leading-relaxed pl-4 relative before:content-['-'] before:absolute before:left-0 before:text-muted"
                >
                  {b}
                </li>
              ))}
            </ul>
          )}
        </section>
      ))}

      {/* Placeholder note */}
      {entry.placeholder && (
        <section className="mx-auto max-w-4xl px-6 pb-4">
          <p className="text-sm text-muted">
            Full write-up in progress. The bullets above are the short version.
          </p>
        </section>
      )}

      {/* Stack */}
      {entry.stack.length > 0 && (
        <section className="mx-auto max-w-4xl px-6 py-16 border-t hairline">
          <p className="overline">Under the hood</p>
          <h2 className="mt-4 text-2xl font-semibold tracking-tight">Stack.</h2>
          <div className="mt-8 grid sm:grid-cols-2 gap-x-10 gap-y-8">
            {entry.stack.map((g) => (
              <div key={g.label}>
                <p className="overline">{g.label}</p>
                <ul className="mt-4 space-y-2">
                  {g.items.map((it) => (
                    <li
                      key={it}
                      className="text-muted-strong pl-4 relative before:content-['-'] before:absolute before:left-0 before:text-muted"
                    >
                      {it}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Prev / next */}
      <section className="mx-auto max-w-4xl px-6 py-12 border-t hairline">
        <div className="flex flex-col sm:flex-row justify-between gap-4 text-sm">
          {prev ? (
            <Link href={hrefFor(prev)} className="text-muted hover:text-foreground transition-colors">
              <span aria-hidden>&lt;-</span> {prev.name}
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link href={hrefFor(next)} className="text-muted hover:text-foreground transition-colors sm:text-right">
              {next.name} <span aria-hidden>-&gt;</span>
            </Link>
          ) : (
            <span />
          )}
        </div>
      </section>
    </div>
  );
}
