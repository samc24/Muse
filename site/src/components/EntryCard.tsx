import Link from "next/link";
import Image from "next/image";
import type { CaseStudy } from "@/content/types";
import { hrefFor } from "@/content";
import { accentVars } from "@/lib/accent";
import { posterFor } from "@/lib/media";
import { AutoplayVideo } from "./AutoplayVideo";

/**
 * Card for a role or project. Renders from a CaseStudy; links to its page.
 * With a posterVideo it becomes the taller media card used for projects.
 */
export function EntryCard({ entry }: { entry: CaseStudy }) {
  const href = hrefFor(entry);
  const media = Boolean(entry.posterVideo);

  return (
    <Link
      href={href}
      style={accentVars(entry.accent)}
      className="entry-card group relative flex flex-col rounded-2xl border overflow-hidden bg-surface-raised"
    >
      {media && (
        <div className="relative aspect-[16/9] w-full overflow-hidden bg-black">
          <AutoplayVideo
            src={entry.posterVideo!}
            poster={posterFor(entry.posterVideo!)}
            className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-surface-raised via-transparent to-transparent" />
        </div>
      )}

      <div className="p-6 sm:p-7 flex flex-col flex-1">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-4">
            {entry.logo && (
              <Image
                src={entry.logo}
                alt={entry.name}
                width={48}
                height={48}
                className="rounded-lg"
              />
            )}
            <div>
              <h3 className="text-xl sm:text-2xl font-semibold tracking-tight text-foreground leading-none">
                {entry.name}
              </h3>
              <p className="entry-accent mt-2 text-xs uppercase tracking-[0.2em] font-semibold">
                {entry.tagline}
              </p>
            </div>
          </div>
          <span className="text-[10px] font-mono uppercase tracking-[0.15em] text-muted/70 whitespace-nowrap pt-1">
            {entry.period}
          </span>
        </div>

        {entry.role && (
          <p className="mt-4 text-sm text-muted">
            {entry.role}
            {entry.team && <span className="text-muted/70"> / {entry.team}</span>}
          </p>
        )}

        <p className="mt-3 text-muted-strong leading-relaxed flex-1">
          {entry.summary}
        </p>

        <span className="mt-6 inline-flex items-center gap-2 text-sm text-muted group-hover:text-foreground transition-colors">
          {entry.href ? "Explore" : "Read more"}
          <span aria-hidden className="transition-transform group-hover:translate-x-1">
            -&gt;
          </span>
        </span>
      </div>
    </Link>
  );
}
