import type { CaseStudy } from "./types";
import { amazonPrimeVideo } from "./work/amazon-prime-video";
import { mitPsfc } from "./work/mit-psfc";
import { cerebroSports } from "./work/cerebro-sports";
import { microsoft } from "./work/microsoft";
import { aiReverie } from "./work/ai-reverie";
import { redHat } from "./work/red-hat";
import { muse } from "./projects/muse";
import { explodingCatan } from "./projects/exploding-catan";

export type { CaseStudy, Section, StackGroup, ExternalLink, Accent } from "./types";

/** Reverse-chronological. Order here is the order everywhere on the site. */
export const WORK: CaseStudy[] = [
  amazonPrimeVideo,
  mitPsfc,
  cerebroSports,
  microsoft,
  aiReverie,
  redHat,
];

export const PROJECTS: CaseStudy[] = [muse, explodingCatan];

export const CURRENT = WORK.filter((w) => w.current);

export function hrefFor(entry: CaseStudy): string {
  if (entry.href) return entry.href;
  return `/${entry.kind === "work" ? "work" : "projects"}/${entry.slug}`;
}

export function findWork(slug: string): CaseStudy | undefined {
  return WORK.find((w) => w.slug === slug);
}

export function findProject(slug: string): CaseStudy | undefined {
  return PROJECTS.find((p) => p.slug === slug);
}

/** Entries that get their own generated page (those without an external href). */
export function pagedEntries(list: CaseStudy[]): CaseStudy[] {
  return list.filter((e) => !e.href);
}
