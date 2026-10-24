/**
 * Content model for the live-resume portfolio.
 *
 * Every role and side project is one typed object. Pages render from these
 * objects through a single template, so adding a role or project is a content
 * edit under src/content/, not a page build.
 */

export type Accent =
  | "orange"
  | "blue"
  | "red"
  | "green"
  | "violet"
  | "sky"
  | "neutral";

export type Section = {
  /** Small uppercase label above the heading. */
  overline?: string;
  heading: string;
  paragraphs?: string[];
  bullets?: string[];
};

export type StackGroup = {
  label: string;
  items: string[];
};

export type ExternalLink = {
  label: string;
  href: string;
};

export type CaseStudy = {
  /** URL segment under /work or /projects. */
  slug: string;
  kind: "work" | "project";
  /** Short display name used on cards and in nav (e.g. "Amazon"). */
  name: string;
  /** Fuller name for the page hero (e.g. "Amazon Prime Video"). */
  org: string;
  role?: string;
  team?: string;
  period: string;
  /** One line under the name on cards. */
  tagline: string;
  /** One or two sentences for cards and the page hero. */
  summary: string;
  accent: Accent;
  current?: boolean;
  /** Resume-grounded bullets. Rendered as "What I did". */
  highlights: string[];
  /** Long-form sections. Empty until source material is added. */
  sections: Section[];
  stack: StackGroup[];
  links: ExternalLink[];
  /** Optional logo under /public. Text wordmark is used when absent. */
  logo?: string;
  /** Optional poster video under /public for the card. */
  posterVideo?: string;
  /**
   * When set, cards link here instead of /work/<slug> or /projects/<slug>,
   * and no case-study page is generated. Used for Muse, which has its own
   * subtree at /muse.
   */
  href?: string;
  /** Renders a short "write-up in progress" note on the page. */
  placeholder?: boolean;
};
