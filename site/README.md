# sameerc.com -- live resume portfolio

Next.js 16 (App Router) + Tailwind v4 + Turbopack. Static-prerendered, zero server
dependencies, deploys as a pure edge/static site on Vercel.

A live resume: a generic home page, one case-study page per role under `/work`,
side projects under `/projects`, and Muse (computer vision for basketball) as its
own subtree at `/muse` with the Follow Through and EyeBall walkthroughs. URL is
meant to be sent cold.

## Quick start

```bash
cd site
npm install   # first time only
npm run dev   # opens http://localhost:3000 (or next free port)
```

## Production build (what Vercel runs)

```bash
npm run build
npm start
```

Every route prerenders to static HTML. `/work/[slug]` and `/projects/[slug]` are
generated from the content registry via `generateStaticParams`; unknown slugs 404
at build time (`dynamicParams = false`).

## Deploying to Vercel

1. Install the Vercel CLI: `npm i -g vercel` (one-time).
2. From `site/`, run `vercel` and follow the prompts. Vercel auto-detects Next.js.
3. For production: `vercel --prod`.

**Root directory setting:** when linking via the Vercel dashboard (or answering the
CLI prompt), the root directory is `site` (not the repo root). Everything above
`site/` is Python research code that Vercel should ignore.

**No environment variables required** for the current build. No API keys, no
database, no auth.

**Custom domain:** add in the Vercel dashboard after the first deploy; point a
CNAME at `cname.vercel-dns.com` from your DNS provider.

## Structure

```
site/
|-- public/
|   |-- branding/       # Muse / Follow Through / EyeBall logos
|   |-- photos/         # hero photos
|   `-- clips/          # MP4s (Shaq / Nash pose overlays, Jordan / Houston tracked, Broadcasting)
|-- src/
|   |-- content/                      # THE resume. One typed object per role / project.
|   |   |-- types.ts                  # CaseStudy, Section, StackGroup, ExternalLink, Accent
|   |   |-- index.ts                  # WORK + PROJECTS registries (order = display order), lookups
|   |   |-- work/<slug>.ts            # amazon-prime-video, mit-psfc, cerebro-sports, microsoft, ai-reverie, red-hat
|   |   `-- projects/<slug>.ts        # muse (links out to /muse instead of generating a page)
|   |-- lib/accent.ts                 # per-entry accent colours as CSS custom properties
|   |-- app/
|   |   |-- layout.tsx                # root layout: Nav + Footer + fonts + site metadata
|   |   |-- page.tsx                  # home: hero, "now", worked-at strip, work grid, projects grid, CTA
|   |   |-- work/page.tsx             # index of all roles
|   |   |-- work/[slug]/page.tsx      # one page per role, rendered by CaseStudyPage
|   |   |-- projects/page.tsx         # index of side projects
|   |   |-- projects/[slug]/page.tsx  # one page per project (none yet besides Muse)
|   |   |-- muse/page.tsx             # Muse hub: thesis, Follow Through + EyeBall cards
|   |   |-- follow-through/page.tsx   # pose analysis + similarity UX
|   |   |-- eyeball/page.tsx          # ball tracking + broadcasting
|   |   |-- about/page.tsx            # bio, purpose, timeline, education, skills
|   |   |-- sitemap.ts                # fixed routes + generated entry routes
|   |   `-- globals.css               # Tailwind v4 theme: dark palette + accents
|   `-- components/
|       |-- Nav.tsx                   # sticky top nav: Work / Muse / About / Contact
|       |-- Footer.tsx
|       |-- EntryCard.tsx             # card for a role or project (from CaseStudy)
|       |-- CaseStudyPage.tsx         # the single template behind /work/* and /projects/*
|       |-- CredentialStrip.tsx       # "worked at" strip, derived from WORK
|       |-- Timeline.tsx              # resume timeline on /about, derived from WORK
|       |-- ProjectCard.tsx           # Follow Through / EyeBall cards on /muse
|       |-- VideoClip.tsx             # reusable <video> + caption
|       `-- SectionHeader.tsx         # overline / title / lede pattern
```

## Adding a role or project

1. Create `src/content/work/<slug>.ts` (or `projects/`) exporting a `CaseStudy`.
2. Add it to `WORK` or `PROJECTS` in `src/content/index.ts` at the right position
   (arrays are reverse-chronological and drive every list on the site).
3. `npm run build`. The page, its card, the timeline entry, the credential strip,
   and the sitemap entry all follow from the registry.

Set `placeholder: true` while `sections` is empty; the page shows a short
"write-up in progress" note. Every bullet must trace to the resume or real work.

## Design notes

- Dark-by-default theme. Site accent is orange (`#f97316`); Muse keeps it as its
  brand colour, Follow Through uses red + blue. Each role / project carries its
  own `accent` (see `src/lib/accent.ts`) applied through `--entry-accent`.
- Fonts: Geist Sans + Geist Mono via `next/font/google`.
- All video clips lazy-load (`preload="metadata"`), controls visible, muted by
  default (autoplay-policy-compliant).
- Video files are committed to `public/clips/` (~18 MB total). If the bundle
  grows, move to Vercel Blob or Cloudflare R2 and update the `src` paths.

## What this site is NOT

- Not a running Streamlit demo (that lives at `../demo/`).
- Not interactive analysis. All numbers and visuals shown here are pre-baked
  outputs from the Streamlit demo's pure-function layer.
- Not a blog. Case studies are structured data rendered by one template, not
  free-form posts.

## Editing copy

Resume content (roles, bullets, stacks, links) lives in `src/content/`. Page-level
prose (home hero, about, Muse hub, Follow Through and EyeBall walkthroughs) lives
inline in the respective `src/app/**/page.tsx`. No CMS, no MDX.
