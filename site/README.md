# Muse -- portfolio site

Next.js 16 (App Router) + Tailwind v4 + Turbopack. Static-prerendered, zero server
dependencies, deploys as a pure edge/static site on Vercel.

This is the public-facing version of Muse: bio, Follow Through walkthrough, EyeBall
walkthrough (with Broadcasting), and contact. Target audience is basketball-tech
professionals; URL is meant to be sent cold.

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

All routes (`/`, `/about`, `/eyeball`, `/follow-through`) prerender to static HTML.

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
|   |-- branding/       # Muse / Follow Through / EyeBall logos + hero image
|   `-- clips/          # MP4s (Shaq / Nash pose overlays, Jordan / Houston tracked, Broadcasting)
|-- src/
|   |-- app/
|   |   |-- layout.tsx                # root layout: Nav + Footer + fonts
|   |   |-- page.tsx                  # home: hero, bio strip, project cards, CTA
|   |   |-- about/page.tsx            # long-form about + contact
|   |   |-- eyeball/page.tsx          # ball tracking + broadcasting
|   |   |-- follow-through/page.tsx   # pose analysis + similarity UX
|   |   `-- globals.css               # Tailwind v4 theme: dark palette + Muse accents
|   `-- components/
|       |-- Nav.tsx                   # sticky top nav with Contact CTA
|       |-- Footer.tsx
|       |-- VideoClip.tsx             # reusable <video> + caption
|       |-- SectionHeader.tsx         # overline / title / lede pattern
|       `-- ProjectCard.tsx           # home project cards
```

## Design notes

- Dark-by-default theme. Accent: basketball orange (`#f97316`) for EyeBall / Muse
  wordmark, red + blue (`#e11d48`, `#2563eb`) for Follow Through when tagged.
- Fonts: Geist Sans + Geist Mono via `next/font/google`.
- All video clips lazy-load (`preload="metadata"`), controls visible, muted by
  default (autoplay-policy-compliant).
- Video files are committed to `public/clips/` (~18 MB total). If the bundle
  grows, move to Vercel Blob or Cloudflare R2 and update the `src` paths.

## What this site is NOT

- Not a running Streamlit demo (that lives at `../demo/`).
- Not interactive analysis. All numbers and visuals shown here are pre-baked
  outputs from the Streamlit demo's pure-function layer.
- Not a blog or case-study platform. It's a self-contained portfolio summary.

## Editing copy

All copy lives inline in the `src/app/**/page.tsx` files. No CMS, no MDX yet. To
change the bio, edit `src/app/page.tsx` (bio strip) and `src/app/about/page.tsx`.
To change project walkthroughs, edit the respective `page.tsx` files.
