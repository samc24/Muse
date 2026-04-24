# design/

Architecture decisions and design docs for the Muse monorepo. One file per decision, `UPPERCASE_SUBJECT.md`. Each doc opens with a status line (Proposed / Approved / Superseded), the scope it covers, and the reasoning trail behind the decision.

## Current decisions

- [MOBILE_ARCHITECTURE.md](MOBILE_ARCHITECTURE.md) -- Follow Through mobile rewrite. Native twins (Kotlin Compose + SwiftUI, no KMP). Approved 2026-04-24.

## What lives here vs elsewhere

- `design/` -- architecture decisions, framework picks, major tradeoffs. Written before implementation; kept as the canonical reference.
- `CODEBASE_GUIDE.md` -- module-by-module tour of the repo as it stands. Descriptive, not prescriptive.
- `README.md` (repo root + subproject) -- public-facing subproject introductions.
- `CLAUDE.md` / `WORK_NOTES.md` -- gitignored personal working context.
