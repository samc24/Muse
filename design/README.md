# design/

Architecture decisions and design docs for the Muse monorepo. One file per decision, `UPPERCASE_SUBJECT.md`. Each doc opens with a status line (Proposed / Approved / Superseded), the scope it covers, and the reasoning trail behind the decision.

## Current decisions

- [MOBILE_ARCHITECTURE.md](MOBILE_ARCHITECTURE.md) -- Follow Through mobile rewrite. Native twins (Kotlin Compose + SwiftUI, no KMP). Approved 2026-04-24. Phasing: v1 iOS / v1.5 watchOS / v2 Android.
- [FT_IOS_V1_PLAN.md](FT_IOS_V1_PLAN.md) -- Concrete v1 implementation plan: bootstrap, screen inventory, 6-phase order, commit + test strategy. Draft 2026-04-25, working on `ft-ios`.
- [pose-model-evaluation-2026-04.md](pose-model-evaluation-2026-04.md) -- Pose-model selection. Production = MediaPipe Pose Landmarker, Heavy variant. Documents disqualified candidates + re-evaluation triggers.

## Architecture decision records (ADRs)

- [adr/0001-vendored-mediapipe-not-cocoapods.md](adr/0001-vendored-mediapipe-not-cocoapods.md) -- Why we vendor MediaPipe xcframeworks via Git LFS + local Swift Packages instead of using CocoaPods.

## What lives here vs elsewhere

- `design/` -- architecture decisions, framework picks, major tradeoffs. Written before implementation; kept as the canonical reference.
- `CODEBASE_GUIDE.md` -- module-by-module tour of the repo as it stands. Descriptive, not prescriptive.
- `README.md` (repo root + subproject) -- public-facing subproject introductions.
- `CLAUDE.md` / `WORK_NOTES.md` -- gitignored personal working context.
