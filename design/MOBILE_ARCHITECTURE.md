# Mobile App Architecture Decision -- Follow Through (2026)

**Status:** Approved 2026-04-24. v1 iOS perf sanity check pending before implementation.
**Scope:** Architecture for the Follow Through mobile rewrite. Phased ship: **v1 iOS phone -> v1.5 watchOS companion -> v2 Android phone.**
**Decision owner:** Sameer (solo).

## Decision

**Native twins, shipped iOS-first.** SwiftUI on iOS in v1; SwiftUI watchOS target in v1.5; Jetpack Compose on Android in v2. Two separate codebases when both exist, no KMP shared layer. When Android lands, analysis logic gets ported via a `mobile/SHARED_SPEC.md` contract + shared JSON test fixtures in `mobile/shared/fixtures/`; both implementations assert the same outputs within float tolerance. The shared-spec mechanism does not need to exist until v2 -- v1 writes `Analysis.swift` directly against `demo/analysis.py`'s contract; the spec gets extracted retroactively when Android starts.

watchOS in v1.5 lives inside the iOS Xcode project as a watchOS app target, sharing a local Swift Package with the iOS phone app. WatchConnectivity carries phone-derived coaching events to the watch; the watch independently samples CoreMotion for parallel wrist-mechanics signal.

## Phasing

| Phase | Scope | Approx. timing | Rationale |
|---|---|---|---|
| v1 | iOS phone app: pro browser, recording, Compare, similarity score + descriptor + advice, per-joint chart, wrist trajectory chart, history. MediaPipe Tasks iOS for on-device pose. | First | Largest validation surface for the product UX. iOS skews to the basketball-tech professional audience. Single-platform velocity during the highest-iteration phase. |
| v1.5 | watchOS companion: haptic live feedback during drills (phone -> WatchConnectivity -> Core Haptics), parallel wrist-IMU signal (CoreMotion). | After v1 | Same Swift ecosystem -- minimal context switch. Locks in the watch architectural payoff before any Kotlin work. If watch ever proves untenable, v2 plans get reconsidered before sunk cost grows. |
| v2 | Android port: Jetpack Compose UI, MediaPipe Tasks Android, port `Analysis.swift` -> `Analysis.kt` with shared spec + fixtures. SQLDelight data layer mirroring SwiftData schema. | After v1.5 | Translating a validated iOS UX + analysis spec to Compose. AI translation is at its strongest when the source spec is concrete and tested. |
| v3+ | Combine-forms, custom-mechanics LLM, overlay export, music-synced drills, expanded on-device ML. | TBD | See Future phase compatibility section. |

The "two of everything" lifetime cost only kicks in at v2. v1 + v1.5 are a single-platform iOS family ship.

## Why this, in one paragraph

Solo development with AI-assisted coding is the dominant velocity constraint. The chosen architecture has to optimise for what AI writes most fluently, and against what AI gets stuck on. Plain Kotlin Compose and plain SwiftUI are AI's strongest mobile outputs; KMP's glue layer (`expect/actual`, cinterop for MediaPipe iOS, Skie-generated Swift APIs, Gradle multiplatform plugin config, iOS framework export) is its weakest mobile output -- corpus of public KMP-Mobile projects with non-trivial shared code is roughly 1/100 the size of the Flutter / React Native corpus. Meanwhile AI-assisted code also devalues the main reason to have a shared layer: translating ~200 LoC of pure analysis math from Kotlin to Swift is a one-prompt task, and test fixtures make parity drift mechanically detectable. Native twins put all the code in AI's strongest spots (idiomatic Compose, idiomatic SwiftUI, first-party MediaPipe Tasks SDKs on each platform) and pay a duplication cost that AI makes cheap. watchOS is clean on native; Flutter and React Native don't have a viable watchOS path. Compose Multiplatform has plugin gaps on iOS for camera + video + audio + ML. KMP's shared-analysis win is real but small (most lifetime LoC is platform-specific UI, audio, haptics, and video-composition) and gets paid for with glue-layer overhead on every new feature.

## Context

The legacy Follow Through mobile app is a Java Android app (`~/Documents/Coding/FollowThrough/`, compileSdk 27, support-v7, ObjectBox) paired with a Django + OpenPose C++ server (`FollowThrough/FTTrial2.0/mysite/`, originally Heroku-hosted). The client uploaded user video; the server returned an MJPEG AVI with skeleton overlay. The Java app is frozen at `android.support` XML namespaces that will not build on modern toolchains. The Heroku server is retired. There is no iOS build.

The Muse Streamlit demo (`demo/analysis.py`) reduced the analysis tier to seven pure functions with zero Streamlit / OpenCV dependencies: `normalize_keypoints`, `per_joint_deviation`, `similarity_score`, `similarity_descriptor`, `generate_advice`, `wrist_trajectories`, `trajectory_figure`. Ten unit tests pin the public contract (`demo/tests/test_analysis.py`). These are the port target.

**Decision required:** pick the mobile architecture that preserves the legacy feature set, adds the demo's quantified analytics UX, ships iOS first then watchOS then Android, retires the server, remains maintainable by one engineer using AI-assisted coding, and leaves a clean path for a watchOS companion in v1.5.

## Goals

1. Ship to the iOS App Store first (v1), then watchOS companion (v1.5), then Android Play Store (v2). Single project root (`Muse/mobile/`) once Android lands.
2. Preserve every legacy feature worth keeping: pro browser + search + favourites, side-by-side compare, in-app recording.
3. Add the demo's Phase-1 UX: similarity score, descriptor, top-N coaching advice, per-joint deviation chart, wrist-trajectory chart.
4. On-device pose via MediaPipe Tasks using the first-party SDKs (MediaPipe Tasks iOS for v1, MediaPipe Tasks Android for v2). No upload-to-server in the critical path.
5. Maintainable solo + AI-assisted. Architecture must keep code in AI's highest-fluency mobile patterns (idiomatic SwiftUI, idiomatic Compose, native SDKs) and out of its lowest-fluency mobile patterns (cross-language cinterop, multi-platform Gradle, generated Swift APIs).
6. watchOS v1.5 companion: haptic + glance live feedback during recording drills, parallel wrist-IMU signal, clean integration path. Lands close on the heels of v1 to lock in the watch architectural payoff before any Kotlin work begins.

## Non-goals

1. Not replacing the Muse Python CV research code. EyeBall, FollowThrough research scripts, Streamlit demo stay.
2. Not shipping a web version. The portfolio `site/` and Streamlit `demo/` already cover that.
3. Not keeping the Django + OpenPose backend. Retire `FTTrial2.0/` once mobile v1 ships.
4. Not uploading user video to a server. Keypoint JSONs only, and only if a template-library backend lands later.

## Constraints

1. **Solo developer, AI-assisted.** Velocity comes from what AI writes fluently. AI-fluency of the stack is a primary decision criterion, not an ergonomic nice-to-have.
2. **Heavy shared domain logic, small UI surface.** Analysis (~200 LoC of math) is non-trivial and tested. UI is ~7 screens of record / list / compare / results.
3. **On-device ML is table stakes.** Architecture must have first-class on-device pose story on both platforms via the first-party MediaPipe Tasks SDKs.
4. **watchOS companion lands in v1.5, before Android.** Phone-to-watch feedback relay during drills + watch-IMU parallel signal. Architectures that cannot cleanly reach watchOS are disqualified.
5. **Muse monorepo.** Project lives at `Muse/mobile/`. Single repo, shared CLAUDE.md and WORK_NOTES.

## Feature inventory to preserve (drives the comparison)

| Capability | Source | Notes |
|---|---|---|
| Splash / logo animation | `Startup.java` | Trivial on any framework. |
| Pro browser with search + favourites | `HomePage.java`, `PlayerAdapter.java` | 15 hardcoded pros seeded from `Startup.java`. |
| Side-by-side compare with independent speed (0.25x-2.0x) + trim | `UserVPro.java` | Legacy has no sync between panes; new build should fix. |
| In-app recording (landscape) | `RecordUser.java` (commented out in legacy) | Legacy punted to external Camera intent. New build runs in-app. |
| On-device pose extraction | New (replaces Django + OpenPose) | MediaPipe Tasks native SDKs on both platforms. |
| Similarity score + descriptor + top-N advice | `demo/analysis.py` | Seven pure functions, ten tests, direct port. |
| Per-joint deviation chart | `demo/analysis.py` + Streamlit rendering | Port data, native chart rendering (Swift Charts, Compose charts). |
| Wrist-trajectory chart | `demo/analysis.py` + Streamlit rendering | Same. |
| Shot history (per user) | New | SQLDelight (Android), SwiftData (iOS). |
| Watch v1.5: haptic live feedback during drill | New | Phone pose -> WatchConnectivity -> Core Haptics. |
| Watch v1.5: parallel wrist IMU signal | New | CoreMotion on watchOS, release detection + wrist snap angle. |
| Template library refresh (optional, later) | New | Thin backend deferred past v1. |

## Candidate architectures

Each candidate evaluated against the constraints above. All assume on-device MediaPipe Tasks for pose.

### E) Native twins -- Kotlin Compose + SwiftUI (CHOSEN)

- Android UI: Jetpack Compose + CameraX + Media3 (ExoPlayer).
- iOS UI: SwiftUI + AVFoundation.
- watchOS: SwiftUI watchOS target inside the iOS Xcode project, sharing a Swift Package with the iOS phone app.
- Analysis: implemented twice (Kotlin + Swift), parity enforced by shared JSON test fixtures + shared spec doc.
- Data: SQLDelight on Android, SwiftData on iOS; schemas mirror.
- No KMP, no Skie, no cinterop.

### A) KMP shared + native UI per OS (runner-up)

- Shared (Kotlin): analysis functions, SQLDelight data layer, Ktor networking, `expect/actual` `PoseExtractor` wrapping MediaPipe Tasks.
- Android UI: Compose + CameraX + Media3.
- iOS UI: SwiftUI + AVFoundation + Skie-generated Swift APIs.
- watchOS: same pattern (SwiftUI target consuming the shared KMP framework).
- UI duplication: yes, each screen written twice.
- Shared-layer duplication: zero.

### B) Flutter

- Shared (Dart): UI + analysis + data + platform plugins.
- Pose: `google_mlkit_pose_detection` or MediaPipe via FFI.
- watchOS: not viable. FlutterEngine does not target watchOS. Any watch companion is a standalone native Swift app with WatchConnectivity, which re-introduces the duplication Flutter is meant to avoid.

### D) React Native + Expo

- Shared (TypeScript): UI + analysis + plugin wrappers.
- Pose: VisionCamera + MediaPipe frame processor.
- watchOS: `react-native-watch` is essentially unmaintained. Same answer as Flutter: watch companion is standalone native Swift.

### C) Compose Multiplatform

- Shared (Kotlin): everything, including UI.
- Camera / video / audio / MediaPipe on iOS: handled via platform interop; plugin maturity lags on iOS in 2026.
- Inherits KMP's glue-layer AI-fluency problem without the payoff of first-party UI on each platform.

## Evaluation criteria

| Criterion | Weight | Why |
|---|---|---|
| AI-assisted coding fluency on the stack's primitives | High | Primary velocity lever for this project. AI writes idiomatic Compose + SwiftUI fluently; writes KMP glue, Skie-generated surfaces, and cinterop more slowly and with more errors. |
| On-device ML + camera + audio + haptic + video-composition plugin maturity | High | Core loop and future phases depend on it. Native SDKs are first-party; Flutter/RN wrappers lag 6-12 months. |
| watchOS companion feasibility | High | v2 scope. Flutter / RN disqualify themselves here. |
| Shared-logic leverage | Medium (down from High) | AI-assisted translation makes dual-impl cheap; test-fixture parity is mechanical. The classical "single source of truth" win matters less when translation is a prompt. |
| Per-screen UI maintenance cost | Medium | AI-assisted coding cuts the classical 2024-era 1.8-2.0x multiplier down to ~1.3-1.5x, shrinks this criterion's weight. Still real. |
| Platform-deep capability later (Widgets, Live Activities, Health, App Intents) | Low-Medium | Not v1 or v2, but architecture shouldn't foreclose. |
| Analysis port effort from Python | Low | ~200 LoC of typed math; trivial to any typed language. Wash. |

## Comparison matrix

| | E) Native twins | A) KMP + native UI | B) Flutter | D) RN + Expo | C) CMP |
|---|---|---|---|---|---|
| AI fluency on UI layer | **High both legs** | High both legs | High | High | Medium (Compose-on-iOS less trodden) |
| AI fluency on non-UI shared layer | N/A (no shared layer) | **Low (KMP glue)** | High (Dart) | **Highest (TS)** | Low (KMP glue) |
| MediaPipe integration | Native SDKs directly | Native SDKs via `expect/actual` cinterop | Plugin wrappers | Plugin wrappers | Plugin gaps on iOS |
| Camera pipeline | Native (CameraX / AVFoundation) | Native | Plugin (mature) | Plugin (VisionCamera, mature) | Plugin (uneven) |
| Two-synced-video compare | Native (ExoPlayer / AVPlayer, paved) | Native | Plugin friction on frame-sync | Plugin friction on layering | Plugin gaps |
| Low-latency audio + haptics (future: music-synced drills) | Native (AVAudioEngine / Oboe / Core Haptics) | Native | Plugin territory, historical weakness | Plugin territory, historical weakness | Plugin gaps |
| watchOS companion | **Clean** (SwiftUI target + Swift Package) | **Clean** (SwiftUI target + KMP framework consumed) | **Not viable** (no FlutterEngine) | **Not viable** (react-native-watch unmaintained) | Unclear / experimental |
| Shared logic across platforms | Dual-impl with spec + fixtures | Single impl in commonMain | Single Dart impl | Single TS impl | Single Kotlin impl |
| Build toolchain | Gradle + Xcode (two clean) | Gradle multiplatform + Xcode (interleaved) | Flutter CLI | Metro / Expo | Gradle multiplatform + Xcode |
| Platform-deep later | Trivial on both | Easy on both | Hard / native module per | Hard / native module per | Moderate |
| Per-screen UI cost (AI-assist era) | ~1.3-1.5x | ~1.5x (UI twice + KMP glue per screen) | 1.0x | 1.0x | 1.0x |

## UI cost under AI-assist, honest (2026 numbers)

Classical multi-UI penalty was a pre-AI argument. Updated numbers for this project:

| Screen type | 2024 classical multiplier | 2026 AI-assist multiplier |
|---|---|---|
| Simple list / detail / form | 2.0x | 1.2-1.3x |
| Camera + per-frame processing | 2.0x | 1.4-1.5x |
| Two-synced-video compare | 2.0x | 1.4-1.5x |
| Charts + results | 2.0x | 1.3x |
| Settings / history / about | 2.0x | 1.1-1.2x |

Lifetime UI duplication under AI-assist for this app's ~7 screens: estimated ~600-900 LoC per platform, compared with the classical ~1500 LoC. What AI-assist does *not* remove: behavior parity drift under bug fixes, testing matrix (two simulators, two store submissions, two crash dashboards), cognitive overhead of holding both platforms when redesigning. All real, none dealbreakers.

## Why E over A (the close call)

Both are native-UI. Both handle watchOS cleanly. Both use first-party MediaPipe Tasks SDKs. The question reduces to: does KMP's shared-analysis win outweigh its glue-layer AI-fluency tax?

Case for A:
- One Kotlin implementation of analysis math. Test once. No parity drift.
- Shared data layer (SQLDelight), shared networking (Ktor), shared session/history code. Compounds as future phases add more analysis math (combine-forms blend math, custom-mechanics prompt-building).
- Over 5 future phases, A potentially shares ~1500-2000 LoC of math + data + networking code.

Case for E (wins):
- KMP glue layer (`expect/actual`, cinterop for MediaPipe iOS, Skie, Gradle multiplatform plugin, iOS framework export) is AI's weakest mobile pattern. Every new platform API added to the shared layer adds another `expect/actual` pair. 5 future phases could mean 5 new glue surfaces in AI's lowest-fluency territory.
- Audio, haptics, video composition, per-platform camera quirks are platform-specific in **both** A and E. They live in `androidMain`/`iosMain` regardless. So A's "shared layer" does not help on the highest-value / most-complex parts of future phases.
- AI-assisted dual-impl + shared JSON fixtures + shared spec doc gets you ~95% of A's parity win mechanically. `Analysis.kt` and `Analysis.swift` load the same `mobile/shared/fixtures/*.json` inputs, compute, and assert the same outputs within float tolerance. Drift is caught on the next CI run.
- The compounding glue tax in A roughly cancels the compounding shared-code win from 5 future phases, leaving E slightly ahead on velocity.
- Spike 1 risk (MediaPipe Tasks iOS via KMP cinterop) disappears entirely in E. Highest-risk assumption in the 2026-04-22 draft becomes a non-issue.

The margin is narrow. A remains the runner-up. But under the "optimise for AI-assisted coding" criterion -- the criterion most specific to this project -- E wins.

## Why not Flutter (B)

- watchOS v1.5 is not viable. Any watch feature becomes a standalone Swift target, which defeats Flutter's single-codebase argument precisely for the feature most likely to differentiate the product.
- Low-latency audio + haptics for music-synced drills (future phase) is native-only territory in practice; Flutter plugins exist but are not used in shipping rhythm/music apps.
- MediaPipe Tasks consumed via plugin wrappers, which lag the first-party SDKs.
- Two-synced-video compare with frame-accurate overlay has known friction in `video_player`.
- AI-fluency win on single UI codebase is real but smaller than it sounds: AI-assisted dual-UI in E is ~1.3-1.5x per screen, not 2.0x.

## Why not React Native + Expo (D)

- Same watchOS problem as Flutter. `react-native-watch` is unmaintained; watch companion is a standalone native Swift target.
- AI fluency on TypeScript is the highest of any stack in this comparison. If watchOS were off the table, D would compete strongly on velocity alone. watchOS being on the roadmap is what disqualifies it.
- VisionCamera + react-native-mediapipe is the most mature cross-platform on-device pose path in 2026, but still a bridge boundary -- native SDKs run in-process.

## Why not Compose Multiplatform (C)

- Inherits KMP's glue-layer AI-fluency tax.
- Plugin gaps on iOS for camera, video, audio, on-device ML.
- No offset win, because the UI layer on iOS is Compose rendered into a UIView, not native SwiftUI -- platform-deep iOS integration (watch, widgets, Live Activities) becomes awkward, not easier.

## Consequences of choosing E (phase-tagged)

### v1 (iOS phone)

- **Repo layout starts:** `Muse/mobile/ios/` (Xcode project, single iOS target). `Muse/mobile/android/` and `Muse/mobile/shared/` do not yet exist.
- **Seed data:** the 15 hardcoded pros from `Startup.java` become `mobile/ios/FollowThrough/Resources/pros_seed.json`, loaded on first launch.
- **Analysis port:** `demo/analysis.py` -> `mobile/ios/FollowThrough/Analysis.swift`. Ten pytest tests -> ten XCTest tests loading fixtures from `mobile/ios/FollowThroughTests/fixtures/*.json`. Float tolerance pinned. The shared-spec doc does not exist yet -- the contract lives in the Swift code + tests.
- **Pose pipeline:** MediaPipe Tasks iOS SDK consumed directly via Swift Package Manager.
- **Data layer:** SwiftData (or GRDB if SwiftData proves limiting for the history queries you actually need).
- **Backend:** none. Bundled templates only.
- **CI:** single iOS job (`macos-latest`, Xcode). Python jobs untouched.
- **Server retirement:** `FTTrial2.0/` archived; deletion deferred to v2 (after Android lands and `WORK_NOTES.md` is updated).

### v1.5 (watchOS companion)

- **watchOS app target** added to the existing `Muse/mobile/ios/` Xcode project.
- **Local Swift Package** extracted from the iOS app: shared models, analysis contract, design tokens. Both iOS phone app and watchOS app consume the package.
- **WatchConnectivity** for phone-derived feedback events (latency budget: 150ms end-to-end).
- **CoreMotion on watchOS** for parallel wrist-IMU signal -- release detection + wrist snap angle.
- **No Android impact** -- Android does not exist yet.

### v2 (Android port)

- **Repo layout extends:** `Muse/mobile/android/` (Gradle project, Compose app) added. `Muse/mobile/shared/` (SHARED_SPEC.md, JSON fixtures, design tokens) extracted from the iOS implementation by reading the Swift code and tests, then ported.
- **Analysis port:** `Analysis.swift` -> `mobile/android/app/src/main/kotlin/analysis/Analysis.kt`. Ten XCTest tests -> ten JUnit tests, both sides now loading fixtures from `mobile/shared/fixtures/*.json`.
- **Pose pipeline:** MediaPipe Tasks Android SDK consumed directly via Gradle.
- **Data layer:** SQLDelight on Android, schema mirrors SwiftData / GRDB schema from iOS.
- **CI:** matrix expands to add an Android job (`ubuntu-latest`, Gradle).
- **Server retirement:** `FTTrial2.0/` deleted.
- **Lifetime UI duplication:** kicks in here. ~600-900 LoC per platform, steady-state ~1.3-1.5x per new screen. Accepted.

## Future phase compatibility

The listed FollowThrough roadmap items hold up cleanly under E, and in most cases favour native over cross-platform more than v1 does.

| Phase | Platform requirement | Fit under E |
|---|---|---|
| Combine shot forms (AI blend of pro templates) | Math (DTW + weighted interpolation) + optional small on-device NN (CoreML / TFLite) | Math is framework-agnostic. Implemented twice under spec + fixtures. ML via CoreML (iOS) + TFLite (Android) first-party SDKs. |
| Generate custom mechanics (LLM-synthesized coaching) | LLM API call + structured prompt + local session history | HTTPS + JSON; framework-agnostic. No advantage to any stack. |
| Overlay forms (your ghost on pro video, exported share clip) | Frame-accurate video composition + overlay rendering + export for share | Native video-composition APIs are the paved road: `AVVideoCompositionCoreAnimationTool` on iOS, ExoPlayer + Surface on Android. Flutter / RN have historical friction here. |
| Music-synced guided drills (shoot on beat, score rhythm) | Low-latency audio engine + precise haptic timing + cross-device watch haptic sync | Strongly native. AVAudioEngine / Oboe / Core Haptics. Cross-framework audio-haptic timing is not viable in practice. |
| Expanded on-device ML models | CoreML (iOS) + TFLite (Android) | First-party SDKs on both platforms. Zero wrapper overhead under E. |

Net: future phases strengthen the native-vs-cross-platform argument decisively. They marginally favour A's shared-math story over E, because each phase adds analysis code; they equally strengthen E's no-glue-tax story, because each phase adds platform APIs. The two compounding effects roughly cancel.

## Rust + UniFFI as a v3+ escape valve

If shared-core hunger becomes acute (say, 5+ phases in with meaningful math duplication pain), the modern industry answer is Rust + UniFFI, not KMP. Mozilla built UniFFI; Signal and Bitwarden ship this pattern.

- Rust crate for analysis + blend math + prompt-building + any future custom models' glue code.
- Consumed as a Swift Package on iOS, an AAR on Android.
- AI fluency on Rust is substantially higher than on KMP's glue layer; UniFFI tooling is clean and well-documented.
- Migration from E is straightforward: add a Rust crate alongside two native apps. Migration from Flutter / RN would require tearing out the framework.

Filed as a post-v3 option, not a v1 choice; requires Rust ramp-up that does not pay off at current scope. But knowing the escape valve exists means the "we might want shared-core someday" argument for A is weaker than it looks.

## Critical files referenced by this decision

- `demo/analysis.py` -- analysis functions to port.
- `demo/tests/test_analysis.py` -- contract tests to port.
- `FollowThrough/source/SkeletonMaker.py` -- MediaPipe config, joint list (18), Savgol params.
- `~/Documents/Coding/FollowThrough/app/src/main/java/com/example/samch/followthrough/Startup.java` -- 15-pro seed data.
- `~/Documents/Coding/FollowThrough/app/src/main/java/com/example/samch/followthrough/UserVPro.java` -- reference behaviour for Compare screen.
- `~/Documents/Coding/FollowThrough/app/src/main/java/com/example/samch/followthrough/HomePage.java` -- reference for ProList search + favourites.
- `FollowThrough/FTTrial2.0/mysite/poses/views.py` -- legacy upload contract being retired.

## Pre-implementation validation

The original draft listed three KMP-specific spikes (MediaPipe cinterop, Skie interop, two-AVPlayer sync). Under E, two evaporate. Under iOS-first phasing, the remaining list shrinks further: only the iOS perf check is a true v1 gate. Other validations either move to v1 implementation work or defer to their relevant phase.

### v1 gate

1. **MediaPipe Tasks iOS in pure Swift hits 30fps on iPhone 12+ for the 18-joint pose config.** Validate, not assume. SDK is first-party; this confirms frame-rate, not interop. Two-day prototype.

If this fails (unlikely for a first-party SDK), v1 falls back to record-then-analyse (no real-time live feedback). v1.5 watch then becomes strictly parallel-IMU-signal-only, not phone-relay.

### v1 implementation work (not pre-validation)

- **AVPlayer dual-player sync.** Two `AVPlayer`s sharing a clock via `CADisplayLink`; drift stays under one frame over 15 seconds. Standard AVFoundation pattern; one afternoon during the Compare screen build.

### v1.5 gate

- **WatchConnectivity feedback latency round-trip.** Phone sends an event; watch receives and fires haptic within 150ms end-to-end. Apple's published benchmark is 50-100ms. Real-device sanity check before committing the live-feedback UX.

### v2 deferred

- **MediaPipe Tasks Android perf check** on Pixel-class device. Same shape as v1's iOS check.
- **SQLDelight schema parity** with iOS data layer.

## Open questions post-decision (not this doc's scope)

1. iOS min version: 17 (SwiftUI 5 maturity, watchOS 10) or 16 (broader install base)? **v1 gate.**
2. SwiftData vs GRDB for v1 history queries. SwiftData if queries stay simple; GRDB if you want raw SQL and known performance characteristics.
3. Fixture authoring for v1 XCTests: capture outputs from `demo/analysis.py` with a one-time script, or hand-write synthetic keypoint sequences for edge cases?
4. Watch v1.5 timing: bundled with v1.0 store release as a "two-app launch", or shipped as v1.5 a few weeks after?
5. `SHARED_SPEC.md` format (deferred to v2): prose + YAML function signatures, or typed markdown with JSON Schema fragments for inputs / outputs? Decision happens when Android port begins.
6. Android min SDK (deferred to v2): API 26 (8.0) for CameraX + modern Compose baseline, or API 29 (10) for simpler permission model?
7. Shot overlay export (v3+): compute overlay at record time and cache video with baked-in overlay, or compute per-frame at playback and export via video-composition? Storage-vs-compute tradeoff.
