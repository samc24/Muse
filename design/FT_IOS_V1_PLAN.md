# Follow Through iOS v1 -- Implementation Plan (2026)

**Status:** Draft 2026-04-25. Working on branch `ft-ios`.
**Scope:** Concrete plan for shipping the iOS phone app (v1) in the architecture set by [`MOBILE_ARCHITECTURE.md`](MOBILE_ARCHITECTURE.md). Covers bootstrap through TestFlight.
**Decision owner:** Sameer (solo).

This doc is a living plan. Phases get marked done inline as they ship; deviations get noted in place. Implementation conversations reference phase numbers.

## Goal

Ship a TestFlight-ready Follow Through iOS app that:

1. Lets the user browse 15 pro shooters, search them, favourite them.
2. Records a user shot in landscape via in-app camera.
3. Compares the user shot side-by-side with a chosen pro, with synced playback + per-side speed control + trim.
4. Extracts pose on-device via MediaPipe Tasks iOS for both videos.
5. Computes similarity score + descriptor + top-3 coaching advice + per-joint deviation chart + wrist-trajectory chart, matching the `demo/analysis.py` contract.
6. Persists shot history per user via SwiftData + CloudKit (cross-device sync free).
7. Runs on iPhone 12+ (iOS 17.4+) at 30fps for the on-device pose pipeline.

Out of scope for v1: watchOS (v1.5), Android (v2), combine-forms / custom-mechanics LLM (v3+), music-synced drills (v3+), overlay export to share (v3+), accounts beyond iCloud user, multiple users on one device.

## Prerequisites (developer setup, before any code)

These are not Sameer-skippable. Resolve all four before Sprint 1.

| Prereq | What | Notes |
|---|---|---|
| Apple Developer Program | Paid enrollment ($99/year) at developer.apple.com | Required for **CloudKit** (free Apple ID can't provision iCloud capability). Without it, SwiftData works but the cross-device sync we picked it for does not. Enrollment is usually 24-48 hours. Sameer's task -- payment method must be his. |
| Xcode 15.3+ | Install from Mac App Store | ~10-15GB. SwiftData + iOS 17.4 SDK floor. Xcode 16 (current) is fine. |
| iCloud signed in on test iPhone | iPhone 17, signed in to the same Apple ID as the dev account | Without it, CloudKit sync test will silently no-op. |
| Test device + USB cable | iPhone 17 paired with the Mac via USB or wireless dev | First-time pairing requires "Trust this Mac" prompt + enabling Developer Mode in Settings. |

While waiting for Apple Developer Program approval (24-48h), Xcode install + iCloud check can happen in parallel.

## Pre-implementation gate (the v1 perf check)

**MediaPipe Tasks iOS perf check** (the only v1 gate from `MOBILE_ARCHITECTURE.md`).

- Bundle a single shot clip from `demo/static/` into the Xcode project, run MediaPipe Tasks iOS on it via Swift Package Manager, measure sustained fps for the 18-joint pose config on iPhone 17.
- Pass: 30fps sustained over 5+ seconds.
- Fail: drops to <20fps. v1 falls back to record-then-analyse (no real-time feedback); v1.5 watch becomes IMU-only.
- iPhone 17 (A19 chip) makes this functionally guaranteed; we still validate on real hardware as a habit.

This gets folded into Sprint 2 (below) rather than living as a separate spike, since the prereqs already cost a couple of days of clock.

## Repo + Xcode bootstrap (Phase 0)

### Directory layout

```
Muse/mobile/ios/
+- FollowThrough.xcodeproj/
+- FollowThrough/                       # app target
|  +- FollowThroughApp.swift            # @main, ModelContainer config
|  +- Resources/
|  |  +- pros_seed.json                 # 15-pro seed lifted from Startup.java
|  |  +- pro_videos/                    # bundled pro shot clips (Shaq, Nash, ...)
|  |  +- Assets.xcassets/
|  +- Models/                           # SwiftData @Model + value types
|  |  +- Pro.swift
|  |  +- Shot.swift
|  |  +- Session.swift
|  |  +- Keypoint.swift                 # struct, not persisted
|  +- Analysis/                         # port of demo/analysis.py
|  |  +- Analysis.swift                 # similarity, deviation, descriptor, advice
|  |  +- PoseExtractor.swift            # MediaPipe Tasks iOS wrapper
|  |  +- Smoothing.swift                # Savitzky-Golay
|  +- Capture/
|  |  +- CameraSession.swift            # AVCaptureSession lifecycle
|  |  +- RecordingService.swift         # save to Documents + thumbnail
|  +- Playback/
|  |  +- DualPlayerSync.swift           # CADisplayLink-synced AVPlayers
|  |  +- OverlayRenderer.swift          # skeleton overlay on top of video
|  +- Views/                            # SwiftUI screens (one file each)
|  |  +- SplashView.swift
|  |  +- ProListView.swift
|  |  +- ProDetailView.swift
|  |  +- RecordView.swift
|  |  +- CompareView.swift
|  |  +- ResultsView.swift
|  |  +- HistoryView.swift
|  |  +- SettingsView.swift
|  +- ViewModels/                       # @Observable
|  +- Utilities/
+- FollowThroughTests/                  # XCTest target
|  +- AnalysisTests.swift               # ports demo/tests/test_analysis.py
|  +- PoseExtractorTests.swift          # integration: bundled fixture video
|  +- fixtures/
|  |  +- shaq_keypoints.json
|  |  +- nash_keypoints.json
|  |  +- expected_outputs.json
+- README.md                            # build + run + test
```

### Dependencies (SPM)

- `MediaPipeTasksVision` (Google) -- on-device pose estimation.
- No others for v1.

Native frameworks (no SPM): SwiftUI, SwiftData, AVFoundation, CoreMotion (idle until v1.5), Swift Charts, CloudKit, OSLog.

### `.gitignore` additions for `mobile/ios/`

```
xcuserdata/
*.xcuserstate
build/
DerivedData/
*.xcworkspace/xcuserdata/
.swiftpm/
```

### Bundle ID + app name

- App name: `Follow Through`
- Bundle ID: `com.sameerc.followthrough`
- iCloud container: `iCloud.com.sameerc.followthrough`

## Screen inventory

| Screen | Complexity | Estimated LoC | Notes |
|---|---|---|---|
| Splash | Tiny | ~30 | Logo + 1s transition. Skip if SwiftUI splash defaults are good enough. |
| ProList | Small | ~150 | NavigationStack + List + searchable + favourite toggle. 15 pros from seed JSON. |
| ProDetail | Small | ~150 | Pro bio + thumbnail + "Record vs them" CTA. |
| Record | Medium | ~400 | AVCaptureSession preview + record button + countdown + landscape lock + save flow. |
| Compare | Large | ~600 | Two `AVPlayer`s synced via `CADisplayLink`, per-side speed (0.25-2.0x), trim handles, scrub, overlay toggle. |
| Results | Medium | ~400 | Similarity + descriptor + top-3 advice + per-joint chart + wrist-trajectory chart. |
| History | Small | ~150 | List of past shots, tap to re-view results. |
| Settings | Small | ~150 | iCloud sync toggle, about, debug, data export. |

Total UI estimate: ~2000 LoC. Plus ~600-800 LoC for analysis + capture + playback + models. Order of ~3000 LoC for v1.

## Implementation order (sprints + phases)

The early phases are restructured into three rapid sprints, each ending with something visible on Sameer's iPhone. After Sprint 3 the pipeline is real on-device; the remaining phases layer screens on top.

### Sprint 1: "Hello, Follow Through" on device (~1 day)

Goal: app builds, signs, deploys to Sameer's iPhone, shows a SwiftUI "Hello" screen.

- Xcode project scaffold per directory layout above.
- Bundle ID `com.sameerc.followthrough`, signing configured for Sameer's Apple Developer team.
- iCloud capability + container `iCloud.com.sameerc.followthrough` provisioned (CloudKit dashboard).
- SPM dependency: `MediaPipeTasksVision` added (not used yet -- just verify it resolves).
- Single SwiftUI screen: app name, version, build number, "MediaPipe loaded: yes/no" status.
- Build to simulator -> works. Build to iPhone 17 over USB -> works. Trust developer profile on device.
- First commits on `ft-ios`.

This sprint is mostly Apple-pipeline-debugging, not coding. Boring, but it has to be solid before anything else makes sense.

### Sprint 2: Pose extraction visible on device (~2 days)

Goal: tap a button on the phone, see MediaPipe extract pose from a bundled video, see frame count + fps printed on screen. **The v1 perf gate validated by holding the phone.**

- Bundle one shot clip from `demo/static/` into `FollowThrough/Resources/pro_videos/`.
- `PoseExtractor.swift` -- MediaPipe Tasks iOS wrapper: `extract(videoURL:) async -> [PoseFrame]`.
- Single screen: "Run pose on Shaq.mp4" button -> async task -> displays "Extracted 84 frames at 31.2 fps avg".
- `OSLog` output visible in Xcode console for debugging.
- One XCTest integration test: bundled video -> non-empty keypoint array.

If perf gate fails here (<20fps sustained), we stop and reconsider before building more. Almost certainly it passes on iPhone 17.

### Sprint 3: Similarity score visible on device (~2 days)

Goal: same screen now also computes Shaq-vs-Nash similarity using the ported analysis layer, displays it.

- SwiftData `@Model` types: `Pro`, `Shot`, `Session` (CloudKit-backed). Keypoint struct (value type, not persisted).
- `Analysis.swift` ported from `demo/analysis.py`. Pure functions: `normalize_keypoints`, `per_joint_deviation`, `similarity_score`, `similarity_descriptor`, `generate_advice`, `wrist_trajectories`, `trajectory_figure`.
- `Smoothing.swift` ported from `FollowThrough/source/SkeletonMaker.py` Savgol params.
- XCTest: 10 tests ported from `demo/tests/test_analysis.py`, loading shared JSON fixtures captured one-shot from a Python script that runs `demo/analysis.py` on canonical inputs. All pass within float tolerance.
- Bundle Nash clip too. Sprint 2's screen extends: "Shaq vs Nash similarity: 50.5% (average). Top advice: bend knees more, ..."
- This is the first moment the app does the actual product thing.

After Sprint 3, the pipeline is real. The remaining phases below build proper screens around it.

### Phase 3: Capture + Compare (3-4 days)

- `CameraSession.swift` + `RecordView` -- landscape-locked record + save to Documents.
- `DualPlayerSync.swift` + `CompareView` -- two `AVPlayer`s, synced clock, per-side speed, trim handles, scrub.
- After this phase, the user can record a shot and view it next to a pro.

### Phase 4: Results + analysis presentation (2 days)

- `ResultsView` -- similarity number, descriptor, top-3 advice text, per-joint Swift Charts bar chart, wrist trajectory line chart.
- Wire up: Compare -> "Analyse" -> ResultsView.

### Phase 5: ProList / ProDetail / History (2 days)

- `ProListView` -- list + search + favourites (favourites persist via SwiftData).
- `ProDetailView` -- bio + "Record vs them" CTA.
- `HistoryView` -- past shots list, tap to re-view ResultsView.

### Phase 6: Polish + TestFlight (3-4 days)

- SwiftData CloudKit container wiring + first-launch iCloud check.
- `SettingsView` -- iCloud sync toggle, app version, debug menu, export data.
- Empty / error / loading states across all screens.
- App icon, launch screen, accessibility passes.
- App Store Connect setup, TestFlight build, internal dogfood.

**Total estimated calendar time: ~3 weeks of focused solo work.** Realistic with AI-assist; sandbag accordingly for life.

## Commit strategy on `ft-ios`

Solo + no review ritual = one PR back to main, but commits inside the branch are small + reviewable + bisectable. Roughly one commit per sub-step within a phase. Examples:

- `Bootstrap: Xcode project + SPM + Hello world`
- `Add SwiftData models (Pro, Shot, Session)`
- `Port analysis.py to Analysis.swift`
- `Add XCTest fixtures from demo/analysis.py outputs`
- `Add PoseExtractor wrapping MediaPipe Tasks iOS`
- `Add bundled-video smoke test`
- `Add CameraSession + RecordView`
- ... etc.

Branch lands as a single fast-forward to main when v1 ships TestFlight. Per the branch-hygiene rule, `ft-ios` deleted local + remote after merge.

## Test strategy

| Layer | Test type | Coverage target |
|---|---|---|
| `Analysis.swift` | XCTest (ported from pytest) | 100% of public functions, all 10 fixture cases pass within float tolerance. |
| `PoseExtractor` | XCTest integration with bundled fixture video | Single happy-path: extract from `shaq.mp4`, assert ~30fps + non-empty keypoint sequence. |
| `Smoothing` | XCTest unit | Savgol output matches Python output within float tolerance on a fixture sequence. |
| `DualPlayerSync` | XCTest timing | Drift under 1 frame over 15 seconds (real `AVPlayer`s, not mocked). |
| SwiftData models | XCTest schema | Migration from v1 schema to a hypothetical v2 schema works (forward-compat habit from day 1). |
| ViewModels | XCTest | State transitions on key flows (record -> analyse -> results). |
| Views | Snapshot tests | Stable screens only: ProList, Results. Compare + Record skipped (too dynamic). |

XCTest runs locally + in CI (single `macos-latest` job).

## Open implementation questions (decide as they come up)

1. **iCloud sync: required or optional?** Default: required, simplifies architecture; user can disable via Settings (data stays local-only).
2. **Recording orientation:** landscape-only or also portrait? Default: landscape-only for shooting form.
3. **Video retention:** keep raw recorded video forever or delete after analysis? Default: keep, with manual delete from Settings + History.
4. **Video resolution:** 1080p 30fps, 720p, or device default? Default: 1080p 30fps; downgrade only if MediaPipe perf forces it.
5. **Pro video storage:** bundled in app or downloaded on first launch? Default: bundled in v1 (15 pros, ~50MB total); reconsider if app size becomes prohibitive.
6. **Snapshot test framework:** Apple's preview snapshots or Point-Free's `swift-snapshot-testing`? Default: Point-Free, mature + actively maintained.
7. **Logging:** OSLog only, or also a debug-only on-screen overlay? Default: OSLog + add overlay if debugging needs it.

## Critical references

- [`MOBILE_ARCHITECTURE.md`](MOBILE_ARCHITECTURE.md) -- parent architecture decision; v1 stack section is canonical.
- `demo/analysis.py` + `demo/tests/test_analysis.py` -- source of truth for analysis port + test fixtures.
- `FollowThrough/source/SkeletonMaker.py` -- MediaPipe + Savgol parameter reference.
- `~/Documents/Coding/FollowThrough/app/src/main/java/com/example/samch/followthrough/Startup.java` -- 15-pro seed metadata.
- `~/Documents/Coding/FollowThrough/app/src/main/java/com/example/samch/followthrough/UserVPro.java` -- legacy Compare interaction model (UX reference only; do not lift code).
- `~/Documents/Coding/FollowThrough/app/src/main/java/com/example/samch/followthrough/HomePage.java` -- legacy ProList search behaviour (UX reference only).
- `samc24/ObjectDetect-iOS` (private repo, AI.Reverie 2020) -- prior Swift + AVFoundation + on-device ML reference; pattern source for `CameraSession` + ML pipeline shape.
