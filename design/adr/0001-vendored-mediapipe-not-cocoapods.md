# ADR 0001: Vendored MediaPipe xcframeworks instead of CocoaPods

**Status:** Accepted 2026-04-28.
**Decision owner:** Sameer (solo).

## Context

Follow Through iOS v1 needs MediaPipe Tasks Vision for on-device pose extraction. Google distributes MediaPipe iOS through three nominal paths in 2026-04:

1. **CocoaPods** -- Google's official channel. Spec at `cocoapods.org/pods/MediaPipeTasksVision`.
2. **Swift Package Manager** -- not available. Google has not shipped SPM support for MediaPipe iOS.
3. **Manual download** -- no public CDN URL pattern for direct xcframework downloads. Binaries are fetched transparently via the CocoaPods spec.

We initially adopted CocoaPods. It cost a full session of debugging without producing a clean build. The detailed failure modes:

- The `MediaPipeTasksVision.xcframework`'s `Info.plist` declares `LibraryPath = MediaPipeTasksVision.a` (a static library at slice root), but the actual contents are a static framework: `MediaPipeTasksVision.framework/MediaPipeTasksVision`, where the inner binary is itself a static `.a` archive wrapped in a `.framework` directory. CocoaPods's xcconfig generation matches the Info.plist's claim, not the actual layout.
- `OTHER_LDFLAGS` from CocoaPods emits `-l"MediaPipeTasksVision"` (looking for `libMediaPipeTasksVision.a`), but the linker actually needs `-framework "MediaPipeTasksVision"` (looking for `MediaPipeTasksVision.framework/MediaPipeTasksVision`).
- `FRAMEWORK_SEARCH_PATHS` points at `Pods/MediaPipeTasksVision/frameworks/` (contains the .xcframework), not at the per-build-products XCFrameworkIntermediates dir where the right slice is unpacked.
- Xcode 26's Swift 6 explicit-module-build fails to resolve the module dependency through CocoaPods's framework integration.

A `post_install` hook in the Podfile can patch all of this, but each of five iterations surfaced a new failure mode (FRAMEWORK_SEARCH_PATHS, then -l vs -framework, then explicit-modules, then library-not-found, then auto-embed). Five workarounds for one dependency is not a sustainable v1 setup.

Apple Vision was rejected as a fallback in a separate decision ("no compromises on the pose model" -- Apple Vision lacks foot/heel keypoints critical to shot mechanics).

## Decision

**Vendor the MediaPipe xcframeworks directly under `mobile/ios/Vendor/MediaPipe/`. Wrap them in local Swift Packages (`PoseKit` for the protocol, `MediaPipeKit` for the implementation) consumed by the FollowThrough Xcode project. Track binaries via Git LFS.**

The vendoring approach:
- Extract `MediaPipeTasksVision.xcframework`, `MediaPipeTasksCommon.xcframework`, and the `graph_libraries/*.a` force-load files from a local `pod install` in a scratch directory.
- Patch the Info.plist of each xcframework so `LibraryPath = <Name>.framework` (matches reality, removes the CocoaPods-required indirection). See `mobile/ios/Vendor/MediaPipe/repackage_xcframeworks.sh`.
- Generate stub Info.plist files inside each `.framework` slice so Xcode's `binaryTarget` auto-embed step can copy the framework into `<App>.app/Frameworks/` (required even though the framework's binary is statically linked).
- `MediaPipeKit/Package.swift` declares `binaryTarget`s pointing at the vendored xcframeworks (relative path) and links the necessary system frameworks.
- The `-force_load` directives for the graph .a files live in the `FollowThrough` target's `OTHER_LDFLAGS` (sdk-conditional), not in the Swift Package, because SPM's `binaryTarget` doesn't express force-load semantics cleanly across SDK variants.
- `mobile/ios/scripts/wire_packages.rb` is a one-time setup script using the `xcodeproj` Ruby gem to add the local-package references + the linker settings to the pbxproj. Run once; the resulting pbxproj is committed and is the source of truth.

## Alternatives considered

### A. Keep CocoaPods, fight harder on the post_install hook

Possible but every release of CocoaPods, Xcode, or MediaPipe risks breaking the workaround. We'd own a debugging burden indefinitely. A core dependency shouldn't be that fragile.

### B. Build MediaPipe from source via Bazel

MediaPipe is open source (Apache 2.0). We could build our own xcframeworks from a pinned MediaPipe commit. Pros: complete control, custom graph slimming for app-size, security audit-friendly. Cons: Bazel toolchain (~5GB), 30-60 min builds, ongoing investment to track upstream. Filed as a future option if we ever need a custom build (e.g., shrinking the graph to pose-only kernels saves ~50% on the .a size).

### C. Apple Vision as a fallback provider

Rejected separately: Apple Vision lacks foot/heel keypoints. Quality of pose model is non-negotiable for shot mechanics.

### D. Custom basketball-trained CoreML model

Long-term roadmap. Requires labeled training data we don't have. Until then, MediaPipe Pose Heavy is the best off-the-shelf pose model satisfying the use-case constraints.

## Consequences

- The repo carries ~280MB of MediaPipe binaries via Git LFS. GitHub's free-tier LFS limits (1GB storage, 1GB bandwidth/month) accommodate this for a solo dev.
- New MediaPipe versions require a manual update procedure (one-shot `pod install` in a scratch dir, copy outputs, rerun `repackage_xcframeworks.sh`, bump `Vendor/MediaPipe/VERSION.md`). Documented; takes ~30 min per upgrade.
- We do not depend on CocoaPods, system Ruby, or its post_install ecosystem. The build is reproducible from `git clone` + `git lfs pull` + opening `FollowThrough.xcodeproj`.
- The `PoseProvider` protocol in PoseKit makes the pose-model choice reversible. A future provider (custom CoreML, MediaPipe v0.11, RTMW-l-via-CoreML) is a new local Swift Package next to MediaPipeKit and a one-line swap at the call site. The vendoring pattern is the same.

## Trigger conditions to revisit this ADR

- Google ships official Swift Package Manager support for MediaPipe iOS. Revisit; SPM is structurally cleaner than vendoring.
- A new iOS-deployable, foot/heel-capable, Apache-or-similar-licensed pose model becomes the production choice. The vendoring approach generalizes; the ADR's reasoning still stands.
- Bandwidth or storage on Git LFS becomes painful (unlikely at solo scale; possible if the repo grows or we add many model variants).

## References

- MediaPipe Pose Landmarker iOS guide: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/ios
- MediaPipe iOS setup guide (states CocoaPods is the only official channel): https://ai.google.dev/edge/mediapipe/solutions/setup_ios
- Pose-model selection rationale: `design/pose-model-evaluation-2026-04.md`
- Architecture plan that this ADR supports: `design/MOBILE_ARCHITECTURE.md`
- Implementation plan: `design/FT_IOS_V1_PLAN.md`
