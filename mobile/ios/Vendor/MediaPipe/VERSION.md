# MediaPipe vendored binaries

## Pinned version

**MediaPipeTasksVision 0.10.33** (and its peer dependency `MediaPipeTasksCommon 0.10.33`).

Pinned 2026-04-28. Validated against Xcode 26.4.1 + iOS 17.4 deployment target.

## Contents

- `MediaPipeTasksVision.xcframework` (Apache 2.0) -- pose, gesture, image classification, etc. The Swift API surface we use.
- `MediaPipeTasksCommon.xcframework` (Apache 2.0) -- shared graph runtime, base options.
- `graph_libraries/libMediaPipeTasksCommon_device_graph.a` -- device-side TFLite graph kernels (~86MB). Force-loaded at link time.
- `graph_libraries/libMediaPipeTasksCommon_simulator_graph.a` -- simulator-side TFLite graph kernels (~169MB). Force-loaded at link time.

The pose model weights are NOT in this directory. They live alongside the app sources at
`mobile/ios/FollowThrough/FollowThrough/Resources/models/pose_landmarker_heavy.task` because the .task file is *app content* (loaded at runtime via `Bundle.main`), not a build-system input.

The bundled model is **`pose_landmarker_heavy.task`** (29MB) -- chosen over `pose_landmarker_lite.task` (5.5MB) and `pose_landmarker_full.task` (~9MB) per the 2026-04 evaluation at `design/pose-model-evaluation-2026-04.md`. Heavy gives materially better foot/heel/ankle accuracy, which is what shot-form analysis cares about. iPhone 17 has the inference headroom.

## Why not CocoaPods or Swift Package Manager

Captured in detail at `design/adr/0001-vendored-mediapipe-not-cocoapods.md`. Short version:
- Google does NOT publish MediaPipe iOS via SPM (as of 2026-04).
- CocoaPods 1.16.2 + Xcode 26 + MediaPipe's xcframework Info.plist mismatch (declares `LibraryPath = MediaPipeTasksVision.a` but ships `MediaPipeTasksVision.framework`) caused 5+ iterations of post_install hook surgery without a clean build. Not worth fighting permanently.
- Vendoring the prebuilt xcframeworks gives us a deterministic, version-pinned, CI-friendly setup that works on any Xcode version Google's binaries support.

## Update procedure

When a new MediaPipe version ships and we want to upgrade:

1. **Verify the new version meets our constraints.** Apache 2.0 license, iOS 17.4+ deployment target, foot/heel/ankle keypoints intact. Read the release notes at https://github.com/google-ai-edge/mediapipe/releases.
2. **Pull the new xcframeworks.** Easiest path is a one-shot `pod install` in a scratch Xcode project:
   ```
   cd /tmp && mkdir mp-fetch && cd mp-fetch
   cat > Podfile <<EOF
   platform :ios, '17.4'
   target 'Scratch' do
     use_frameworks! :linkage => :static
     pod 'MediaPipeTasksVision', '<NEW_VERSION>'
   end
   EOF
   pod install
   ```
3. **Copy the new binaries** into `mobile/ios/Vendor/MediaPipe/`:
   - `Pods/MediaPipeTasksVision/frameworks/MediaPipeTasksVision.xcframework`
   - `Pods/MediaPipeTasksCommon/frameworks/MediaPipeTasksCommon.xcframework`
   - `Pods/MediaPipeTasksCommon/frameworks/graph_libraries/*.a`
4. **Update model weights if Google released new model versions.** Download from
   `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task`
   into `mobile/ios/FollowThrough/FollowThrough/Resources/models/`.
5. **Bump the pinned-version line at the top of this file.** Update the date.
6. **Run the contract tests** (`Packages/PoseKit` + `Packages/MediaPipeKit`) before committing. If anything regresses, revert -- don't ship a degraded provider.
7. **Commit via Git LFS.** Verify with `git lfs ls-files` that the new binaries are LFS-tracked.

## Original source

The xcframeworks shipped to Vendor/MediaPipe/ are the same prebuilt binaries Google distributes
through the CocoaPods `MediaPipeTasksVision` and `MediaPipeTasksCommon` pods. The pod spec at
`https://cocoapods.org/pods/MediaPipeTasksVision` is the canonical version list. Google does not
publish a direct CDN URL pattern for the xcframework tarballs; pull via CocoaPods as in the
update procedure above.
