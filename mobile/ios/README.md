# Follow Through iOS -- developer setup

This is the iOS app target for [Follow Through](../../FollowThrough/), the basketball shot-form analyzer. v1 phase per [`design/FT_IOS_V1_PLAN.md`](../../design/FT_IOS_V1_PLAN.md). Architecture per [`design/MOBILE_ARCHITECTURE.md`](../../design/MOBILE_ARCHITECTURE.md).

## Prerequisites

- macOS Tahoe (26+) with Xcode 26.4+ installed
- Apple Developer Program enrollment (any tier; free Personal Team works for device-only sideloading)
- iPhone 17 (deployment target is iOS 17.4+) or simulator
- Homebrew + `git-lfs` (for the vendored MediaPipe binaries)
- Ruby 2.6+ + the `xcodeproj` gem (only needed if you re-run the package-wiring script)

```bash
# One-time: install LFS support
brew install git-lfs
git lfs install
```

## First-time setup on a fresh clone

```bash
git clone https://github.com/samc24/Muse.git
cd Muse
git lfs pull             # fetches Vendor/MediaPipe binaries (~280MB)
open mobile/ios/FollowThrough/FollowThrough.xcodeproj
```

Then in Xcode:
1. Select your team under **Signing & Capabilities**.
2. Pick a destination (iPhone 17 simulator or your physical device).
3. Cmd+R.

The pbxproj is committed and authoritative. You should not need to re-run any setup scripts on a normal clone.

## Repo layout

```
mobile/ios/
├── FollowThrough/                  -- Xcode app target
│   ├── FollowThrough.xcodeproj
│   ├── FollowThrough/              -- app sources (synced folder group)
│   │   ├── Resources/models/       -- pose_landmarker_heavy.task (LFS)
│   │   └── Resources/pro_videos/   -- shaq.mp4, nash.mp4 (small, in git)
│   ├── FollowThroughTests/
│   └── FollowThroughUITests/
├── Packages/
│   ├── PoseKit/                    -- pose-source-agnostic protocol + types
│   └── MediaPipeKit/               -- MediaPipe-specific PoseProvider impl
├── Vendor/MediaPipe/               -- vendored xcframeworks (LFS)
│   ├── MediaPipeTasksVision.xcframework/
│   ├── MediaPipeTasksCommon.xcframework/
│   ├── graph_libraries/            -- *_device_graph.a, *_simulator_graph.a
│   ├── repackage_xcframeworks.sh   -- post-vendor patch script
│   └── VERSION.md                  -- pinned version + update procedure
├── scripts/
│   └── wire_packages.rb            -- one-time pbxproj surgery (idempotent)
├── .gitattributes                  -- LFS patterns
└── README.md                       -- this file
```

## Common tasks

### Build + run on a device

Open `FollowThrough.xcodeproj` in Xcode 26+. Pick your iPhone in the toolbar device picker. Cmd+R. First time signs the app to your Personal Team and prompts you to trust the developer profile in iPhone Settings.

### Build via CLI

```bash
xcodebuild \
  -project mobile/ios/FollowThrough/FollowThrough.xcodeproj \
  -scheme FollowThrough \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  -configuration Debug build
```

### Run unit tests

```bash
# PoseKit (pure Swift, runs on macOS host)
swift test --package-path mobile/ios/Packages/PoseKit

# MediaPipeKit (iOS only because of the binaryTarget xcframeworks)
xcodebuild test \
  -project mobile/ios/FollowThrough/FollowThrough.xcodeproj \
  -scheme FollowThrough \
  -destination 'platform=iOS Simulator,name=iPhone 17'
```

### Update MediaPipe to a newer version

See [`Vendor/MediaPipe/VERSION.md`](Vendor/MediaPipe/VERSION.md) for the manual update procedure (one-shot `pod install` in a scratch dir, copy outputs, rerun `repackage_xcframeworks.sh`, bump version, commit via LFS).

### Recover from a destructive pbxproj edit

If the `FollowThrough.xcodeproj/project.pbxproj` ever gets restored to a state where PoseKit / MediaPipeKit aren't linked (e.g., regenerated from a fresh Xcode template), re-run the wiring script:

```bash
gem install xcodeproj   # one-time, if not already installed
ruby mobile/ios/scripts/wire_packages.rb
```

The script is idempotent. On normal clones with the committed pbxproj, you do not need to run this.

## Architecture in one paragraph

The pose model is the heart of the product, so it lives behind a `PoseProvider` protocol in `PoseKit` (pure-Swift, pose-source-agnostic). The production implementation, `MediaPipePoseProvider`, lives in `MediaPipeKit`, which depends on `PoseKit` and binds to MediaPipe Tasks Vision via vendored xcframeworks under `Vendor/MediaPipe/`. The FollowThrough app target depends on both packages. Swapping pose models (custom CoreML, future RTMW-l, MediaPipe v0.11) is a one-line change at the call site -- see [`design/pose-model-evaluation-2026-04.md`](../../design/pose-model-evaluation-2026-04.md) and [`design/adr/0001-vendored-mediapipe-not-cocoapods.md`](../../design/adr/0001-vendored-mediapipe-not-cocoapods.md) for the rationale.
