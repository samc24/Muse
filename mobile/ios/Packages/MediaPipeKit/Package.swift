// swift-tools-version: 5.10
import PackageDescription

// MediaPipeKit wraps Google's MediaPipe Tasks Vision iOS SDK behind PoseKit's
// PoseProvider protocol. The vendored xcframeworks live at ../../Vendor/MediaPipe/
// (relative to this Package.swift). See ../../Vendor/MediaPipe/VERSION.md for
// the pinned version + update procedure.
//
// The binaryTargets handle the .framework-wrapped static libs.
// The graph_libraries .a files (force-loaded at link time to register TFLite
// op kernels) are wired up at the app-target level in FollowThrough.xcodeproj's
// Other Linker Flags -- SPM's binaryTarget doesn't expose force_load semantics,
// and pushing it down to this package would tie path resolution to SPM's build
// directory layout, which is fragile.

let package = Package(
    name: "MediaPipeKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "MediaPipeKit", targets: ["MediaPipeKit"])
    ],
    dependencies: [
        .package(path: "../PoseKit")
    ],
    targets: [
        .binaryTarget(
            name: "MediaPipeTasksVision",
            path: "../../Vendor/MediaPipe/MediaPipeTasksVision.xcframework"
        ),
        .binaryTarget(
            name: "MediaPipeTasksCommon",
            path: "../../Vendor/MediaPipe/MediaPipeTasksCommon.xcframework"
        ),
        .target(
            name: "MediaPipeKit",
            dependencies: [
                "PoseKit",
                "MediaPipeTasksVision",
                "MediaPipeTasksCommon"
            ],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("CoreImage"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("QuartzCore"),
                .linkedLibrary("c++")
            ]
        ),
        .testTarget(
            name: "MediaPipeKitTests",
            dependencies: ["MediaPipeKit"]
        )
    ]
)
