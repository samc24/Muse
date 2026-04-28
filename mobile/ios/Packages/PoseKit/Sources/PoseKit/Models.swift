import CoreGraphics
import Foundation

/// The 18 canonical joints used across Follow Through pose providers.
///
/// This superset is intentional. The current iOS analysis layer (ported from
/// `demo/analysis.py`) only consumes 12 of these (wrists, elbows, shoulders,
/// hips, knees, ankles), but the data-extraction layer in
/// `FollowThrough/source/SkeletonMaker.py` standardised on 18 to keep
/// foot/heel/foot_index available for shot-mechanics features that haven't
/// been written yet. Locking the schema at 18 now keeps the door open for
/// stance-and-base analysis without a future schema migration.
public enum CanonicalJoint: String, CaseIterable, Codable, Sendable {
    case head = "HEAD"
    case neck = "NECK"
    case leftShoulder = "L_SHOULDER"
    case rightShoulder = "R_SHOULDER"
    case leftElbow = "L_ELBOW"
    case rightElbow = "R_ELBOW"
    case leftWrist = "L_WRIST"
    case rightWrist = "R_WRIST"
    case leftHip = "L_HIP"
    case rightHip = "R_HIP"
    case leftKnee = "L_KNEE"
    case rightKnee = "R_KNEE"
    case leftAnkle = "L_ANKLE"
    case rightAnkle = "R_ANKLE"
    case leftHeel = "L_HEEL"
    case rightHeel = "R_HEEL"
    case leftFootIndex = "L_FOOT_INDEX"
    case rightFootIndex = "R_FOOT_INDEX"
}

/// A single keypoint detection.
///
/// Coordinates are normalized to the source image, with origin at top-left
/// (matches MediaPipe and CoreImage conventions; iOS Vision uses bottom-left,
/// providers wrapping it must flip y in their canonical mapping).
///
/// - `x`, `y`: 0..1 in image space.
/// - `z`: depth from the hip-midpoint plane, in roughly meter-scale, positive
///   toward the camera. Provider-defined; if a provider does not emit depth
///   it returns 0. Consumers must not assume z is meaningful unless the
///   provider documents it.
/// - `visibility`: 0..1, "probability the joint is reliably localised in
///   this frame." Combines presence (joint is in the frame) and visibility
///   (joint is not occluded) for MediaPipe; pure confidence for other
///   providers. Downstream code that drops low-confidence joints should
///   threshold against this value (Follow Through uses 0.5 as the default cut).
public struct Keypoint: Codable, Sendable, Hashable {
    public let x: Float
    public let y: Float
    public let z: Float
    public let visibility: Float

    public init(x: Float, y: Float, z: Float = 0, visibility: Float) {
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
    }
}

/// One frame's worth of pose landmarks.
///
/// Implementations may omit joints (the dictionary entry is absent rather
/// than nil) when a joint was not detected. Downstream code must treat
/// missing joints as "unknown for this frame", not "joint at origin".
public struct PoseFrame: Sendable, Hashable {
    public let timestampMs: Int
    public let keypoints: [CanonicalJoint: Keypoint]

    public init(timestampMs: Int, keypoints: [CanonicalJoint: Keypoint]) {
        self.timestampMs = timestampMs
        self.keypoints = keypoints
    }
}

/// Result of running a `PoseProvider` over an entire video.
///
/// `imageSize` is the source video's pixel dimensions and is needed to
/// denormalize keypoints back to pixel space (e.g., for skeleton overlay
/// rendering on top of the original frame in the Compare screen).
public struct PoseExtractionResult: Sendable {
    public let frames: [PoseFrame]
    public let elapsedSeconds: Double
    public let videoDurationSeconds: Double
    public let videoFps: Double
    public let imageSize: CGSize

    public init(
        frames: [PoseFrame],
        elapsedSeconds: Double,
        videoDurationSeconds: Double,
        videoFps: Double,
        imageSize: CGSize
    ) {
        self.frames = frames
        self.elapsedSeconds = elapsedSeconds
        self.videoDurationSeconds = videoDurationSeconds
        self.videoFps = videoFps
        self.imageSize = imageSize
    }

    /// Average per-frame processing throughput, in frames per second.
    /// Comparable against `videoFps` to know whether real-time live analysis
    /// is feasible on the current device.
    public var processingFps: Double {
        guard elapsedSeconds > 0 else { return 0 }
        return Double(frames.count) / elapsedSeconds
    }
}

/// Canonical error type all `PoseProvider` implementations must throw.
public enum PoseProviderError: LocalizedError, Sendable {
    /// The provider's model weights or configuration file is missing.
    case modelMissing(String)
    /// The video URL could not be opened or has no video track.
    case videoUnreadable(String)
    /// Provider-specific failure that doesn't fit the canonical cases.
    case providerSpecific(String)

    public var errorDescription: String? {
        switch self {
        case .modelMissing(let detail):
            return "Pose model missing: \(detail)"
        case .videoUnreadable(let detail):
            return "Video unreadable: \(detail)"
        case .providerSpecific(let detail):
            return detail
        }
    }
}
