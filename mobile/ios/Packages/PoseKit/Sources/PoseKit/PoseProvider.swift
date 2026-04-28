import CoreVideo
import Foundation

/// Per-frame pose detection. The single load-bearing abstraction in PoseKit.
///
/// Used by both the offline `extractPose(from:using:)` AVAssetReader loop
/// and the live `AVCaptureVideoDataOutput` callback path that Sprint 4+
/// will introduce for in-recording pose feedback. Implementations may be
/// stateful (MediaPipe's `PoseLandmarker` in `.video` mode tracks across
/// frames). The contract is:
///
/// - Timestamps are strictly monotonically increasing within a session.
///   Resetting (e.g., starting a new recording) requires creating a new
///   provider instance, not reusing one with a backwards timestamp.
/// - The returned `PoseFrame` contains the canonical 18-joint set; the
///   provider owns mapping its native output to that set. Joints the
///   provider can't produce (e.g., HEEL on Apple Vision) are simply absent
///   from the keypoints dictionary.
/// - `nil` return means the provider did not detect a pose in this frame
///   (no person visible, fully occluded, etc.), not that an error occurred.
///   For errors throw `PoseProviderError`.
///
/// Provider configuration (model variant, device vs. cloud, etc.) lives in
/// the concrete provider's init, not on the protocol. The protocol stays
/// one-method clean so swapping providers is a one-line change at the
/// call site.
public protocol PoseProvider: Sendable {
    func detect(pixelBuffer: CVPixelBuffer, timestampMs: Int) async throws -> PoseFrame?
}
