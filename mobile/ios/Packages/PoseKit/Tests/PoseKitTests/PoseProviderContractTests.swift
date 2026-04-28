import CoreVideo
import XCTest
@testable import PoseKit

/// Contract tests for PoseKit's protocol semantics, run against a mock
/// provider. Concrete provider integration tests (MediaPipeKit) verify the
/// real implementation against bundled fixture videos at the package level
/// where they live.
final class PoseProviderContractTests: XCTestCase {

    func testCanonicalJointHasAllEighteen() {
        XCTAssertEqual(CanonicalJoint.allCases.count, 18)
        // Spot-check the foot/heel keypoints that MediaPipe Heavy gives us
        // and that Apple Vision and YOLO11-Pose lack.
        XCTAssertTrue(CanonicalJoint.allCases.contains(.leftHeel))
        XCTAssertTrue(CanonicalJoint.allCases.contains(.rightFootIndex))
    }

    func testKeypointDefaultZIsZero() {
        let kp = Keypoint(x: 0.5, y: 0.5, visibility: 0.9)
        XCTAssertEqual(kp.z, 0)
    }

    func testPoseFrameMissingJointsAreAbsentNotZero() {
        let frame = PoseFrame(timestampMs: 0, keypoints: [
            .leftWrist: Keypoint(x: 0.5, y: 0.5, visibility: 0.9)
        ])
        XCTAssertNotNil(frame.keypoints[.leftWrist])
        XCTAssertNil(frame.keypoints[.rightWrist],
                     "Missing joints must be absent from the dict, not stored as origin keypoints.")
    }

    func testPoseExtractionResultProcessingFps() {
        let result = PoseExtractionResult(
            frames: Array(repeating: PoseFrame(timestampMs: 0, keypoints: [:]), count: 90),
            elapsedSeconds: 3.0,
            videoDurationSeconds: 3.0,
            videoFps: 30,
            imageSize: .init(width: 1280, height: 720)
        )
        XCTAssertEqual(result.processingFps, 30, accuracy: 0.001)
    }

    func testProcessingFpsZeroWhenNoElapsedTime() {
        let result = PoseExtractionResult(
            frames: [],
            elapsedSeconds: 0,
            videoDurationSeconds: 0,
            videoFps: 30,
            imageSize: .zero
        )
        XCTAssertEqual(result.processingFps, 0)
    }
}

/// Minimal mock provider for testing the protocol contract without pulling
/// in MediaPipe.
private struct MockPoseProvider: PoseProvider {
    let result: PoseFrame?

    func detect(pixelBuffer: CVPixelBuffer, timestampMs: Int) async throws -> PoseFrame? {
        // Return the canned frame with the timestamp from the caller, so
        // the contract that timestamps come from the AVAssetReader-driven
        // call site (not from the provider) is exercised.
        guard let result else { return nil }
        return PoseFrame(timestampMs: timestampMs, keypoints: result.keypoints)
    }
}

final class PoseProviderProtocolTests: XCTestCase {
    func testProviderReturnsNilForNoDetection() async throws {
        let provider = MockPoseProvider(result: nil)
        let pixelBuffer = makeBlankPixelBuffer(width: 16, height: 16)
        let frame = try await provider.detect(pixelBuffer: pixelBuffer, timestampMs: 100)
        XCTAssertNil(frame)
    }

    func testProviderReturnsCallerTimestamp() async throws {
        let canned = PoseFrame(
            timestampMs: 999, // ignored
            keypoints: [.leftWrist: Keypoint(x: 0.5, y: 0.5, visibility: 0.9)]
        )
        let provider = MockPoseProvider(result: canned)
        let pixelBuffer = makeBlankPixelBuffer(width: 16, height: 16)
        let frame = try await provider.detect(pixelBuffer: pixelBuffer, timestampMs: 100)
        XCTAssertEqual(frame?.timestampMs, 100)
    }

    private func makeBlankPixelBuffer(width: Int, height: Int) -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        return pb!
    }
}
