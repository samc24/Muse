import XCTest
@testable import MediaPipeKit

final class MediaPipePoseProviderTests: XCTestCase {

    /// Sanity-check that the variant -> filename mapping is correct.
    /// Acts as a tripwire if MediaPipe's model filename convention shifts
    /// in a future version.
    func testVariantFilenames() {
        XCTAssertEqual(MediaPipePoseProvider.ModelVariant.lite.bundledFilename,
                       "pose_landmarker_lite")
        XCTAssertEqual(MediaPipePoseProvider.ModelVariant.full.bundledFilename,
                       "pose_landmarker_full")
        XCTAssertEqual(MediaPipePoseProvider.ModelVariant.heavy.bundledFilename,
                       "pose_landmarker_heavy")
    }

    // Note: integration tests against real MediaPipe inference (load model,
    // detect on a fixture video) live at the app level rather than here,
    // because the bundled model file lives in the FollowThrough app's
    // Resources/, not in this package. App-level Sprint 2 validation
    // (tap button -> see frame count + fps on iPhone 17) is the integration
    // checkpoint that exercises this provider end-to-end.
}
