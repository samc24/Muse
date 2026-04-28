import CoreVideo
import Foundation
import MediaPipeTasksVision
import PoseKit

/// Production pose provider for Follow Through v1.
///
/// Wraps Google's MediaPipe Tasks Vision PoseLandmarker behind PoseKit's
/// `PoseProvider` protocol. The 33 BlazePose GHUM landmarks are mapped to
/// the canonical 18-joint set in `mapToCanonical(_:)`. Two joints are
/// derived rather than directly read:
/// - `.head` from the NOSE landmark (BlazePose has no explicit head joint;
///   nose is the closest single-point proxy and is the convention used
///   by the Python pipeline at `FollowThrough/source/SkeletonMaker.py`).
/// - `.neck` from the midpoint of L_SHOULDER and R_SHOULDER (also no native
///   neck landmark; midpoint is the standard derivation).
///
/// A future provider (custom CoreML basketball-trained model, or a SOTA
/// MediaPipe successor) may have native HEAD and NECK keypoints; that
/// provider's mapping won't need these derivations.
public final class MediaPipePoseProvider: PoseProvider, @unchecked Sendable {

    public enum ModelVariant: String, Sendable {
        case lite, full, heavy

        var bundledFilename: String {
            switch self {
            case .lite: return "pose_landmarker_lite"
            case .full: return "pose_landmarker_full"
            case .heavy: return "pose_landmarker_heavy"
            }
        }
    }

    private let landmarker: PoseLandmarker

    /// - Parameters:
    ///   - variant: which BlazePose model variant to load. `.heavy` is the
    ///     v1 production default per the 2026-04 evaluation
    ///     (`design/pose-model-evaluation-2026-04.md`).
    ///   - bundle: which Bundle to look in for the model `.task` file.
    ///     Defaults to `.main`, which is correct for app-side usage. Pass
    ///     a test bundle for unit tests.
    public init(variant: ModelVariant = .heavy, bundle: Bundle = .main) throws {
        let candidatePaths: [String?] = [
            bundle.path(forResource: variant.bundledFilename, ofType: "task",
                        inDirectory: "Resources/models"),
            bundle.path(forResource: variant.bundledFilename, ofType: "task",
                        inDirectory: "models"),
            bundle.path(forResource: variant.bundledFilename, ofType: "task")
        ]
        guard let modelPath = candidatePaths.compactMap({ $0 }).first else {
            throw PoseProviderError.modelMissing(
                "\(variant.bundledFilename).task not found in bundle \(bundle.bundlePath)"
            )
        }

        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .video
        options.numPoses = 1

        do {
            self.landmarker = try PoseLandmarker(options: options)
        } catch {
            throw PoseProviderError.providerSpecific(
                "PoseLandmarker init failed: \(error.localizedDescription)"
            )
        }
    }

    public func detect(pixelBuffer: CVPixelBuffer, timestampMs: Int) async throws -> PoseFrame? {
        let mpImage: MPImage
        do {
            mpImage = try MPImage(pixelBuffer: pixelBuffer)
        } catch {
            throw PoseProviderError.providerSpecific(
                "MPImage construction failed: \(error.localizedDescription)"
            )
        }

        let result: PoseLandmarkerResult
        do {
            result = try landmarker.detect(videoFrame: mpImage,
                                           timestampInMilliseconds: timestampMs)
        } catch {
            throw PoseProviderError.providerSpecific(
                "PoseLandmarker.detect failed: \(error.localizedDescription)"
            )
        }

        guard let firstPose = result.landmarks.first, !firstPose.isEmpty else {
            return nil
        }

        return PoseFrame(
            timestampMs: timestampMs,
            keypoints: Self.mapToCanonical(firstPose)
        )
    }

    /// Map MediaPipe's 33 BlazePose landmarks to the canonical 18-joint set.
    /// Indices match BlazePose GHUM topology:
    /// 0=NOSE, 11=L_SHOULDER, 12=R_SHOULDER, 13=L_ELBOW, 14=R_ELBOW,
    /// 15=L_WRIST, 16=R_WRIST, 23=L_HIP, 24=R_HIP, 25=L_KNEE, 26=R_KNEE,
    /// 27=L_ANKLE, 28=R_ANKLE, 29=L_HEEL, 30=R_HEEL, 31=L_FOOT_INDEX,
    /// 32=R_FOOT_INDEX. NECK and HEAD are derived; see class docs.
    static func mapToCanonical(_ landmarks: [NormalizedLandmark]) -> [CanonicalJoint: Keypoint] {
        guard landmarks.count >= 33 else { return [:] }

        func keypoint(_ index: Int) -> Keypoint {
            let lm = landmarks[index]
            return Keypoint(
                x: lm.x,
                y: lm.y,
                z: lm.z,
                visibility: lm.visibility?.floatValue ?? 0
            )
        }

        var result: [CanonicalJoint: Keypoint] = [
            .leftShoulder:  keypoint(11),
            .rightShoulder: keypoint(12),
            .leftElbow:     keypoint(13),
            .rightElbow:    keypoint(14),
            .leftWrist:     keypoint(15),
            .rightWrist:    keypoint(16),
            .leftHip:       keypoint(23),
            .rightHip:      keypoint(24),
            .leftKnee:      keypoint(25),
            .rightKnee:     keypoint(26),
            .leftAnkle:     keypoint(27),
            .rightAnkle:    keypoint(28),
            .leftHeel:      keypoint(29),
            .rightHeel:     keypoint(30),
            .leftFootIndex: keypoint(31),
            .rightFootIndex: keypoint(32)
        ]

        // HEAD := NOSE (BlazePose's closest single-point head proxy).
        result[.head] = keypoint(0)

        // NECK := midpoint of shoulders.
        let ls = landmarks[11]
        let rs = landmarks[12]
        let neckVisibility = min(
            ls.visibility?.floatValue ?? 0,
            rs.visibility?.floatValue ?? 0
        )
        result[.neck] = Keypoint(
            x: (ls.x + rs.x) / 2,
            y: (ls.y + rs.y) / 2,
            z: (ls.z + rs.z) / 2,
            visibility: neckVisibility
        )

        return result
    }
}
