import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation

/// Iterate every frame of a video and run pose detection through a provider.
///
/// The AVAssetReader iteration, CMSampleBuffer ownership, and timestamp
/// extraction live here once. Every `PoseProvider` implementation gets the
/// same iteration semantics for free; provider tests and benchmarks share
/// this code path so per-provider perf comparisons are apples-to-apples.
///
/// - Parameters:
///   - videoURL: a file URL to a readable video asset.
///   - provider: any concrete `PoseProvider`.
/// - Returns: a `PoseExtractionResult` with one `PoseFrame` per video frame
///   that produced a detection (frames with no detection are dropped).
/// - Throws: `PoseProviderError.videoUnreadable` if the asset can't be
///   opened or has no video track. Re-throws any error from `provider.detect`.
public func extractPose(
    from videoURL: URL,
    using provider: PoseProvider
) async throws -> PoseExtractionResult {
    let asset = AVURLAsset(url: videoURL)
    let videoTracks = try await asset.loadTracks(withMediaType: .video)
    guard let track = videoTracks.first else {
        throw PoseProviderError.videoUnreadable("No video track in \(videoURL.lastPathComponent)")
    }

    let nominalFrameRate = try await track.load(.nominalFrameRate)
    let durationCMTime = try await asset.load(.duration)
    let videoDuration = CMTimeGetSeconds(durationCMTime)
    let naturalSize = try await track.load(.naturalSize)

    let reader: AVAssetReader
    do {
        reader = try AVAssetReader(asset: asset)
    } catch {
        throw PoseProviderError.videoUnreadable(error.localizedDescription)
    }
    let outputSettings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]
    let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
    reader.add(trackOutput)

    guard reader.startReading() else {
        let detail = reader.error?.localizedDescription ?? "AVAssetReader.startReading returned false"
        throw PoseProviderError.videoUnreadable(detail)
    }

    var frames: [PoseFrame] = []
    let started = Date()

    while let sampleBuffer = trackOutput.copyNextSampleBuffer() {
        defer { CMSampleBufferInvalidate(sampleBuffer) }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let timestampMs = Int(CMTimeGetSeconds(presentationTime) * 1000)

        if let frame = try await provider.detect(pixelBuffer: pixelBuffer, timestampMs: timestampMs) {
            frames.append(frame)
        }
    }

    if reader.status == .failed {
        let detail = reader.error?.localizedDescription ?? "AVAssetReader failed mid-extraction"
        throw PoseProviderError.videoUnreadable(detail)
    }

    let elapsed = Date().timeIntervalSince(started)
    return PoseExtractionResult(
        frames: frames,
        elapsedSeconds: elapsed,
        videoDurationSeconds: videoDuration,
        videoFps: Double(nominalFrameRate),
        imageSize: naturalSize
    )
}
