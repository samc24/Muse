import MediaPipeKit
import PoseKit
import SwiftUI

struct ContentView: View {
    @State private var status: ExtractionStatus = .idle

    var body: some View {
        VStack(spacing: 28) {
            VStack(spacing: 4) {
                Text("Follow Through")
                    .font(.system(size: 32, weight: .bold))
                Text("Sprint 2 — pose perf check")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 32)

            Spacer()

            statusView

            Spacer()

            VStack(spacing: 12) {
                Button(action: { run(clip: "shaq") }) {
                    label("Run pose on Shaq", running: status.isRunning)
                }
                .disabled(status.isRunning)

                Button(action: { run(clip: "nash") }) {
                    label("Run pose on Nash", running: status.isRunning)
                        .opacity(0.85)
                }
                .disabled(status.isRunning)
            }
            .padding(.horizontal)
            .padding(.bottom, 24)
        }
        .padding(.horizontal)
    }

    @ViewBuilder
    private var statusView: some View {
        switch status {
        case .idle:
            Text("Ready.")
                .font(.title3)
                .foregroundStyle(.secondary)
        case .running(let clip):
            VStack(spacing: 12) {
                ProgressView().controlSize(.large)
                Text("Extracting pose from \(clip).mp4…")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        case .success(let clip, let result):
            VStack(spacing: 14) {
                Text(clip)
                    .font(.headline)
                    .foregroundStyle(.secondary)
                    .textCase(.uppercase)

                Text("\(result.frames.count)")
                    .font(.system(size: 64, weight: .bold, design: .monospaced))
                Text("frames extracted")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack(spacing: 24) {
                    statTile(label: "Video", value: String(format: "%.1f fps", result.videoFps))
                    statTile(label: "Wall time", value: String(format: "%.2fs", result.elapsedSeconds))
                    statTile(label: "Processing", value: String(format: "%.1f fps", result.processingFps))
                }
            }
        case .failure(let clip, let message):
            VStack(spacing: 8) {
                Text("Failed on \(clip)")
                    .font(.headline)
                    .foregroundStyle(.red)
                Text(message)
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
        }
    }

    private func label(_ text: String, running: Bool) -> some View {
        Text(running ? "Running…" : text)
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(Color.accentColor)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func statTile(label: String, value: String) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.system(.body, design: .monospaced))
                .fontWeight(.semibold)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
        }
    }

    private func run(clip: String) {
        status = .running(clip: clip)
        Task {
            do {
                let url = try locateBundledClip(named: clip)
                let provider: PoseProvider = try MediaPipePoseProvider(variant: .heavy)
                let result = try await extractPose(from: url, using: provider)
                await MainActor.run { status = .success(clip: clip, result: result) }
            } catch {
                await MainActor.run { status = .failure(clip: clip, message: error.localizedDescription) }
            }
        }
    }

    private func locateBundledClip(named name: String) throws -> URL {
        let candidates: [URL?] = [
            Bundle.main.url(forResource: name, withExtension: "mp4", subdirectory: "Resources/pro_videos"),
            Bundle.main.url(forResource: name, withExtension: "mp4", subdirectory: "pro_videos"),
            Bundle.main.url(forResource: name, withExtension: "mp4")
        ]
        guard let url = candidates.compactMap({ $0 }).first else {
            throw NSError(
                domain: "FollowThrough",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "\(name).mp4 not found in bundle"]
            )
        }
        return url
    }
}

private enum ExtractionStatus {
    case idle
    case running(clip: String)
    case success(clip: String, result: PoseExtractionResult)
    case failure(clip: String, message: String)

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }
}

#Preview {
    ContentView()
}
