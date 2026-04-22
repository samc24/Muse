# Broadcasting -- Application of EyeBall's Outputs

This doc describes one downstream application of EyeBall's ball-tracking and play-detection outputs: **automated broadcasting of amateur and semi-pro basketball games**. Broadcasting is not EyeBall's core mission -- play detection and tracking are -- but it's a useful concrete consumer of the pipeline's outputs, and the constraints it imposes shape some design choices upstream.

## The problem

Amateur basketball clubs don't have cost-effective ways to produce professional-looking broadcasts. Commercial AI-camera products cost several thousand dollars in hardware and additional thousands per year in licensing -- out of reach for most amateur leagues. A parent filming on a phone is the typical alternative. The gap is a "media gateway" that takes a panoramic feed from a commodity camera and produces a broadcast-quality cropped stream automatically, with no camera operator.

## What the system does

```
panoramic camera (RTSP/RTMP) → motion + ball tracking → virtual-pan decision → cropped 16:9 → YouTube/RTMP out
```

The panoramic feed covers the full court at 32:9 (or similar wide aspect). The output is a standard 16:9 broadcast-style video that appears to pan with the action. The pan is virtual -- implemented by shifting the crop window over the panoramic input, with no physical camera movement.

EyeBall contributes the ball-tracking input; play-detection outputs (possession state, play boundaries) inform the panning logic at a higher semantic level than raw motion.

## Virtual panning

Virtual panning means cropping a 16:9 window out of the 32:9 panoramic frame and moving that window frame-to-frame so the action stays centred. No tilt, no zoom -- pan-only.

The window movement needs to:

- keep the ball in frame most of the time,
- keep as many players in frame as practical,
- never move faster than a human viewer can tolerate.

## Game states

Broadcasting logic is gated by the semantic state of the game. The four states the panning logic needs to distinguish:

| State | Signal | Camera behaviour |
|---|---|---|
| **Static play** | All motion concentrated on one side of the court; camera already near that side | Hold the current view; reduce detection frequency to save compute |
| **Transition** | Motion spreading across court; ball trajectory crossing mid-line | Follow max-motion point; track ball when it exits the hot area |
| **Out of bounds** | Ball leaves area of interest or is momentarily lost | Hold last known position; apply transition rules once ball re-enters |
| **Not playing** (timeouts, halftime) | Motion distribution adopts a stable bi-modal shape over many seconds | Zoom out to full-court view; optionally insert pre-recorded content |

Detecting these states uses both the ball trajectory (from EyeBall) and a complementary motion signal -- a 1D projection of grayscale frame-differencing onto the horizontal axis, summarised as (leftmost, max, rightmost) motion points plus a bi-modal flag. The ball trajectory is authoritative when the ball is visible; the motion projection is the fallback.

## Framing rules

Two rules govern where the crop window should sit:

**The 1/3 rule.** In professional broadcasting the ball (or the lead actor of the play) should sit at the 1/3 or 2/3 vertical line of the frame. The pan follows this as the play moves.

**The hot area.** The middle 3/5ths of the crop window (from the 1/5 line to the 4/5 line) is the "hot area". As long as the ball stays in that band, the window follows the general motion mass slowly and smoothly. The moment the ball exits the hot area, the window snaps toward the ball at higher speed to recover it -- exit on the 1/5 side triggers a re-centring to the 1/5 line, 4/5 side triggers re-centring to 4/5.

If the ball is lost entirely -- occluded for more than the filter's patience -- the logic assumes it's still at the last tracked position and applies the rules against that.

## Pan speeds

Three speeds, each a tunable parameter:

- **V_safe** -- the normal pan speed, capped. Used when the ball is safely inside the hot area and the window is following the motion mass.
- **V_unsafe** -- faster, used when the ball has exited the hot area and needs to be recovered. Viewer-noticeable but acceptable.
- **V_critic** -- fastest, used when the ball is fully out of frame. Visibly aggressive; accepted because losing the ball is worse.

Plus a separate `Zoom speed` parameter for state transitions (e.g. in/out of timeout full-court view).

## Real-time constraints

Any broadcast application runs at least 30 fps on a 5k-pixel-wide panoramic feed. This has consequences that feed back into detector and tracker design:

- Detection must be fast enough to amortise across frames; dropping to every Nth frame is acceptable during static-play state.
- The tracker's prediction has to be reliable enough that the pipeline can skip detections and still render a moving crop -- otherwise the pan stutters.
- If the target hardware is edge-class (Jetson Nano, Movidius), detector choice is constrained to models that compile for those accelerators.

These constraints are why the pipeline is architected detector → tracker → consumer rather than putting detection directly in the crop decision loop -- the tracker absorbs per-frame detector latency and the crop decision runs off the tracker's smoothed output.

## What EyeBall exposes for this application

To make broadcasting a clean consumer of EyeBall's outputs, the pipeline's public surface needs to include:

- **Ball position** each frame (from tracker, not raw detector -- so it tolerates missed detections)
- **Ball confidence** -- tracked vs. predicted-during-occlusion so the broadcast layer can decide when to fall back to the motion projection
- **Play state** -- static / transition / out-of-bounds / not-playing, derived from trajectory + motion signal
- **Possession boundaries** -- the downstream play-detection primitive

The first two exist today; the last two are scoped as part of the play-detection work (see `../README.md` roadmap).

## Reference implementation: motion-based panning

A pragmatic alternative to full ball-tracking for the panning signal is a motion-detection approach that does not need the ball directly. It's the fallback path when the detector can't keep up, and the primary path for an MVP of the broadcasting application. EyeBall has a working reference implementation of this approach; the design decisions are worth recording.

- **Court ROI polygon.** Define the playable area as a polygon in camera coordinates (set at installation time). Filter motion contours by whether their ground line is inside the polygon -- this excludes coaches, benches, public, and referees moving *outside* the playing area, which would otherwise pull the pan target off the action.
- **Static background from median.** Sample N random frames at startup, take the per-pixel median, use it as a fixed background image. No live update -- covered amateur facilities have stable enough lighting that a one-shot calibration holds for a session, and live-updating introduces noise that costs more than it saves.
- **Downsampled motion detection.** Process motion on a ~1/10-scale version of the frame. Reduces noise, accelerates frame-differencing by two orders of magnitude. Contours detected at this scale map back to full resolution via the scale factor.
- **Frame differencing → threshold → dilation.** Grayscale absolute difference against the background, Gaussian blur, binary threshold, dilation to merge nearby blobs. Discard contours below an area floor (scaled by the downsample factor) to ignore twitches and noise.
- **Area-weighted centre of motion.** Compute a weighted mean of contour x-centres, weighted by area. Larger motion blobs pull the pan target harder than small ones. This is the target crop centre.
- **Thresholded pan smoothing.** Only move the crop window if the target differs from the current position by more than a pixel threshold (e.g. 200 px). Pan speed is proportional to delta / 30, with a minimum (amt > 10) to avoid micro-jitter. Below threshold, hold. This is what keeps the output from visibly shimmering when the motion mass barely moves.
- **Async frame capture.** Run the capture loop on a separate thread (e.g. `concurrent_videocapture`). This raises throughput from ~18 FPS to ~24 FPS on commodity hardware -- the difference between "lags real-time" and "tracks real-time".
- **Skip-factor for motion detection.** Run motion detection every Nth frame (e.g. every 8). Crop and write every frame, interpolating against the most recent target. Output looks smooth because only the target update is sparse; action doesn't change direction faster than 8 frames at 30 FPS.

The motion-detection path does not need EyeBall's ball tracker at all -- it works on player-mass motion alone, and makes a valid MVP broadcasting system. EyeBall's trajectory upgrades the panning quality where the ball is reliably visible (especially during transitions and fast breaks, where the ball is the clearest signal of where the action is going).

## Out of scope for EyeBall

- **Player tracking** -- the broadcasting logic uses a coarse 1D motion projection rather than per-player tracking. EyeBall is a ball tracker, not a player tracker.
- **Stream ingest/egress plumbing** -- RTSP decode, FFmpeg encode, RTMP push to YouTube. This is infrastructure for a broadcasting product, not the CV pipeline.
- **Camera calibration and dewarping** -- assumed handled by the camera itself or a thin pre-processing stage; not a concern of EyeBall.
