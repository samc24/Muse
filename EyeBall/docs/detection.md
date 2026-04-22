# Ball Detection — Design Rationale

EyeBall's detector is the front end of the entire play-detection pipeline: every downstream signal (trajectory, passes, possession boundaries, shot attempts) derives from it. Detection quality sets the ceiling for everything that follows.

This document records the method choices that got the detector to its current state and the failure modes each method was rejected for.

## What the detector has to handle

Any basketball detector is up against:

- **Colour confusion.** The ball is close in hue and saturation to many court surfaces and to skin tone under broadcast lighting. Any pure colour-based filter misfires against these regularly.
- **Occlusion.** Players palm, pass, and dribble the ball. A significant fraction of frames show only part of it, or none at all.
- **Scale variability.** The ball is tens of pixels wide in a broadcast-angle shot and single-digit pixels wide in a wide panoramic view.
- **Real-time pressure.** For the broadcasting application the detector has to keep pace with a 30 fps feed at HD resolution.

These constraints rule out the simpler methods that might otherwise be fine.

## Approaches evaluated

### Template matching — rejected

Cross-correlating against a fixed template image of the ball. Fails because the ball's appearance varies too much frame-to-frame and video-to-video (motion blur, occlusion by hand, lighting, spin). Can't scale across unseen games.

### Hough Circle Transform — rejected

Detecting circles directly via the Hough transform. Fails under occlusion (the ball isn't a full circle when gripped) and over-fires on any circle-shaped region in the scene — shoulder curves, face outlines, logos on the court.

### HSV filter + contour filtering + circularity + area — partially works

A multi-stage classical pipeline: Gaussian blur → HSV thresholding → binary image → morphological open/close → contour detection → filter contours by approximate circularity and by area band (known from camera height and zoom) → take the most circular remaining contour.

Works well when the ball is visible and the court contrasts cleanly with the ball hue. Fails on same-colour courts, under occlusion, and in motion blur. It's also brittle to changes in lighting and broadcast style — every new dataset requires re-tuning thresholds.

### YOLOv3 detector — current choice

A learned single-shot detector (`yolo_detection.py` wraps PyTorch-YOLOv3) trained to emit ball bounding boxes per frame. This is the current primary detector.

Why it wins:

- Robust to the colour-confusion failure mode — the detector learns the ball's full-appearance distribution, not just its hue.
- Handles partial occlusion gracefully — the learned representation doesn't require a full circle.
- Generalises across datasets without per-video threshold tuning.
- Throughput is acceptable on commodity GPUs for the broadcast application.

Cost: requires a local checkout of the upstream YOLOv3 repo and trained weights (see [`../yolo_detection.md`](../yolo_detection.md)), which means the setup is heavier than the classical pipeline.

## What the detector is not responsible for

- **Trajectory smoothing and occlusion bridging.** The Kalman filter (`KalmanFilter.py`) does this. The detector emits raw per-frame detections; the filter stitches them into a coherent track and survives brief misses.
- **Rejecting ball-coloured distractors.** The filter's distance-gated prediction handles this too.
- **Event detection (passes, possessions).** Downstream of the filtered trajectory.

Keeping these concerns separate is a deliberate design choice — it lets the detector stay stateless and frame-local, and makes each stage testable in isolation.

## Known gaps

- The detector currently runs on CPU / generic GPU; no optimisation for edge-class devices (Jetson, Movidius, etc.). Only relevant if the broadcast-application path gets productionised.
- No evaluation harness for the detector in isolation — quality is only measured end-to-end through pass-detection accuracy.
- Model weights are not committed to Muse; the upstream YOLOv3 setup has to be followed manually.

## Roadmap

- Add an isolated detector evaluation: precision/recall per frame against a labelled clip set, so detector changes can be measured without the full pipeline.
- Explore fine-tuning on basketball-specific data once enough labels exist — the generic YOLOv3 weights are fine but leave accuracy on the table.
- Decide whether to consolidate on a newer detector family (YOLOv8/v11) or stick with v3 — only worth the migration if the classification or latency benefit is concrete.
