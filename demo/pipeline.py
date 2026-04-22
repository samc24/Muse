"""Pre-processing pipelines for the Muse demo.

Two subsystems, three paths:

  preprocess_ft   -- MediaPipe pose extraction over Follow Through clips.
  transcode_eb    -- Transcode a pre-tracked AVI to browser-friendly H.264 MP4
                     (used for the Jordan clip, whose historical tracking was
                     dialled in at the time and is hard to reproduce live).
  detect_eb       -- Live classical-CV ball detection (HSV + contour +
                     circularity + Kalman), with two-pass Savitzky-Golay
                     smoothing and outlier rejection, then angle-based pass
                     detection on the smoothed trace.

All outputs go into `static/` (videos) and `data/` (CSVs + JSON), shared with
app.py so the demo is zero-friction after preprocess runs.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# -- EyeBall live-detection tuning ------------------------------------------

_SAVGOL_WINDOW = 7
_SAVGOL_POLYORDER = 2
_PASS_ANGLE_MAX_DEG = 75
_PASS_COOLDOWN_FRAMES = 20
_PASS_MIN_VECTOR_PX = 5
_PASS_MIN_WINDOW_MOTION_PX = 35
_PASS_MOTION_WINDOW = 15
_OUTLIER_MAX_SPEED_PX_PER_FRAME = 90
_DIST_THRESHOLD = 150
_TRACK_FRAMES = 60
_TRACE_LENGTH = 3500
_TRAIL_LENGTH = 60
_HUD_ORANGE_BGR = (34, 119, 232)
_H264_FOURCC = cv2.VideoWriter_fourcc(*"avc1")

# -- Broadcasting virtual-pan tuning ----------------------------------------

_PAN_SMOOTH_WINDOW = 31                   # Savitzky-Golay window on the pan target
_PAN_SMOOTH_POLYORDER = 2
_PAN_ACTIVE_COLUMN_FRAC = 0.15            # column counts as "active" when density >= frac of per-frame peak
_PAN_MOTION_THRESHOLD = 18                # binary-threshold cutoff on frame-diff blur
_PAN_BG_SAMPLE_COUNT = 20                 # frames used to build the median background
_PAN_OUTPUT_W = 1280                      # broadcast-output resolution
_PAN_OUTPUT_H = 720
_PAN_SOURCE_DISPLAY_W = 2048              # scaled-down panoramic for display
_PAN_SOURCE_DISPLAY_H = 576
_PAN_ZOOM_RATIO = 2.8                     # 1/zoom of panorama width per crop
_PAN_OVERLAY_COLOR_BGR = (0, 255, 127)    # green crop-window rectangle on the source
_PAN_OVERLAY_THICKNESS = 10


# ===========================================================================
# Follow Through -- MediaPipe pose pipeline
# ===========================================================================

def preprocess_ft(
    source_dir: Path,
    data_dir: Path,
    static_dir: Path,
    clips: Iterable[tuple[str, str]],
    skeleton_maker,
) -> None:
    """Process each clip with the given SkeletonMaker instance."""
    for name, source_filename in clips:
        src = source_dir / source_filename
        _require_file(src)
        _process_ft_clip(skeleton_maker, name, src, data_dir, static_dir)


def _process_ft_clip(
    maker,
    name: str,
    src: Path,
    data_dir: Path,
    static_dir: Path,
) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read first frame of {src}")
    height, width = first.shape[:2]

    overlay_path = static_dir / f"{name}_overlay.mp4"
    keypoints_path = data_dir / f"{name}_keypoints.csv"
    meta_path = data_dir / f"{name}_meta.json"

    writer = cv2.VideoWriter(str(overlay_path), _H264_FOURCC, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {overlay_path}")

    joint_names = maker.landmark_names + ["NECK"]
    with keypoints_path.open("w", newline="") as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(
            ["frame"] + [f"{j}_{axis}" for j in joint_names for axis in ("x", "y")]
        )

        buffers = maker._init_buffers()  # noqa: SLF001 -- internal API on the existing class
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        processed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_points, neck_point = maker._extract_pose_data(frame)  # noqa: SLF001
            points = (
                maker._smooth_points(raw_points, buffers) if maker.smoothing else raw_points  # noqa: SLF001
            )
            maker._draw_pose(frame, points, neck_point)  # noqa: SLF001
            writer.write(frame)

            row: list[object] = [processed]
            for pt in points:
                row.extend(["", ""] if pt is None else [pt[0], pt[1]])
            row.extend(["", ""] if neck_point is None else [neck_point[0], neck_point[1]])
            csv_writer.writerow(row)
            processed += 1

    cap.release()
    writer.release()

    meta_path.write_text(json.dumps({
        "source": str(src),
        "fps": fps,
        "frame_count": processed,
        "width": width,
        "height": height,
        "joints": joint_names,
    }, indent=2))

    print(f"  FT {name}: {processed} frames @ {fps:.1f} fps")


# ===========================================================================
# EyeBall -- transcode pre-tracked overlay
# ===========================================================================

def transcode_eb(
    source_dir: Path,
    data_dir: Path,
    static_dir: Path,
    name: str,
    source_filename: str,
) -> None:
    """Re-encode an existing tracked AVI to H.264 MP4 and recover the ball
    trajectory from the red marker drawn on each frame."""
    src = source_dir / source_filename
    _require_file(src)

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = static_dir / f"{name}_tracked.mp4"
    writer = cv2.VideoWriter(str(out_video), _H264_FOURCC, fps, (width, height))

    traj_rows: list[str] = ["frame,t_sec,x,y,detected"]
    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        centre = _extract_red_marker(frame)
        if centre is None:
            traj_rows.append(f"{frame_idx},{frame_idx / fps:.3f},,,0")
        else:
            traj_rows.append(f"{frame_idx},{frame_idx / fps:.3f},{centre[0]},{centre[1]},1")
        writer.write(frame)

    cap.release()
    writer.release()

    (data_dir / f"{name}_trajectory.csv").write_text("\n".join(traj_rows))
    events, distance = _events_from_trajectory_csv(data_dir / f"{name}_trajectory.csv", total_frames, fps)
    (data_dir / f"{name}_events.json").write_text(json.dumps(events, indent=2))
    (data_dir / f"{name}_meta.json").write_text(json.dumps({
        "source": str(src),
        "fps": fps,
        "frame_count": total_frames,
        "width": width,
        "height": height,
    }, indent=2))

    print(
        f"  EB {name} (transcoded): {total_frames} frames @ {fps:.1f} fps, "
        f"distance {distance:.0f}px"
    )


def _extract_red_marker(frame: np.ndarray) -> tuple[int, int] | None:
    """Locate the filled red ball marker drawn on a pre-tracked frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, np.array([0, 200, 150]), np.array([10, 255, 255]))
    hi = cv2.inRange(hsv, np.array([170, 200, 150]), np.array([180, 255, 255]))
    mask = lo | hi
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def _events_from_trajectory_csv(
    traj_path: Path, total_frames: int, fps: float,
) -> tuple[dict, float]:
    df = pd.read_csv(traj_path)
    detected = df.dropna(subset=["x", "y"])
    distance = 0.0
    prev: tuple[float, float] | None = None
    for _, row in detected.iterrows():
        if prev is not None:
            distance += float(np.hypot(row["x"] - prev[0], row["y"] - prev[1]))
        prev = (row["x"], row["y"])
    events = {
        "pass_count": 0,
        "pass_frames": [],
        "pass_timestamps": [],
        "detected_frames": int(len(detected)),
        "total_frames": total_frames,
        "detection_rate": round(len(detected) / total_frames, 3) if total_frames else 0.0,
        "trajectory_distance_px": round(distance, 1),
        "duration_sec": round(total_frames / fps, 3),
    }
    return events, distance


# ===========================================================================
# EyeBall -- live detection + tracking + pass detection
# ===========================================================================

def detect_eb(
    source_dir: Path,
    data_dir: Path,
    static_dir: Path,
    repo_root: Path,
    name: str,
    source_filename: str,
    tuning: dict,
    trim_sec: int | None,
) -> None:
    """Run live ball detection + two-pass smoothing + pass detection on a clip."""
    sys.path.insert(0, str(repo_root / "EyeBall"))
    from KalmanFilter import Tracker  # noqa: E402

    src = source_dir / source_filename
    _require_file(src)

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if trim_sec is not None:
        total_frames = min(total_frames, int(fps * trim_sec))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hsv_lower = np.array(tuning["hsv_lower"])
    hsv_upper = np.array(tuning["hsv_upper"])

    # Pass 1 -- detect per frame, feed Kalman, record both.
    tracker = Tracker(_DIST_THRESHOLD, _TRACK_FRAMES, _TRACE_LENGTH, 1)
    tracker_positions: list[tuple[int, int] | None] = []
    raw_detected: list[bool] = []
    for _ in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        centre = _detect_ball_in_frame(
            frame, hsv_lower, hsv_upper,
            tuning["area_lo"], tuning["area_hi"], tuning["min_circ"],
        )
        raw_detected.append(centre is not None)
        centres = (
            [np.array([[centre[0]], [centre[1]]], dtype=np.float32)] if centre else []
        )
        tracker.Update(centres)
        if tracker.tracks and tracker.tracks[0].trace:
            latest = tracker.tracks[0].trace[-1]
            tracker_positions.append((int(latest[0][0]), int(latest[1][0])))
        else:
            tracker_positions.append(None)

    positions = _smooth_trajectory(tracker_positions)
    pass_frames = _detect_passes(positions, fps)

    # Pass 2 -- re-render frames with the smoothed trail + running HUD.
    cap.release()
    cap = cv2.VideoCapture(str(src))
    out_video = static_dir / f"{name}_tracked.mp4"
    writer = cv2.VideoWriter(str(out_video), _H264_FOURCC, fps, (width, height))

    total_distance = 0.0
    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        lo = max(0, frame_idx - _TRAIL_LENGTH)
        for j in range(lo + 1, frame_idx + 1):
            cv2.line(frame, positions[j - 1], positions[j], (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(frame, positions[frame_idx], 9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, positions[frame_idx], 4, (0, 0, 255), -1, cv2.LINE_AA)
        running_passes = sum(1 for f in pass_frames if f <= frame_idx)
        hud = f"PASSES: {running_passes}   FRAME: {frame_idx + 1}/{total_frames}"
        cv2.rectangle(frame, (30, 20), (30 + 12 * len(hud), 60), (0, 0, 0), -1)
        cv2.putText(
            frame, hud, (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            _HUD_ORANGE_BGR, 2, cv2.LINE_AA,
        )
        writer.write(frame)

        if frame_idx > 0:
            total_distance += math.hypot(
                positions[frame_idx][0] - positions[frame_idx - 1][0],
                positions[frame_idx][1] - positions[frame_idx - 1][1],
            )

    cap.release()
    writer.release()

    _write_eb_artifacts(
        name, positions, raw_detected, pass_frames, total_frames, total_distance,
        fps, width, height, src, data_dir,
    )
    print(
        f"  EB {name} (live): {total_frames} frames @ {fps:.1f} fps, "
        f"{len(pass_frames)} passes, "
        f"det-rate {sum(raw_detected) / total_frames:.0%}, "
        f"distance {total_distance:.0f}px"
    )


def _detect_ball_in_frame(
    frame: np.ndarray,
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
    area_lo: int,
    area_hi: int,
    min_circ: float,
) -> tuple[int, int] | None:
    """Return the centroid of the most-circular basketball-coloured blob, if any."""
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_circ = -1.0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (area_lo < area < area_hi):
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circ = (4 * math.pi * area) / (perim ** 2)
        if circ > best_circ:
            best_circ = circ
            best_cnt = cnt

    if best_cnt is None or best_circ < min_circ:
        return None
    moments = cv2.moments(best_cnt)
    if moments["m00"] == 0:
        return None
    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def _reject_outliers(
    raw: list[tuple[int, int] | None],
) -> list[tuple[int, int] | None]:
    """Drop positions that imply unphysical ball velocity between frames."""
    cleaned = list(raw)
    last_good: tuple[int, int] | None = None
    last_good_frame = -1
    for i, point in enumerate(cleaned):
        if point is None:
            continue
        if last_good is None:
            last_good, last_good_frame = point, i
            continue
        gap = i - last_good_frame
        max_jump = _OUTLIER_MAX_SPEED_PX_PER_FRAME * max(gap, 1)
        d = math.hypot(point[0] - last_good[0], point[1] - last_good[1])
        if d > max_jump:
            cleaned[i] = None
        else:
            last_good, last_good_frame = point, i
    return cleaned


def _smooth_trajectory(
    raw: list[tuple[int, int] | None],
) -> list[tuple[int, int]]:
    """Reject outliers, interpolate through gaps, apply Savitzky-Golay."""
    cleaned = _reject_outliers(raw)
    n = len(cleaned)
    xs = np.array([p[0] if p else np.nan for p in cleaned], dtype=float)
    ys = np.array([p[1] if p else np.nan for p in cleaned], dtype=float)

    mask = ~np.isnan(xs)
    if mask.sum() < 2:
        return [(0, 0)] * n

    idx = np.arange(n)
    xs = np.interp(idx, idx[mask], xs[mask])
    ys = np.interp(idx, idx[mask], ys[mask])
    if n >= _SAVGOL_WINDOW:
        xs = savgol_filter(xs, _SAVGOL_WINDOW, _SAVGOL_POLYORDER)
        ys = savgol_filter(ys, _SAVGOL_WINDOW, _SAVGOL_POLYORDER)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def _detect_passes(positions: list[tuple[int, int]], fps: float) -> list[int]:
    """Angle-pivot pass detection on a smoothed trajectory."""
    passes: list[int] = []
    since_last = 10_000
    startup = int(fps)
    for i in range(1, len(positions) - 1):
        v_back = np.array(
            [positions[i - 1][0] - positions[i][0], positions[i - 1][1] - positions[i][1]],
            dtype=float,
        )
        v_fwd = np.array(
            [positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1]],
            dtype=float,
        )
        n_back = float(np.linalg.norm(v_back))
        n_fwd = float(np.linalg.norm(v_fwd))
        if n_back < _PASS_MIN_VECTOR_PX or n_fwd < _PASS_MIN_VECTOR_PX:
            since_last += 1
            startup = max(0, startup - 1)
            continue
        cos = float(np.dot(v_back, v_fwd) / (n_back * n_fwd))
        cos = max(-1.0, min(1.0, cos))
        angle_deg = math.degrees(math.acos(cos))
        sharp = angle_deg < _PASS_ANGLE_MAX_DEG

        lo = max(0, i - _PASS_MOTION_WINDOW // 2)
        hi = min(len(positions) - 1, i + _PASS_MOTION_WINDOW // 2)
        window_motion = sum(
            math.hypot(
                positions[j][0] - positions[j - 1][0],
                positions[j][1] - positions[j - 1][1],
            )
            for j in range(lo + 1, hi + 1)
        )
        enough_motion = window_motion >= _PASS_MIN_WINDOW_MOTION_PX

        if sharp and enough_motion and (since_last > _PASS_COOLDOWN_FRAMES or startup > 0):
            passes.append(i)
            since_last = 0
        else:
            since_last += 1
        startup = max(0, startup - 1)
    return passes


def _write_eb_artifacts(
    name: str,
    positions: list[tuple[int, int]],
    raw_detected: list[bool],
    pass_frames: list[int],
    total_frames: int,
    total_distance: float,
    fps: float,
    width: int,
    height: int,
    src: Path,
    data_dir: Path,
) -> None:
    with (data_dir / f"{name}_trajectory.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "t_sec", "x", "y", "detected"])
        for frame_idx, (pos, detected) in enumerate(zip(positions, raw_detected)):
            writer.writerow([
                frame_idx, round(frame_idx / fps, 3),
                pos[0], pos[1], int(detected),
            ])

    (data_dir / f"{name}_events.json").write_text(json.dumps({
        "pass_count": len(pass_frames),
        "pass_frames": pass_frames,
        "pass_timestamps": [round(f / fps, 3) for f in pass_frames],
        "detected_frames": sum(raw_detected),
        "total_frames": total_frames,
        "detection_rate": round(sum(raw_detected) / total_frames, 3) if total_frames else 0.0,
        "trajectory_distance_px": round(total_distance, 1),
        "duration_sec": round(total_frames / fps, 3),
    }, indent=2))
    (data_dir / f"{name}_meta.json").write_text(json.dumps({
        "source": str(src),
        "fps": fps,
        "frame_count": total_frames,
        "width": width,
        "height": height,
    }, indent=2))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")


# ===========================================================================
# Broadcasting -- virtual-pan visualisation over a panoramic source
# ===========================================================================

def render_broadcasting(
    source_video: Path,
    static_dir: Path,
    court_polygon: np.ndarray,
    trim_sec: int | None = None,
) -> None:
    """Render a broadcast-style 16:9 pan output and an annotated source clip.

    Pass 1: motion detection inside the court ROI. For each frame, pick the pan
    target as the midpoint of the "active column" envelope (leftmost and
    rightmost columns whose motion density >= a fraction of the per-frame
    peak). Smooth the target trace with Savitzky-Golay so the virtual camera
    glides rather than jerks.

    Pass 2: render two outputs from the same source:
      - broadcasting_source.mp4  -- scaled-down panorama with a green rectangle
        showing the crop window per frame (so the pan decision is legible).
      - broadcasting_output.mp4  -- broadcast-style 16:9 crop, resampled to
        1280x720.

    Both share frame rate + duration so they can be played back in sync.
    """
    _require_file(source_video)

    cap = cv2.VideoCapture(str(source_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if trim_sec is not None:
        total_frames = min(total_frames, int(fps * trim_sec))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_mask = np.zeros((src_h, src_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [court_polygon], 255)

    bg_gray = _build_background(cap, total_frames, _PAN_BG_SAMPLE_COUNT)
    targets = _compute_pan_targets(cap, bg_gray, roi_mask, total_frames, src_w)
    crop_w, crop_h, crop_y = _pan_crop_geometry(src_w, src_h)

    cap.release()
    cap = cv2.VideoCapture(str(source_video))
    out_path = static_dir / "broadcasting_output.mp4"
    src_path = static_dir / "broadcasting_source.mp4"
    out_path.unlink(missing_ok=True)
    src_path.unlink(missing_ok=True)

    output_writer = cv2.VideoWriter(str(out_path), _H264_FOURCC, fps, (_PAN_OUTPUT_W, _PAN_OUTPUT_H))
    source_writer = cv2.VideoWriter(
        str(src_path), _H264_FOURCC, fps, (_PAN_SOURCE_DISPLAY_W, _PAN_SOURCE_DISPLAY_H),
    )
    if not output_writer.isOpened() or not source_writer.isOpened():
        raise RuntimeError(f"Failed to open broadcasting video writers under {static_dir}")

    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        target_x = int(targets[frame_idx])
        x0 = max(0, min(src_w - crop_w, target_x - crop_w // 2))

        crop = frame[crop_y:crop_y + crop_h, x0:x0 + crop_w]
        output_writer.write(
            cv2.resize(crop, (_PAN_OUTPUT_W, _PAN_OUTPUT_H), interpolation=cv2.INTER_AREA),
        )

        annotated = frame.copy()
        cv2.rectangle(
            annotated, (x0, crop_y), (x0 + crop_w, crop_y + crop_h),
            _PAN_OVERLAY_COLOR_BGR, _PAN_OVERLAY_THICKNESS,
        )
        cv2.line(
            annotated, (target_x, crop_y), (target_x, crop_y + crop_h),
            _PAN_OVERLAY_COLOR_BGR, 3,
        )
        source_writer.write(
            cv2.resize(
                annotated, (_PAN_SOURCE_DISPLAY_W, _PAN_SOURCE_DISPLAY_H),
                interpolation=cv2.INTER_AREA,
            ),
        )

    cap.release()
    output_writer.release()
    source_writer.release()

    print(
        f"  Broadcasting: {total_frames} frames @ {fps:.1f} fps, "
        f"zoom ~{src_w / crop_w:.1f}x, "
        f"pan range x={int(targets.min())}-{int(targets.max())} of {src_w}"
    )


def _build_background(
    cap: cv2.VideoCapture, total_frames: int, sample_count: int,
) -> np.ndarray:
    """Median-average sampled frames into a greyscale background model."""
    sample_idx = np.linspace(0, max(0, total_frames - 1), sample_count).astype(int)
    samples: list[np.ndarray] = []
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            samples.append(frame)
    if not samples:
        raise RuntimeError("Could not sample any frames for background model")
    bg = np.median(samples, axis=0).astype(np.uint8)
    return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)


def _compute_pan_targets(
    cap: cv2.VideoCapture,
    bg_gray: np.ndarray,
    roi_mask: np.ndarray,
    total_frames: int,
    src_w: int,
) -> np.ndarray:
    """Per-frame pan target x (midpoint of the active-column envelope), Savgol-smoothed."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    raw_targets: list[float] = []
    for _ in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(bg_gray, gray)
        blurred = cv2.GaussianBlur(diff, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, _PAN_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=3)
        thresh = cv2.bitwise_and(thresh, roi_mask)

        col_density = np.sum(thresh, axis=0).astype(float)
        peak = col_density.max()
        if peak > 0:
            active = np.where(col_density >= peak * _PAN_ACTIVE_COLUMN_FRAC)[0]
            if len(active) > 0:
                raw_targets.append(float((active.min() + active.max()) / 2.0))
                continue
        raw_targets.append(raw_targets[-1] if raw_targets else src_w / 2.0)

    targets = np.array(raw_targets, dtype=float)
    if len(targets) >= _PAN_SMOOTH_WINDOW:
        targets = savgol_filter(targets, _PAN_SMOOTH_WINDOW, _PAN_SMOOTH_POLYORDER)
    return targets


def _pan_crop_geometry(src_w: int, src_h: int) -> tuple[int, int, int]:
    """Return the 16:9 crop window geometry (crop_w, crop_h, y-offset) inside the source frame."""
    crop_w = int(src_w / _PAN_ZOOM_RATIO)
    crop_h = int(crop_w * _PAN_OUTPUT_H / _PAN_OUTPUT_W)
    if crop_h > src_h:
        crop_h = src_h
        crop_w = int(crop_h * _PAN_OUTPUT_W / _PAN_OUTPUT_H)
    crop_y = (src_h - crop_h) // 2
    return crop_w, crop_h, crop_y
