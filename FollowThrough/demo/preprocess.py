"""Pre-process source clips into demo-ready artifacts.

Reads source MP4s, runs MediaPipe Pose via SkeletonMaker, and writes for each clip:

  data/<name>_overlay.mp4  -- the original frames with skeleton drawn on top
  data/<name>_keypoints.csv -- per-frame keypoint coordinates (x,y per joint)
  data/<name>_meta.json     -- fps, frame count, source file

Run once before `streamlit run app.py`:

    python3 FollowThrough/demo/preprocess.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2

DEMO_DIR = Path(__file__).resolve().parent
REPO = DEMO_DIR.parent.parent
DATA_DIR = DEMO_DIR / "data"          # keypoint CSVs + metadata (read server-side)
STATIC_DIR = DEMO_DIR / "static"      # overlay videos (served via HTTP to browser)
SOURCE_DIR = REPO / "FollowThrough" / "2kvids"

# Make SkeletonMaker importable
sys.path.insert(0, str(REPO / "FollowThrough" / "source"))
from SkeletonMaker import SkeletonMaker  # noqa: E402


CLIPS = [
    # (display_name, source_filename)
    ("shaq", "shaq1.mp4"),
    ("nash", "nash2.mp4"),
]


def process_clip(maker: SkeletonMaker, name: str, src: Path) -> None:
    """Write overlay video + keypoint CSV + metadata for one source clip."""
    if not src.exists():
        raise FileNotFoundError(f"Missing source: {src}")

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    has, first = cap.read()
    if not has:
        raise RuntimeError(f"Could not read frames: {src}")
    h, w = first.shape[:2]

    overlay_path = STATIC_DIR / f"{name}_overlay.mp4"
    csv_path = DATA_DIR / f"{name}_keypoints.csv"
    meta_path = DATA_DIR / f"{name}_meta.json"

    # avc1 = H.264 in MP4 -- playable by Chrome, Firefox, Safari, iOS/Android browsers.
    # mp4v (MPEG-4 Part 2) would be smaller but Chrome/Firefox won't play it.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {overlay_path}")

    # Joint column order: landmark_names
    joint_names = maker.landmark_names + ["NECK"]

    csv_fh = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_fh)
    header = ["frame"] + [f"{j}_{axis}" for j in joint_names for axis in ("x", "y")]
    csv_writer.writerow(header)

    buffers = maker._init_buffers()
    frame_idx = 0
    # rewind to start -- we already consumed the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        has, frame = cap.read()
        if not has:
            break

        raw_points, neck_point = maker._extract_pose_data(frame)
        points = maker._smooth_points(raw_points, buffers) if maker.smoothing else raw_points
        maker._draw_pose(frame, points, neck_point)
        writer.write(frame)

        row: list[object] = [frame_idx]
        for pt in points:
            if pt is None:
                row.extend(["", ""])
            else:
                row.extend([pt[0], pt[1]])
        if neck_point is None:
            row.extend(["", ""])
        else:
            row.extend([neck_point[0], neck_point[1]])
        csv_writer.writerow(row)

        frame_idx += 1

    cap.release()
    writer.release()
    csv_fh.close()

    meta = {
        "source": str(src.relative_to(REPO)),
        "fps": fps,
        "frame_count": frame_idx,
        "width": w,
        "height": h,
        "joints": joint_names,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"  {name}: {frame_idx} frames @ {fps:.1f} fps -> {overlay_path.name}, {csv_path.name}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    maker = SkeletonMaker(model_complexity=2, smoothing=True)
    for name, fn in CLIPS:
        process_clip(maker, name, SOURCE_DIR / fn)
    print(f"Done. Videos in {STATIC_DIR.relative_to(REPO)}, keypoints in {DATA_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
