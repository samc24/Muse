"""Pre-process all Muse demo clips.

Thin CLI. The real work lives in pipeline.py.

Run from the demo directory:

    python3 preprocess.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from pipeline import detect_eb, preprocess_ft, render_broadcasting, transcode_eb

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parent
DATA = DEMO / "data"
STATIC = DEMO / "static"

# -- Follow Through ---------------------------------------------------------

FT_SOURCE_DIR = REPO / "FollowThrough" / "2kvids"
FT_CLIPS: list[tuple[str, str]] = [
    ("shaq", "shaq1.mp4"),
    ("nash", "nash2.mp4"),
]

# -- EyeBall ----------------------------------------------------------------

EB_SOURCE_DIR = REPO / "EyeBall" / "videos"

# Transcode-only: the pre-tracked overlay is historical best-case output.
EB_TRANSCODE_CLIPS: list[tuple[str, str]] = [
    ("jordan", "jordan3_track.avi"),
]

# Live detection path: each clip brings its own HSV / area / circularity tuning
# because classical CV is acutely sensitive to arena lighting and court colour.
EB_DETECT_CLIPS: list[tuple[str, str, int | None, dict]] = [
    (
        "houston",
        "2k_trim_1.mp4",
        14,  # seconds of the source clip to use
        {
            "hsv_lower": (8, 150, 60),
            "hsv_upper": (14, 230, 130),
            "area_lo": 20,
            "area_hi": 150,
            "min_circ": 0.60,
        },
    ),
]

# -- Broadcasting -----------------------------------------------------------
#
# The panoramic source isn't checked into the Muse repo (it's a ~280 MB capture
# that lives alongside the historic EyeBall-v2 iteration). Override with the
# MUSE_BROADCASTING_SOURCE env var if your copy is elsewhere; if the source
# isn't present we silently skip this stage so the other subsystems still run.

BROADCASTING_SOURCE = Path(
    os.environ.get(
        "MUSE_BROADCASTING_SOURCE",
        str(REPO.parent / "EyeBall-v2" / "videos_3_23" / "5.avi"),
    )
)

# Court ROI polygon in source-pixel coordinates. Motion outside this polygon
# is excluded so coaches, benches, and public can't pull the virtual pan off
# the action.
BROADCASTING_COURT_POLY = np.array([
    [270, 940], [1380, 1440], [3500, 1440], [4490, 985],
    [3475, 380], [3000, 300], [2385, 255], [1810, 280], [1310, 345],
], dtype=np.int32)

BROADCASTING_TRIM_SEC = 18


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)

    print("Follow Through:")
    sys.path.insert(0, str(REPO / "FollowThrough" / "source"))
    from SkeletonMaker import SkeletonMaker  # noqa: E402

    maker = SkeletonMaker(model_complexity=2, smoothing=True)
    preprocess_ft(FT_SOURCE_DIR, DATA, STATIC, FT_CLIPS, skeleton_maker=maker)

    print("\nEyeBall:")
    for name, filename in EB_TRANSCODE_CLIPS:
        transcode_eb(EB_SOURCE_DIR, DATA, STATIC, name, filename)
    for name, filename, trim, tuning in EB_DETECT_CLIPS:
        detect_eb(
            EB_SOURCE_DIR, DATA, STATIC, REPO,
            name, filename, tuning, trim_sec=trim,
        )

    print("\nBroadcasting:")
    if BROADCASTING_SOURCE.exists():
        render_broadcasting(
            BROADCASTING_SOURCE, STATIC, BROADCASTING_COURT_POLY,
            trim_sec=BROADCASTING_TRIM_SEC,
        )
    else:
        print(
            f"  Skipping -- source not found at {BROADCASTING_SOURCE}. "
            "Set MUSE_BROADCASTING_SOURCE if your copy is elsewhere."
        )

    print(
        f"\nDone. Videos in {STATIC.relative_to(REPO)}, "
        f"data in {DATA.relative_to(REPO)}"
    )


if __name__ == "__main__":
    main()
