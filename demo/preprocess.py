"""Pre-process all Muse demo clips.

Thin CLI. The real work lives in pipeline.py.

Run from the demo directory:

    python3 preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipeline import detect_eb, preprocess_ft, transcode_eb

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

    print(
        f"\nDone. Videos in {STATIC.relative_to(REPO)}, "
        f"data in {DATA.relative_to(REPO)}"
    )


if __name__ == "__main__":
    main()
