"""Pure analysis functions for the Muse demo.

No Streamlit or Plotly imports -- these functions operate on pandas / numpy
and return primitive values or matplotlib figures so they are trivially testable
and reusable outside the Streamlit UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -- Follow Through ---------------------------------------------------------

#: Display labels for the joint codes used in the pose CSVs.
FT_JOINT_LABELS: Mapping[str, str] = {
    "R_WRIST": "shooting wrist",
    "L_WRIST": "guide-hand wrist",
    "R_ELBOW": "shooting elbow",
    "L_ELBOW": "guide-hand elbow",
    "R_SHOULDER": "shooting shoulder",
    "L_SHOULDER": "guide-hand shoulder",
    "R_HIP": "right hip",
    "L_HIP": "left hip",
    "R_KNEE": "right knee",
    "L_KNEE": "left knee",
    "R_ANKLE": "right ankle",
    "L_ANKLE": "left ankle",
}

# Frame-alignment length for per-joint deviation: both shooters are
# resampled to this many frames so their full shot motions are comparable.
_DEVIATION_FRAMES = 60

# Thresholds for coaching advice (in normalised units, where 1.0 is
# shoulder-to-hip distance).
_ADVICE_AXIS_THRESHOLD = 0.08


def normalize_keypoints(keypoints: pd.DataFrame, joints: Sequence[str]) -> pd.DataFrame:
    """Centre on hip-midpoint and scale by shoulder-hip distance.

    Makes poses from different cameras / body sizes comparable.
    """
    out = keypoints.copy()
    hip_cx = (out["L_HIP_x"] + out["R_HIP_x"]) / 2
    hip_cy = (out["L_HIP_y"] + out["R_HIP_y"]) / 2
    sho_cx = (out["L_SHOULDER_x"] + out["R_SHOULDER_x"]) / 2
    sho_cy = (out["L_SHOULDER_y"] + out["R_SHOULDER_y"]) / 2
    scale = np.sqrt((sho_cx - hip_cx) ** 2 + (sho_cy - hip_cy) ** 2).replace(0, np.nan)

    for joint in joints:
        xc, yc = f"{joint}_x", f"{joint}_y"
        if xc not in out.columns:
            continue
        out[xc] = (out[xc] - hip_cx) / scale
        out[yc] = (out[yc] - hip_cy) / scale
    return out


def per_joint_deviation(
    shooter_a: pd.DataFrame,
    shooter_b: pd.DataFrame,
    joints: Sequence[str],
) -> pd.DataFrame:
    """Euclidean distance between two normalised shooters, per joint, averaged
    across a common 60-frame time axis.

    Returns a DataFrame sorted by mean_deviation descending.
    """
    rows: list[dict[str, float | str]] = []
    for joint in joints:
        xc, yc = f"{joint}_x", f"{joint}_y"
        if xc not in shooter_a.columns or xc not in shooter_b.columns:
            continue
        ax = _resample(shooter_a[xc])
        ay = _resample(shooter_a[yc])
        bx = _resample(shooter_b[xc])
        by = _resample(shooter_b[yc])
        dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        rows.append({"joint": joint, "mean_deviation": float(np.nanmean(dist))})
    return pd.DataFrame(rows).sort_values("mean_deviation", ascending=False).reset_index(drop=True)


def _resample(series: pd.Series) -> np.ndarray:
    """Resample a (possibly sparse) 1D series to `_DEVIATION_FRAMES` points."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.full(_DEVIATION_FRAMES, np.nan)
    idx = np.linspace(0, mask.sum() - 1, _DEVIATION_FRAMES)
    return np.interp(idx, np.arange(mask.sum()), values[mask])


def wrist_trajectories(clips: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Stack shooting-wrist vertical trajectories from each clip into one long
    DataFrame (columns: t, wrist_y, shooter).

    Picks whichever wrist has the larger vertical range -- that's the shooting
    wrist regardless of handedness.
    """
    frames: list[pd.DataFrame] = []
    for clip in clips:
        kp: pd.DataFrame = clip["keypoints"]  # type: ignore[assignment]
        r_range = _vertical_range(kp, "R_WRIST_y")
        l_range = _vertical_range(kp, "L_WRIST_y")
        wrist = "R_WRIST" if r_range >= l_range else "L_WRIST"

        t = np.linspace(0, 1, len(kp))
        y = pd.to_numeric(kp[f"{wrist}_y"], errors="coerce").to_numpy(dtype=float)
        y = -(y - np.nanmean(y))  # invert so "up" is up; centre on the mean
        frames.append(pd.DataFrame({
            "t": t,
            "wrist_y": y,
            "shooter": str(clip["name"]).capitalize(),
        }))
    return pd.concat(frames, ignore_index=True)


def _vertical_range(kp: pd.DataFrame, col: str) -> float:
    s = pd.to_numeric(kp[col], errors="coerce").dropna()
    return float(s.max() - s.min()) if len(s) else 0.0


def similarity_score(deviation: pd.DataFrame) -> int:
    """Convert per-joint deviation into a 0-99 similarity percentage.

    Heuristic: mean_dev of 0.0 -> 100%, mean_dev of 0.5 -> 0%, linear.
    Capped at 99 to avoid an unrealistic perfect-match claim.
    """
    mean_dev = float(deviation["mean_deviation"].mean())
    return max(0, min(99, round(100 * (1 - mean_dev / 0.5))))


def similarity_descriptor(score: int) -> str:
    if score >= 80:
        return "Very similar form"
    if score >= 60:
        return "Similar in most mechanics"
    if score >= 40:
        return "Noticeably different mechanics"
    return "Very different form"


def generate_advice(
    target_norm: pd.DataFrame,
    reference_norm: pd.DataFrame,
    deviation: pd.DataFrame,
    reference_name: str,
    top_n: int = 3,
) -> list[str]:
    """Produce short coaching lines for the top-N most-deviated joints.

    Each line describes how the `target` differs from the `reference` and what
    to do about it. HTML-safe: wraps joint names in <strong>.
    """
    advice: list[str] = []
    for _, row in deviation.head(top_n).iterrows():
        joint = str(row["joint"])
        label = FT_JOINT_LABELS.get(joint, joint.replace("_", " ").lower())
        tx, ty = _mean_xy(target_norm, joint)
        rx, ry = _mean_xy(reference_norm, joint)
        dy, dx = ty - ry, tx - rx

        if abs(dy) >= abs(dx) and abs(dy) > _ADVICE_AXIS_THRESHOLD:
            advice.append(_vertical_advice(label, dy, reference_name))
        elif abs(dx) > _ADVICE_AXIS_THRESHOLD:
            advice.append(_lateral_advice(label, tx, rx, reference_name))
        else:
            advice.append(
                f"<strong>{label.capitalize()}</strong> shows timing differences vs. "
                f"{reference_name}'s."
            )
    return advice


def _mean_xy(kp: pd.DataFrame, joint: str) -> tuple[float, float]:
    x = pd.to_numeric(kp[f"{joint}_x"], errors="coerce").mean()
    y = pd.to_numeric(kp[f"{joint}_y"], errors="coerce").mean()
    return float(x), float(y)


def _vertical_advice(label: str, dy: float, reference_name: str) -> str:
    if dy > 0:
        return (
            f"<strong>{label.capitalize()}</strong> sits lower than "
            f"{reference_name}'s. Lift it higher through the shot."
        )
    return (
        f"<strong>{label.capitalize()}</strong> is higher than "
        f"{reference_name}'s. Relax the position slightly."
    )


def _lateral_advice(label: str, tx: float, rx: float, reference_name: str) -> str:
    flared = abs(tx) > abs(rx)
    direction = "flared outward" if flared else "tucked too close"
    fix = "tuck it closer to your body" if flared else "let it extend outward naturally"
    return (
        f"<strong>{label.capitalize()}</strong> is {direction} vs. "
        f"{reference_name}'s. Try to {fix}."
    )


# -- EyeBall ----------------------------------------------------------------

def trajectory_figure(
    trajectory: pd.DataFrame,
    events: Mapping[str, object],
    meta: Mapping[str, object],
) -> plt.Figure:
    """Matplotlib figure of the ball path across the court, with passes starred.

    Uses pixel-space axes (origin top-left) so the plot matches the camera view.
    """
    width = int(meta["width"])  # type: ignore[arg-type]
    height = int(meta["height"])  # type: ignore[arg-type]

    df = trajectory.copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

    pass_frames = {int(f) for f in events.get("pass_frames", [])}  # type: ignore[arg-type]
    pass_rows = df[df["frame"].isin(pass_frames)]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0e0e10")
    ax.set_facecolor("#14141a")

    if not df.empty:
        ax.plot(df["x"], df["y"], color="#E87722", linewidth=2.5, alpha=0.9, label="Ball path")
        ax.scatter(df["x"].iloc[0], df["y"].iloc[0], s=180, color="#4ade80",
                   edgecolors="white", linewidth=1.5, zorder=5, label="Start")
        ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], s=180, color="#f87171",
                   edgecolors="white", linewidth=1.5, zorder=5, label="End")
    if not pass_rows.empty:
        ax.scatter(pass_rows["x"], pass_rows["y"], s=260, color="#E87722", marker="*",
                   edgecolors="white", linewidth=1.5, zorder=6, label="Pass")

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # pixel-space: origin at top-left
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("#2a2a2c")
    ax.set_aspect("equal", adjustable="box")

    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4,
        frameon=False, labelcolor="#cfcfcf",
    )
    for text in legend.get_texts():
        text.set_color("#cfcfcf")

    fig.tight_layout()
    return fig


# -- Data loading (framework-agnostic) --------------------------------------

class ArtifactsMissingError(FileNotFoundError):
    """Raised when preprocessed artifacts for a clip aren't on disk."""


def load_ft_clip(data_dir: Path, name: str) -> dict:
    """Load a Follow Through clip's artifacts (keypoint CSV + meta JSON)."""
    kp_path = data_dir / f"{name}_keypoints.csv"
    meta_path = data_dir / f"{name}_meta.json"
    _require(kp_path, meta_path, clip=name, kind="Follow Through")
    return {
        "name": name,
        "video_filename": f"{name}_overlay.mp4",
        "keypoints": pd.read_csv(kp_path),
        "meta": json.loads(meta_path.read_text()),
    }


def load_eb_clip(data_dir: Path, name: str) -> dict:
    """Load an EyeBall clip's artifacts (trajectory CSV + events + meta)."""
    traj_path = data_dir / f"{name}_trajectory.csv"
    events_path = data_dir / f"{name}_events.json"
    meta_path = data_dir / f"{name}_meta.json"
    _require(traj_path, events_path, meta_path, clip=name, kind="EyeBall")
    return {
        "name": name,
        "video_filename": f"{name}_tracked.mp4",
        "trajectory": pd.read_csv(traj_path),
        "events": json.loads(events_path.read_text()),
        "meta": json.loads(meta_path.read_text()),
    }


def _require(*paths: Path, clip: str, kind: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise ArtifactsMissingError(
            f"{kind} clip '{clip}' is missing preprocessed artifacts: "
            + ", ".join(missing)
            + ". Run `python3 preprocess.py` from the demo directory first."
        )
