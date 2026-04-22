"""Unit tests for pure analysis functions.

Run:
    cd demo
    python3 -m pytest tests/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `analysis` importable whether pytest is run from demo/ or the repo root.
DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

import analysis  # noqa: E402


# -- Fixtures ---------------------------------------------------------------

def _synth_keypoints(joints: list[str], n_frames: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthesise a keypoint DataFrame with realistic column names."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {"frame": np.arange(n_frames)}
    for j in ["HEAD"] + joints + ["NECK"]:
        data[f"{j}_x"] = rng.integers(200, 800, n_frames)
        data[f"{j}_y"] = rng.integers(200, 800, n_frames)
    return pd.DataFrame(data)


@pytest.fixture
def ft_joints() -> list[str]:
    return list(analysis.FT_JOINT_LABELS.keys())


@pytest.fixture
def synthetic_pair(ft_joints):
    shaq = _synth_keypoints(ft_joints, seed=1)
    nash = _synth_keypoints(ft_joints, seed=2)
    return shaq, nash


# -- normalize_keypoints ----------------------------------------------------

def test_normalize_centres_on_hip_midpoint(ft_joints, synthetic_pair):
    shaq, _ = synthetic_pair
    out = analysis.normalize_keypoints(shaq, ft_joints)
    hip_mid_x = (out["L_HIP_x"] + out["R_HIP_x"]) / 2
    hip_mid_y = (out["L_HIP_y"] + out["R_HIP_y"]) / 2
    assert np.allclose(hip_mid_x, 0, atol=1e-9)
    assert np.allclose(hip_mid_y, 0, atol=1e-9)


# -- per_joint_deviation ----------------------------------------------------

def test_deviation_returns_one_row_per_joint(ft_joints, synthetic_pair):
    shaq, nash = synthetic_pair
    shaq_n = analysis.normalize_keypoints(shaq, ft_joints)
    nash_n = analysis.normalize_keypoints(nash, ft_joints)
    dev = analysis.per_joint_deviation(shaq_n, nash_n, ft_joints)
    assert set(dev["joint"]) <= set(ft_joints)
    assert (dev["mean_deviation"] >= 0).all()
    # Sorted descending
    assert list(dev["mean_deviation"]) == sorted(dev["mean_deviation"], reverse=True)


def test_deviation_zero_for_identical_shooters(ft_joints, synthetic_pair):
    shaq, _ = synthetic_pair
    shaq_n = analysis.normalize_keypoints(shaq, ft_joints)
    dev = analysis.per_joint_deviation(shaq_n, shaq_n, ft_joints)
    assert np.allclose(dev["mean_deviation"], 0)


# -- similarity_score -------------------------------------------------------

def test_similarity_score_bounds():
    low = pd.DataFrame({"joint": ["A"], "mean_deviation": [1e-6]})
    assert analysis.similarity_score(low) == 99  # capped below 100

    high = pd.DataFrame({"joint": ["A"], "mean_deviation": [5.0]})
    assert analysis.similarity_score(high) == 0  # clamped to 0


def test_similarity_descriptor_thresholds():
    assert "Very similar" in analysis.similarity_descriptor(85)
    assert "Similar in most" in analysis.similarity_descriptor(70)
    assert "Noticeably different" in analysis.similarity_descriptor(50)
    assert "Very different" in analysis.similarity_descriptor(20)


# -- generate_advice --------------------------------------------------------

def test_generate_advice_returns_top_n_lines(ft_joints, synthetic_pair):
    shaq, nash = synthetic_pair
    shaq_n = analysis.normalize_keypoints(shaq, ft_joints)
    nash_n = analysis.normalize_keypoints(nash, ft_joints)
    dev = analysis.per_joint_deviation(shaq_n, nash_n, ft_joints)
    lines = analysis.generate_advice(shaq_n, nash_n, dev, reference_name="Nash", top_n=3)
    assert len(lines) == 3
    # Each line is HTML-safe: wraps joint labels in <strong>
    for line in lines:
        assert "<strong>" in line
        assert "Nash" in line


# -- wrist_trajectories -----------------------------------------------------

def test_wrist_trajectories_picks_more_active_wrist(ft_joints):
    """Given a pose where the R_WRIST has more vertical motion, the R wrist
    trajectory should be the one returned."""
    n = 30
    kp = _synth_keypoints(ft_joints, n_frames=n, seed=0)
    # Force R_WRIST_y to swing widely, L_WRIST_y to stay flat
    kp["R_WRIST_y"] = np.linspace(200, 800, n).astype(int)
    kp["L_WRIST_y"] = np.full(n, 500, dtype=int)

    clip = {"name": "test", "keypoints": kp}
    traj = analysis.wrist_trajectories([clip])
    assert len(traj) == n
    # Non-trivial variance -> it chose the active wrist
    assert traj["wrist_y"].var() > 0


# -- load_ft_clip / load_eb_clip --------------------------------------------

def test_load_ft_missing_artifacts_raises(tmp_path: Path):
    with pytest.raises(analysis.ArtifactsMissingError):
        analysis.load_ft_clip(tmp_path, "does_not_exist")


def test_load_eb_missing_artifacts_raises(tmp_path: Path):
    with pytest.raises(analysis.ArtifactsMissingError):
        analysis.load_eb_clip(tmp_path, "does_not_exist")


def test_load_ft_round_trip(tmp_path: Path, ft_joints):
    kp = _synth_keypoints(ft_joints, n_frames=5)
    (tmp_path / "sam_keypoints.csv").write_text(kp.to_csv(index=False))
    (tmp_path / "sam_meta.json").write_text(json.dumps({"fps": 30, "frame_count": 5, "width": 640, "height": 480}))

    clip = analysis.load_ft_clip(tmp_path, "sam")
    assert clip["name"] == "sam"
    assert clip["video_filename"] == "sam_overlay.mp4"
    assert len(clip["keypoints"]) == 5
