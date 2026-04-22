"""FollowThrough demo -- side-by-side shot-form analysis.

Run from the demo directory so Streamlit picks up `.streamlit/config.toml`
(which enables static file serving for the overlay videos):

    cd FollowThrough/demo
    /Users/Sameer/anaconda3/envs/eyeball/bin/streamlit run app.py

For phone viewing over Wi-Fi:

    cd FollowThrough/demo
    /Users/Sameer/anaconda3/envs/eyeball/bin/streamlit run app.py \\
        --server.address 0.0.0.0

Pre-process clips first (from repo root OR demo dir -- paths are absolute):

    /Users/Sameer/anaconda3/envs/eyeball/bin/python3 FollowThrough/demo/preprocess.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

DEMO = Path(__file__).resolve().parent
DATA = DEMO / "data"
STATIC = DEMO / "static"

# ---------- Data loading ----------

def load_clip(name: str) -> dict:
    kp = pd.read_csv(DATA / f"{name}_keypoints.csv")
    meta = json.loads((DATA / f"{name}_meta.json").read_text())
    return {
        "name": name,
        # Relative filename only -- the iframe builds the full URL at runtime using
        # `window.location.hostname` and the sidecar-server port, so this works both
        # on localhost and when accessed from a phone over LAN.
        "video_filename": f"{name}_overlay.mp4",
        "keypoints": kp,
        "meta": meta,
    }


def normalize_keypoints(kp: pd.DataFrame, joints: list[str]) -> pd.DataFrame:
    """Translate by hip midpoint, scale by shoulder-hip distance.

    Makes two shots comparable regardless of camera distance and player height.
    Returns a copy with _x/_y columns overwritten in normalised space.
    """
    out = kp.copy()
    hip_cx = (out["L_HIP_x"] + out["R_HIP_x"]) / 2
    hip_cy = (out["L_HIP_y"] + out["R_HIP_y"]) / 2
    sho_cx = (out["L_SHOULDER_x"] + out["R_SHOULDER_x"]) / 2
    sho_cy = (out["L_SHOULDER_y"] + out["R_SHOULDER_y"]) / 2
    scale = np.sqrt((sho_cx - hip_cx) ** 2 + (sho_cy - hip_cy) ** 2).replace(0, np.nan)

    for j in joints:
        xc, yc = f"{j}_x", f"{j}_y"
        if xc in out.columns and yc in out.columns:
            out[xc] = (out[xc] - hip_cx) / scale
            out[yc] = (out[yc] - hip_cy) / scale
    return out


def per_joint_deviation(a: pd.DataFrame, b: pd.DataFrame, joints: list[str]) -> pd.DataFrame:
    """Resample both shots to a common length then compute per-joint Euclidean distance."""
    N = 60  # common frame count
    rows = []
    for j in joints:
        xc, yc = f"{j}_x", f"{j}_y"
        if xc not in a.columns or xc not in b.columns:
            continue

        def resample(series: pd.Series) -> np.ndarray:
            s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            mask = ~np.isnan(s)
            if mask.sum() < 2:
                return np.full(N, np.nan)
            idx = np.linspace(0, mask.sum() - 1, N)
            return np.interp(idx, np.arange(mask.sum()), s[mask])

        ax = resample(a[xc])
        ay = resample(a[yc])
        bx = resample(b[xc])
        by = resample(b[yc])
        dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        rows.append({"joint": j, "mean_deviation": float(np.nanmean(dist))})
    return pd.DataFrame(rows).sort_values("mean_deviation", ascending=False).reset_index(drop=True)


JOINT_LABELS = {
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


def similarity_score(dev_df: pd.DataFrame) -> int:
    """Aggregate per-joint deviation into a 0--99 similarity percentage."""
    mean_dev = float(dev_df["mean_deviation"].mean())
    # Heuristic scale: 0.0 -> 100%, 0.5 -> 0%, linear.
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
    dev_df: pd.DataFrame,
    reference_name: str,
    top_n: int = 3,
) -> list[str]:
    """Produce coaching advice for the target shooter, referenced to the reference shooter."""
    advice: list[str] = []
    for _, row in dev_df.head(top_n).iterrows():
        joint = row["joint"]
        label = JOINT_LABELS.get(joint, joint.replace("_", " ").lower())
        tx = pd.to_numeric(target_norm[f"{joint}_x"], errors="coerce").mean()
        ty = pd.to_numeric(target_norm[f"{joint}_y"], errors="coerce").mean()
        rx = pd.to_numeric(reference_norm[f"{joint}_x"], errors="coerce").mean()
        ry = pd.to_numeric(reference_norm[f"{joint}_y"], errors="coerce").mean()

        # Pixel-y increases downward; smaller y = higher in the frame.
        dy = ty - ry
        dx = tx - rx

        # Prefer vertical guidance when it dominates, otherwise lateral.
        if abs(dy) >= abs(dx) and abs(dy) > 0.08:
            if dy > 0:
                advice.append(f"<strong>{label.capitalize()}</strong> sits lower than {reference_name}'s. Lift it higher through the shot.")
            else:
                advice.append(f"<strong>{label.capitalize()}</strong> is higher than {reference_name}'s. Relax the position slightly.")
        elif abs(dx) > 0.08:
            direction = "flared outward" if abs(tx) > abs(rx) else "tucked too close"
            fix = "tuck it closer to your body" if abs(tx) > abs(rx) else "let it extend outward naturally"
            advice.append(f"<strong>{label.capitalize()}</strong> is {direction} vs. {reference_name}'s. Try to {fix}.")
        else:
            advice.append(f"<strong>{label.capitalize()}</strong> shows timing differences vs. {reference_name}'s. Focus on matching the tempo.")

    return advice


def wrist_trajectories(clips: list[dict], joints: list[str]) -> pd.DataFrame:
    """Return a long-form DataFrame of shooting-wrist y-over-time for each clip."""
    # Pick the wrist with greater vertical range -- that's the shooting wrist.
    frames = []
    for c in clips:
        kp = c["keypoints"]
        def y_range(col):
            s = pd.to_numeric(kp[col], errors="coerce").dropna()
            return float(s.max() - s.min()) if len(s) else 0.0
        wrist = "R_WRIST" if y_range("R_WRIST_y") >= y_range("L_WRIST_y") else "L_WRIST"
        # Normalize time to [0, 1]
        t = np.linspace(0, 1, len(kp))
        y = pd.to_numeric(kp[f"{wrist}_y"], errors="coerce").to_numpy(dtype=float)
        # Invert y so "up" is up on the chart
        y = -(y - np.nanmean(y))
        frames.append(pd.DataFrame({"t": t, "wrist_y": y, "shooter": c["name"]}))
    return pd.concat(frames, ignore_index=True)


# ---------- UI ----------

st.set_page_config(
    page_title="FollowThrough -- Muse",
    page_icon=str(STATIC / "Muse_FT_EB.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      /* Hide Streamlit's floating top header that was clipping our logo */
      header[data-testid="stHeader"] {display: none !important;}
      #MainMenu {visibility: hidden;}
      .stApp > header {display: none;}
      /* Block container padding */
      .block-container {padding-top: 2.5rem !important; padding-bottom: 2rem; max-width: 1400px;}
      /* Header: flex row, image stays intrinsic size */
      .ft-header {display:flex; align-items:center; gap: 1.5rem; padding: 0.5rem 0;}
      .ft-header img {width: 140px; height: 140px; display:block; flex: 0 0 140px;}
      .ft-wordmark {font-weight: 800; font-size: 3rem; letter-spacing: -0.02em; line-height: 1.05; margin: 0; color: #f0f0f0;}
      .ft-tagline {color: #a0a0a0; font-size: 1.05rem; margin: 0.45rem 0 0 0;}
      .ft-divider {height: 3px; background: linear-gradient(90deg, #E87722 0%, #E87722 140px, #2a2a2c 140px, #2a2a2c 100%); margin: 1.25rem 0 1.5rem 0; border: none; border-radius: 2px;}
      /* Section headings */
      h3, h2 {margin-top: 0; color: #f0f0f0; font-weight: 700;}
      p, .stMarkdown p {color: #cfcfcf;}
      /* Section headings inside the metrics row */
      .stCaption, small {color: #8a8a8a !important;}
      /* Similarity card */
      .ft-sim-card {background: linear-gradient(135deg, #1a1a1c 0%, #232326 100%); border: 1px solid #2a2a2c; border-radius: 12px; padding: 24px 28px; margin: 1.5rem 0;}
      .ft-sim-header {display:flex; align-items:baseline; gap: 1rem; flex-wrap:wrap; margin-bottom: 1rem;}
      .ft-sim-score {font-size: 3.5rem; font-weight: 800; color: #E87722; line-height: 1; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;}
      .ft-sim-label {color: #8a8a8a; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700;}
      .ft-sim-descriptor {color: #f0f0f0; font-size: 1.1rem; font-weight: 600;}
      .ft-sim-advice-title {color: #E87722; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; margin: 1rem 0 0.6rem 0;}
      .ft-sim-advice {color: #e0e0e0; font-size: 1rem; line-height: 1.55; margin: 0; padding-left: 1.25rem;}
      .ft-sim-advice li {margin-bottom: 0.5rem;}
      .ft-sim-advice strong {color: #E87722; font-weight: 600;}
      /* Footer */
      .ft-footer {margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #2a2a2c; color: #7a7a7a; font-size: 0.85rem; text-align: center;}
      .ft-footer strong {color: #f0f0f0;}
      .ft-footer a {color: #E87722; text-decoration: none; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Branded header as a single HTML flex row so the logo renders at its intrinsic size
# (no Streamlit image container constraining it).
st.markdown(
    """
    <div class="ft-header">
      <img src="app/static/Muse_FT_EB.png" alt="Muse">
      <div>
        <h1 class="ft-wordmark">Follow Through</h1>
        <p class="ft-tagline">Analyze and improve shot mechanics via comparisons with NBA players.</p>
      </div>
    </div>
    <hr class="ft-divider">
    """,
    unsafe_allow_html=True,
)

shaq = load_clip("shaq")
nash = load_clip("nash")

# Dual-player component -- all playback controls live inside the iframe so they act
# in real time without round-tripping through Streamlit.
PLAYER_HTML = """
<!doctype html><html><head><meta charset="utf-8"><style>
  :root {
    --muse-orange: #E87722;
    --muse-orange-hi: #ff8f3c;
    --panel-bg: #121212;
    --panel-border: #222;
    --text-muted: #9a9a9a;
  }
  body {margin:0; padding:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#eee; background:transparent;}
  .grid {display:grid; grid-template-columns: 1fr 1fr; gap: 18px;}
  @media (max-width: 720px) { .grid {grid-template-columns: 1fr;} }
  .panel {background: var(--panel-bg); border: 1px solid var(--panel-border); border-radius: 10px; padding: 14px 14px 12px 14px;}
  .panel h3 {margin: 0 0 10px 0; font-size: 0.78rem; color: var(--muse-orange); letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700;}
  video {width:100%; background:#000; border-radius:6px; display:block; aspect-ratio: 16 / 9; object-fit: contain;}
  .row {display:flex; gap:10px; align-items:center; margin-top:10px; font-size:0.85rem;}
  .row label {color: var(--text-muted); min-width:60px; font-weight: 500;}
  .row input[type=range] {flex:1; accent-color: var(--muse-orange);}
  .controls {margin-top:18px; padding:14px; background: var(--panel-bg); border: 1px solid var(--panel-border); border-radius: 10px;}
  .btn {padding:9px 18px; background: var(--muse-orange); color:white; border:none; border-radius:6px; cursor:pointer; font-weight:600; font-size: 0.9rem; letter-spacing: 0.02em;}
  .btn:hover {background: var(--muse-orange-hi);}
  .btn.secondary {background: #2a2a2a; color: #ddd;}
  .btn.secondary:hover {background: #3a3a3a;}
  .speed-display {color: var(--muse-orange); font-weight:700; min-width:52px; text-align:right; font-variant-numeric: tabular-nums;}
</style></head><body>

<div class="grid">
  <div class="panel">
    <h3>Shaq</h3>
    <video id="v1" preload="auto" muted playsinline data-clip="__SHAQ__"></video>
    <div class="row"><label>Trim</label>
      <input type="range" id="v1_in" min="0" max="100" value="0" step="1">
      <input type="range" id="v1_out" min="0" max="100" value="100" step="1">
    </div>
  </div>
  <div class="panel">
    <h3>Nash</h3>
    <video id="v2" preload="auto" muted playsinline data-clip="__NASH__"></video>
    <div class="row"><label>Trim</label>
      <input type="range" id="v2_in" min="0" max="100" value="0" step="1">
      <input type="range" id="v2_out" min="0" max="100" value="100" step="1">
    </div>
  </div>
</div>

<div class="controls">
  <div class="row">
    <label>Speed</label>
    <input type="range" id="speed" min="0.1" max="2.0" step="0.05" value="0.5">
    <div class="speed-display" id="speed_val">0.50x</div>
  </div>
  <div class="row" style="margin-top:14px; gap:10px;">
    <button class="btn" id="sync">Sync &amp; Play</button>
    <button class="btn secondary" id="pause">Pause</button>
    <button class="btn secondary" id="reset">Reset</button>
  </div>
</div>

<script>
  // Build video URLs pointing at the sidecar server on port __VIDEO_PORT__.
  // Host priority: srcdoc iframes lose window.location, so fall back to the parent.
  const VIDEO_PORT = __VIDEO_PORT__;
  function resolveHost() {
    const h = window.location.hostname;
    if (h && h !== '') return h;
    try { const ph = window.parent.location.hostname; if (ph) return ph; } catch (e) {}
    return 'localhost';
  }
  const VIDEO_HOST = resolveHost();
  const videoUrl = (filename) => `http://${VIDEO_HOST}:${VIDEO_PORT}/${filename}`;

  const v1 = document.getElementById('v1');
  const v2 = document.getElementById('v2');
  v1.src = videoUrl(v1.dataset.clip);
  v2.src = videoUrl(v2.dataset.clip);
  const speed = document.getElementById('speed');
  const speedVal = document.getElementById('speed_val');
  const v1In = document.getElementById('v1_in');
  const v1Out = document.getElementById('v1_out');
  const v2In = document.getElementById('v2_in');
  const v2Out = document.getElementById('v2_out');

  function applySpeed() {
    const s = parseFloat(speed.value);
    v1.playbackRate = s;
    v2.playbackRate = s;
    speedVal.textContent = s.toFixed(2) + 'x';
  }
  speed.addEventListener('input', applySpeed);

  function trimStart(v, inSlider) {
    const d = v.duration || 0;
    return d * (parseFloat(inSlider.value) / 100);
  }
  function trimEnd(v, outSlider) {
    const d = v.duration || 0;
    return d * (parseFloat(outSlider.value) / 100);
  }
  function clampTime(v, inSlider, outSlider) {
    const start = trimStart(v, inSlider);
    const end = trimEnd(v, outSlider);
    if (v.currentTime < start) v.currentTime = start;
    if (v.currentTime > end) { v.pause(); v.currentTime = start; }
  }
  v1.addEventListener('timeupdate', () => clampTime(v1, v1In, v1Out));
  v2.addEventListener('timeupdate', () => clampTime(v2, v2In, v2Out));

  document.getElementById('sync').onclick = () => {
    applySpeed();
    v1.currentTime = trimStart(v1, v1In);
    v2.currentTime = trimStart(v2, v2In);
    Promise.all([v1.play(), v2.play()]).catch(() => {});
  };
  document.getElementById('pause').onclick = () => { v1.pause(); v2.pause(); };
  document.getElementById('reset').onclick = () => {
    v1.pause(); v2.pause();
    v1.currentTime = trimStart(v1, v1In);
    v2.currentTime = trimStart(v2, v2In);
  };
  applySpeed();
</script>
</body></html>
"""

VIDEO_PORT = 8502  # serve_videos.py default

html = (
    PLAYER_HTML
    .replace("__SHAQ__", shaq["video_filename"])
    .replace("__NASH__", nash["video_filename"])
    .replace("__VIDEO_PORT__", str(VIDEO_PORT))
)

components.html(html, height=620, scrolling=False)

# ---------- Analysis: compute once, used by similarity card and charts ----------

JOINTS_FOR_METRIC = list(JOINT_LABELS.keys())
shaq_norm = normalize_keypoints(shaq["keypoints"], JOINTS_FOR_METRIC)
nash_norm = normalize_keypoints(nash["keypoints"], JOINTS_FOR_METRIC)
dev = per_joint_deviation(shaq_norm, nash_norm, JOINTS_FOR_METRIC)
wrist_df = wrist_trajectories([shaq, nash], JOINTS_FOR_METRIC)

# Similarity card (target = left player Shaq, reference = right player Nash).
score = similarity_score(dev)
descriptor = similarity_descriptor(score)
advice = generate_advice(shaq_norm, nash_norm, dev, reference_name="Nash", top_n=3)
advice_html = "".join(f"<li>{a}</li>" for a in advice)

st.markdown(
    f"""
    <div class="ft-sim-card">
      <div class="ft-sim-header">
        <div>
          <div class="ft-sim-label">Form Similarity</div>
          <div class="ft-sim-score">{score}%</div>
        </div>
        <div class="ft-sim-descriptor">{descriptor}</div>
      </div>
      <div class="ft-sim-advice-title">To shoot more like Nash, Shaq should:</div>
      <ul class="ft-sim-advice">{advice_html}</ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Form analysis")

col_a, col_b = st.columns([2, 3], gap="large")

with col_a:
    st.markdown("**Per-joint deviation**")
    st.caption("Euclidean distance in normalised space between the two shooters, averaged over 60 aligned frames. Higher = more different form.")
    st.bar_chart(
        dev.set_index("joint")["mean_deviation"],
        height=320,
        color="#E87722",
    )

with col_b:
    st.markdown("**Shooting-wrist vertical motion**")
    st.caption("Inverted pixel-y (up is up) of the more-active wrist, time normalised to [0, 1]. Shows when the ball goes up and when it's released.")
    st.line_chart(
        wrist_df,
        x="t",
        y="wrist_y",
        color="shooter",
        height=320,
    )

with st.expander("Clip metadata"):
    st.write({"shaq": shaq["meta"], "nash": nash["meta"]})

# Subtle footer tying the demo back to the broader Muse system.
st.markdown(
    """
    <div class="ft-footer">
      FollowThrough -- part of the <strong>Muse</strong> basketball-analytics system.
      Pose extraction runs on-device via MediaPipe; no server dependency.
    </div>
    """,
    unsafe_allow_html=True,
)
