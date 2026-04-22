"""Muse demo -- unified Follow Through + EyeBall showcase.

A single Streamlit app with two outer tabs (Follow Through, EyeBall), each with
its own subsystem branding. Overlay videos are served by a sidecar HTTP server
on a separate port because Streamlit's built-in static handler refuses MP4 MIME.

Run from this directory:

    ./run_demo.sh

Or manually:

    python3 serve_videos.py --host 0.0.0.0 --port 8602 &
    streamlit run app.py --server.address 0.0.0.0 --server.port 8601

Re-generate overlay videos + keypoint/trajectory artifacts:

    python3 preprocess.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import analysis
from analysis import (
    ArtifactsMissingError,
    FT_JOINT_LABELS,
    generate_advice,
    normalize_keypoints,
    per_joint_deviation,
    similarity_descriptor,
    similarity_score,
    trajectory_figure,
    wrist_trajectories,
)

# -- Configuration ----------------------------------------------------------

DEMO = Path(__file__).resolve().parent
DATA = DEMO / "data"
STATIC = DEMO / "static"
TEMPLATES = DEMO / "templates"

#: Sidecar video server port. Keep in sync with run_demo.sh.
VIDEO_PORT = 8602

FT_JOINTS = list(FT_JOINT_LABELS.keys())


# -- Cached data loaders ----------------------------------------------------

@st.cache_data(show_spinner=False)
def load_ft_clip_cached(name: str) -> dict:
    return analysis.load_ft_clip(DATA, name)


@st.cache_data(show_spinner=False)
def load_eb_clip_cached(name: str) -> dict:
    return analysis.load_eb_clip(DATA, name)


@st.cache_data(show_spinner=False)
def load_template(name: str) -> str:
    return (TEMPLATES / name).read_text()


# -- Page + global CSS ------------------------------------------------------

st.set_page_config(
    page_title="Muse",
    page_icon=str(STATIC / "Muse_FT_EB.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

_GLOBAL_CSS = """
<style>
  /* Hide Streamlit's floating top chrome so our header reaches the viewport top. */
  header[data-testid="stHeader"] {display: none !important;}
  #MainMenu {visibility: hidden;}
  .stApp > header {display: none;}
  .block-container {padding-top: 1.5rem !important; padding-bottom: 2rem; max-width: 1400px;}

  /* Muse attribution strip */
  .muse-strip {display:flex; align-items:center; justify-content:space-between; gap: 1.5rem; padding: 0.25rem 0 1.25rem 0; border-bottom: 1px solid #222225; margin-bottom: 1.5rem;}
  .muse-strip-left {display:flex; align-items:center; gap: 18px;}
  .muse-strip img {width: 96px; height: 96px; display:block;}
  .muse-strip-name {color: #f0f0f0; font-weight: 900; letter-spacing: 0.02em; font-size: 2.6rem; line-height: 1;}
  .muse-strip-sub {color: #a0a0a0; font-size: 1rem; margin-top: 0.3rem; font-weight: 500;}
  .muse-strip-tag {color: #8a8a8a; font-size: 0.85rem; font-style: italic; text-align: right;}
  @media (max-width: 720px) {
    .muse-strip {flex-direction: column; align-items: flex-start;}
    .muse-strip-tag {text-align: left;}
  }

  /* Tab bar */
  .stTabs [role="tablist"] {gap: 10px; border-bottom: 1px solid #222225;}
  .stTabs [role="tab"] {
    background: transparent; color: #888; border: none;
    padding: 10px 18px; font-size: 0.95rem; font-weight: 700;
    letter-spacing: 0.04em; text-transform: uppercase;
  }
  .stTabs [role="tab"][aria-selected="true"] {
    color: #f0f0f0; border-bottom: 3px solid var(--tab-accent, #E87722);
  }

  /* Subsystem headers */
  .sub-header {display:flex; align-items:center; gap: 1.25rem; padding: 0.5rem 0 0.25rem 0;}
  .sub-header img.icon {width: 100px; height: 100px; display:block; flex: 0 0 100px; object-fit: contain;}
  /* The EyeBall wordmark PNG is black-on-transparent; invert it on the dark theme. */
  .sub-header img.wordmark {height: 78px; display:block; object-fit: contain; filter: invert(1);}
  .sub-wordmark-text {font-weight: 800; font-size: 2.5rem; letter-spacing: -0.02em; line-height: 1.05; margin: 0;}
  .sub-tagline {color: #a0a0a0; font-size: 1rem; margin: 0.3rem 0 0 0;}

  /* Follow Through -- NBA red/blue vibe */
  .ft-divider {height: 3px; background: linear-gradient(90deg, #C8102E 0%, #C8102E 70px, #1D428A 70px, #1D428A 140px, #2a2a2c 140px, #2a2a2c 100%); border: none; border-radius: 2px; margin: 0.75rem 0 1.5rem 0;}

  /* EyeBall -- basketball-orange vibe */
  .eb-divider {height: 3px; background: linear-gradient(90deg, #E87722 0%, #E87722 140px, #2a2a2c 140px, #2a2a2c 100%); border: none; border-radius: 2px; margin: 0.75rem 0 1.5rem 0;}

  h3, h2 {margin-top: 0; color: #f0f0f0; font-weight: 700;}
  p, .stMarkdown p {color: #cfcfcf;}

  /* Follow Through similarity card */
  .ft-sim-card {background: linear-gradient(135deg, #1a1a1c 0%, #232326 100%); border: 1px solid #2a2a2c; border-radius: 12px; padding: 24px 28px; margin: 1.5rem 0;}
  .ft-sim-header {display:flex; align-items:baseline; gap: 1rem; flex-wrap:wrap; margin-bottom: 1rem;}
  .ft-sim-score {font-size: 3.5rem; font-weight: 800; color: #C8102E; line-height: 1; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;}
  .ft-sim-label {color: #8a8a8a; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700;}
  .ft-sim-descriptor {color: #f0f0f0; font-size: 1.1rem; font-weight: 600;}
  .ft-sim-advice-title {color: #1D428A; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; margin: 1rem 0 0.6rem 0;}
  .ft-sim-advice {color: #e0e0e0; font-size: 1rem; line-height: 1.55; margin: 0; padding-left: 1.25rem;}
  .ft-sim-advice li {margin-bottom: 0.5rem;}
  .ft-sim-advice strong {color: #C8102E; font-weight: 600;}

  /* EyeBall stat cards */
  .eb-stats {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1.5rem 0;}
  @media (max-width: 780px) {.eb-stats {grid-template-columns: repeat(2, 1fr);}}
  .eb-stat-card {background: linear-gradient(135deg, #1a1a1c 0%, #232326 100%); border: 1px solid #2a2a2c; border-radius: 10px; padding: 16px 18px;}
  .eb-stat-label {color: #8a8a8a; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;}
  .eb-stat-value {color: #E87722; font-size: 2rem; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums;}
  .eb-stat-suffix {color: #888; font-size: 0.85rem; margin-left: 4px; font-weight: 500;}

  /* Footer */
  .muse-footer {margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #2a2a2c; color: #7a7a7a; font-size: 0.85rem; text-align: center;}
  .muse-footer strong {color: #f0f0f0;}
</style>
"""

st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# -- Muse attribution strip -------------------------------------------------

st.markdown(
    """
    <div class="muse-strip">
      <div class="muse-strip-left">
        <img src="app/static/Muse_FT_EB.png" alt="Muse">
        <div>
          <div class="muse-strip-name">Muse</div>
          <div class="muse-strip-sub">A basketball computer-vision system</div>
        </div>
      </div>
      <div class="muse-strip-tag">Bridging the eye-test and counting stats</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -- Tab renderers ----------------------------------------------------------

def _render_artifacts_missing(err: ArtifactsMissingError) -> None:
    st.error(str(err))
    st.info("Run `python3 preprocess.py` from this directory, then refresh.")


def render_follow_through() -> None:
    try:
        shaq = load_ft_clip_cached("shaq")
        nash = load_ft_clip_cached("nash")
    except ArtifactsMissingError as err:
        _render_artifacts_missing(err)
        return

    st.markdown(
        """
        <div class="sub-header">
          <img class="icon" src="app/static/followthrough_logo.png" alt="Follow Through">
          <div>
            <h1 class="sub-wordmark-text">Follow Through</h1>
            <p class="sub-tagline">Analyze and improve shot mechanics via comparisons with NBA players.</p>
          </div>
        </div>
        <hr class="ft-divider">
        """,
        unsafe_allow_html=True,
    )

    player_html = load_template("ft_player.html").replace("__VIDEO_PORT__", str(VIDEO_PORT))
    components.html(player_html, height=620, scrolling=False)

    shaq_norm = normalize_keypoints(shaq["keypoints"], FT_JOINTS)
    nash_norm = normalize_keypoints(nash["keypoints"], FT_JOINTS)
    deviation = per_joint_deviation(shaq_norm, nash_norm, FT_JOINTS)
    wrists = wrist_trajectories([shaq, nash])

    score = similarity_score(deviation)
    descriptor = similarity_descriptor(score)
    advice_html = "".join(
        f"<li>{line}</li>"
        for line in generate_advice(shaq_norm, nash_norm, deviation, reference_name="Nash")
    )

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
    col_left, col_right = st.columns([2, 3], gap="large")
    with col_left:
        st.markdown("**Per-joint deviation**")
        st.caption(
            "Euclidean distance in normalised space between the two shooters, "
            "averaged over 60 aligned frames."
        )
        st.bar_chart(
            deviation.set_index("joint")["mean_deviation"],
            height=320,
            color="#C8102E",
        )
    with col_right:
        st.markdown("**Shooting-wrist vertical motion**")
        st.caption(
            "Inverted pixel-y (up is up) of the more-active wrist across the shot."
        )
        st.line_chart(wrists, x="t", y="wrist_y", color="shooter", height=320)


def render_eyeball() -> None:
    st.markdown(
        """
        <div class="sub-header">
          <img class="wordmark" src="app/static/eyeball_wordmark.png" alt="EyeBall">
          <p class="sub-tagline" style="margin-left: 1rem;">Ball tracking and play-level analysis from raw game film.</p>
        </div>
        <hr class="eb-divider">
        """,
        unsafe_allow_html=True,
    )

    jordan_tab, houston_tab = st.tabs([
        "Jordan -- Drive to the Rim",
        "Spurs at Houston -- Half-Court",
    ])
    with jordan_tab:
        _render_eb_clip("jordan", "Jordan -- Drive to the Rim")
    with houston_tab:
        _render_eb_clip("houston", "Spurs at Houston -- Half-Court")


def _render_eb_clip(name: str, label: str) -> None:
    try:
        clip = load_eb_clip_cached(name)
    except ArtifactsMissingError as err:
        _render_artifacts_missing(err)
        return

    player_html = (
        load_template("eb_player.html")
        .replace("__CLIP__", clip["video_filename"])
        .replace("__LABEL__", label)
        .replace("__VIDEO_PORT__", str(VIDEO_PORT))
    )
    components.html(player_html, height=620, scrolling=False)

    events = clip["events"]
    st.markdown(
        f"""
        <div class="eb-stats">
          <div class="eb-stat-card">
            <div class="eb-stat-label">Passes Detected</div>
            <div class="eb-stat-value">{events['pass_count']}</div>
          </div>
          <div class="eb-stat-card">
            <div class="eb-stat-label">Play Duration</div>
            <div class="eb-stat-value">{events['duration_sec']:.1f}<span class="eb-stat-suffix">s</span></div>
          </div>
          <div class="eb-stat-card">
            <div class="eb-stat-label">Ball Distance</div>
            <div class="eb-stat-value">{events['trajectory_distance_px']:.0f}<span class="eb-stat-suffix">px</span></div>
          </div>
          <div class="eb-stat-card">
            <div class="eb-stat-label">Detection Rate</div>
            <div class="eb-stat-value">{events['detection_rate'] * 100:.0f}<span class="eb-stat-suffix">%</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Ball trajectory")
    st.caption(
        "Orange path shows the full ball track. Stars mark detected passes. "
        "Green = start, red = end."
    )
    fig = trajectory_figure(clip["trajectory"], events, clip["meta"])
    st.pyplot(fig, use_container_width=True)

    if events["pass_timestamps"]:
        times = ", ".join(f"{t:.2f}s" for t in events["pass_timestamps"])
        st.markdown(f"**Pass timeline:** {times}")


# -- Layout -----------------------------------------------------------------

tab_ft, tab_eb = st.tabs(["Follow Through", "EyeBall"])
with tab_ft:
    render_follow_through()
with tab_eb:
    render_eyeball()

st.markdown(
    """
    <div class="muse-footer">
      <strong>Muse</strong> -- basketball CV system. Follow Through: MediaPipe
      pose + Savitzky-Golay smoothing + per-joint deviation scoring.
      EyeBall: HSV + contour + Kalman + two-pass Savgol + angle-based pass detection.
      All on-device, no server dependency.
    </div>
    """,
    unsafe_allow_html=True,
)
