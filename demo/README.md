# Muse demo

A single Streamlit app showcasing both Muse subsystems side-by-side:

- **Follow Through** -- shot-form pose comparison (Shaq vs. Nash), with similarity score + coaching advice + per-joint deviation chart.
- **EyeBall** -- ball tracking + pass detection on basketball video, with trajectory plot and live stats.

Dark theme, NBA-style red/blue accents on Follow Through, basketball-orange accents on EyeBall.

## Quick start

```bash
./run_demo.sh
```

Opens:
- Streamlit app: <http://localhost:8601>
- Sidecar video server: <http://localhost:8602>

Both bind to `0.0.0.0` so the demo is viewable from a phone on the same Wi-Fi. Ctrl+C stops both.

To use a specific Python interpreter or different ports:

```bash
MUSE_PYTHON=/path/to/python3 ./run_demo.sh
APP_PORT=9000 VIDEO_PORT=9001 ./run_demo.sh
```

## Re-generate pre-processed clips

The demo loads pre-processed artifacts from `data/` (keypoint + trajectory CSVs, events + meta JSON) and overlay videos from `static/`. Regenerate them with:

```bash
python3 preprocess.py
```

This takes ~30-60 s depending on clip lengths. Source footage comes from `FollowThrough/2kvids/` (pose clips) and `EyeBall/videos/` (ball-tracking clips).

## Architecture

```
demo/
|-- app.py              # thin Streamlit entry, UI only
|-- analysis.py         # pure analysis functions (no Streamlit imports)
|-- pipeline.py         # pre-processing implementations (FT + EB)
|-- preprocess.py       # CLI orchestrator that calls pipeline.py
|-- serve_videos.py     # sidecar HTTP server (MP4 + CORS + Range)
|-- run_demo.sh         # launcher (Streamlit + sidecar)
|-- requirements.txt
|-- README.md
|-- .streamlit/
|   `-- config.toml     # dark theme + static serving for PNGs
|-- templates/
|   |-- ft_player.html  # Follow Through dual-video player
|   `-- eb_player.html  # EyeBall single-video player
|-- static/             # served by the sidecar (MP4s) + Streamlit (PNGs)
|   |-- *.mp4           # overlay videos
|   `-- *.png           # logos
|-- data/               # read by app.py
|   |-- *_keypoints.csv
|   |-- *_trajectory.csv
|   |-- *_events.json
|   `-- *_meta.json
`-- tests/
    `-- test_analysis.py
```

### Why two HTTP servers?

Streamlit's built-in static file handler forces `Content-Type: text/plain` on MP4 files as a security measure (MP4 is not on its allowlist). Browsers refuse to play the overlay videos under that MIME. `serve_videos.py` is a ~120-line sidecar (stdlib `http.server` + CORS headers + HTTP Range) that serves MP4s with the correct MIME so the embedded `<video>` tags work everywhere, including iOS Safari.

PNGs (logos) are served by Streamlit directly -- PNG is on its allowlist, so the MIME issue doesn't apply.

### Module boundaries

- `analysis.py` has zero Streamlit dependencies. All functions are pure (DataFrame in, DataFrame / scalar / matplotlib figure out). Unit-testable without the server.
- `pipeline.py` is the heavy preprocessing (MediaPipe + OpenCV + Kalman). Side-effects are isolated to writes into `data/` and `static/`.
- `app.py` loads cached artifacts and renders them. If artifacts are missing it surfaces a clear error with the fix command.

## Tests

```bash
python3 -m pytest tests/
```

Covers the analysis functions (similarity score, per-joint deviation shape, advice generation). No Streamlit or OpenCV needed to run.

## Troubleshooting

**"Follow Through clip 'shaq' is missing preprocessed artifacts..."**
Run `python3 preprocess.py` from this directory and refresh.

**Video player shows a black frame on Chrome / Firefox**
The overlay MP4 wasn't encoded as H.264. Re-run `python3 preprocess.py`, which uses `avc1` explicitly. If `cv2.VideoWriter_fourcc(*"avc1")` fails to open a writer, your OpenCV build doesn't include FFmpeg H.264 -- install `opencv-python` (not `opencv-python-headless`).

**Trajectory plot is blank**
Matplotlib-based; should always render. If it doesn't, check the Streamlit server log for a Python traceback.

**Phone can't connect**
Make sure both the laptop and phone are on the same Wi-Fi. The launcher prints a **Network URL** -- hit that from the phone (the `localhost` URL won't work from a different device).

## What the demo is saying

- **Follow Through** -- Pose extraction + Savitzky-Golay smoothing + normalised keypoint distance yields a quantified similarity score. Shaq vs. Nash is a visually obvious contrast; the pipeline says the shooting wrist and guide-hand elbow deviate the most, which matches the visible form difference. The coaching advice is generated from the top-N most-deviated joints with directional heuristics.
- **EyeBall** -- Classical CV (HSV + contour + circularity) + Kalman tracking produce a per-frame (x, y). Two-pass smoothing (outlier rejection + Savitzky-Golay) cleans up false detections. Angle-pivot pass detection on the smoothed trace counts direction changes that look like passes.

Both pipelines run entirely on the laptop -- no cloud, no server dependency.
