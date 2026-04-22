# FollowThrough Demo

Side-by-side shot-form analysis. Loads two pre-processed clips (Shaq vs. Nash), plays them in sync or independently, and surfaces per-joint deviation plus a shooting-wrist trajectory comparison.

Designed for a laptop-on-projector demo with optional phone viewing over LAN.

## Quick start

```bash
cd FollowThrough/demo
./run_demo.sh
```

That launches two servers:
- **Streamlit UI** on `http://localhost:8501`
- **Video server** on `http://localhost:8502` (serves the overlay MP4s with proper Content-Type)

Both bind to `0.0.0.0` so your phone on the same Wi-Fi can reach them. The Streamlit start-up log prints the LAN URL; use that address on your phone.

Ctrl+C in the terminal stops both.

## Why two servers?

Streamlit's built-in static file handler forces `Content-Type: text/plain` on anything outside a small whitelist (images, fonts, pdf, json, xml) as a security measure. MP4 is not on that list, which means the iframe refuses to play the video. The sidecar `serve_videos.py` (plain Python `http.server`) hands out MP4s with correct MIME types and CORS headers.

## Phone viewing (same Wi-Fi)

After `./run_demo.sh` starts, look for the Network URL it prints, e.g. `http://192.168.1.156:8501`. Hit that from your phone. The iframe auto-detects the hostname and fetches videos from `http://192.168.1.156:8502/...`.

If the videos don't load on phone: the phone's browser probably cached the old base64 version. Hard-refresh / close-and-reopen the tab.

## Re-generate clips (only needed if sources change)

```bash
/Users/Sameer/anaconda3/envs/eyeball/bin/python3 FollowThrough/demo/preprocess.py
```

Writes `static/{shaq,nash}_overlay.mp4` (H.264, browser-playable) and `data/{shaq,nash}_keypoints.csv` + `data/{shaq,nash}_meta.json`. Source clips pull from `FollowThrough/2kvids/`.

## Controls

| Control | What it does |
|---|---|
| Speed slider | Slomo / fast-forward both videos simultaneously (0.1x to 2.0x). |
| Trim (two sliders per video) | Set in/out points. Playback auto-clamps and loops back to the in-point. |
| **Sync & Play** | Resets both videos to their in-point and starts them together. |
| Pause | Pauses both. |
| Reset | Pauses and rewinds both to their in-point. |

## What it shows

- **Dual player** with skeleton overlays baked in (MediaPipe pose landmarks + Savitzky-Golay smoothing).
- **Per-joint mean deviation** (bar chart): Euclidean distance between the two shooters' normalised joint positions, averaged over 60 aligned frames. Normalisation centres on hip midpoint and scales by shoulder-hip distance, so the comparison is invariant to camera distance and player height.
- **Shooting-wrist vertical motion** (line chart): inverted pixel-y of the more-active wrist across the shot. Shows when the ball goes up and when it's released.

## Files

```
demo/
├── app.py               # Streamlit UI + dual player + metrics
├── preprocess.py        # Uses SkeletonMaker to produce overlays + CSVs
├── serve_videos.py      # Sidecar HTTP server for MP4s (correct MIME + CORS)
├── run_demo.sh          # Launches both servers
├── .streamlit/
│   └── config.toml      # Minor Streamlit config
├── static/              # Overlay videos (served by serve_videos.py)
│   ├── shaq_overlay.mp4
│   └── nash_overlay.mp4
├── data/                # Keypoint CSVs + metadata (read server-side by app.py)
│   ├── shaq_keypoints.csv, shaq_meta.json
│   └── nash_keypoints.csv, nash_meta.json
└── README.md
```

## Architecture in one paragraph (for interview)

Source clips run through `FollowThrough/source/SkeletonMaker.py` (MediaPipe Pose + Savitzky-Golay smoothing) at preprocess time, producing H.264 overlay videos and per-frame keypoint CSVs. The Streamlit app loads the CSVs server-side, renders a custom HTML/JS dual-video player inside a single iframe (so playback speed, trim, and sync all live in one place without Streamlit round-trips), and computes per-joint deviation in-process. Overlay videos are served by a sidecar Python HTTP server because Streamlit's static handler refuses MP4s. End-to-end runs on a laptop; no Heroku, no Django, no network dependency.
