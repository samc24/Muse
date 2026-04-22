# `yolo_detection.py` -- Setup and Usage

Thin wrapper around a PyTorch-YOLOv3 model, used by `track_ball.py` to detect the basketball in each frame.

## Installation

1. Clone and install [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) following its README.
2. Ensure the working directory is inside the `PyTorch-YOLOv3` checkout when running, so the model weights resolve correctly.

## Functions

### `save_tagged_frames(vid, output)`

Tags frames from a video in "real time" and writes them to an output folder.

- `vid` -- input video filename
- `output` -- output folder for tagged frames
- Returns: measured FPS of the video

"Real time" here means: each tag call is timed, and the loop waits that measured interval before pulling the next frame.

### `create_tagged_video(outputs, video_output, fps)`

Reconstructs a playable video from a folder of tagged frames.

- `outputs` -- folder of tagged frames produced by `save_tagged_frames`
- `video_output` -- filename of the output video
- `fps` -- target frames-per-second

## Example

```python
fps = save_tagged_frames('../pano1.avi', '../output')
# This takes a while.
create_tagged_video('../output', '../tagged_pano1.avi', fps)
```
