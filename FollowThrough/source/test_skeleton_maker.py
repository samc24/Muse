# test_skeleton_video.py

from SkeletonMaker import SkeletonMaker

# Define input and output paths
input_video = "/Users/Sameer/Documents/Coding/Muse/FollowThrough/2kvids/bird3.mp4"
output_video = "/Users/Sameer/Documents/Coding/Muse/FollowThrough/unsmooth/bird3_mediapipe_output.avi"

# Create instance of SkeletonMaker
skel = SkeletonMaker(
    model_complexity=2,     # Accurate but slower
    window_length=7,        # For smoothing
    polyorder=3,            # For smoothing
    smoothing=True          # Toggle smoothing on/off
)

# Run the pose estimation on video
skel.process_video(input_video, output_video)
