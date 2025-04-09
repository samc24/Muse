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

# test_image_overlay.py

# Set your input/output paths
input_image = "/Users/Sameer/Documents/Coding/Muse/FollowThrough/basketball_photos/release/release001.jpg"
output_image = "/Users/Sameer/Documents/Coding/Muse/FollowThrough/output_jpgs/release001_output.jpg"

# Create instance and run
skel = SkeletonMaker()
skel.generate_pose_overlay_image(input_image, output_image)

# test_template_generation.py

# Provide glob path and output name
input_glob = "/Users/Sameer/Documents/Coding/Muse/FollowThrough/basketball_photos/release/*.jpg"
output_name = "release"

# Create instance and generate averaged template
skel = SkeletonMaker()
skel.generate_template_from_images(input_glob, output_name)


