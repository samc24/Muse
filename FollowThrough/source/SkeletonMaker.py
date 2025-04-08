import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from collections import deque
import time
import os
import pickle
from glob import iglob

class SkeletonMaker:
    """
    A class for pose estimation using MediaPipe, including smoothing, drawing, and template generation.
    """

    def __init__(self, model_complexity=2, window_length=7, polyorder=3, smoothing=True):
        """
        Initialize the pose estimation model and settings.

        Parameters:
        - model_complexity (int): Controls accuracy vs. speed of the MediaPipe pose model.
        - window_length (int): Number of frames used for smoothing.
        - polyorder (int): Polynomial order for Savitzky-Golay smoothing.
        - smoothing (bool): Whether to apply smoothing to pose keypoints.
        """
        self.model_complexity = model_complexity
        self.window_length = window_length
        self.polyorder = polyorder
        self.smoothing = smoothing

        # Initialize MediaPipe pose estimator
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # False means it's optimized for video
            model_complexity=self.model_complexity,  # 0, 1, or 2 depending on speed vs accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

        # Define keypoints used for analysis; only a subset of all available landmarks
        self.keypoints = {
            "L_SHOULDER": 11, "R_SHOULDER": 12,
            "L_ELBOW": 13, "R_ELBOW": 14,
            "L_WRIST": 15, "R_WRIST": 16,
            "L_HIP": 23, "R_HIP": 24,
            "L_KNEE": 25, "R_KNEE": 26,
            "L_ANKLE": 27, "R_ANKLE": 28,
            "L_HEEL": 29, "R_HEEL": 30,
            "L_FOOT_INDEX": 31, "R_FOOT_INDEX": 32
        }

        # HEAD is virtual here, referring to the nose point (index 0)
        self.landmark_names = ["HEAD"] + list(self.keypoints.keys())
        self.landmark_ids = [0] + list(self.keypoints.values())
        self.num_points = len(self.landmark_ids)

        # Pose connections for drawing lines between points
        self.pose_pairs = [
            (0, 1), (0, 2),  # head to shoulders
            (1, 3), (3, 5),  # left arm
            (2, 4), (4, 6),  # right arm
            (1, 7), (2, 8),  # shoulders to hips
            (7, 9), (9, 11), # left leg
            (8, 10), (10, 12), # right leg
            (11, 13), (12, 14)  # feet
        ]

    def _init_buffers(self):
        """Initialize a list of buffers for each keypoint to store past positions for smoothing."""
        return [deque(maxlen=self.window_length) for _ in range(self.num_points)]

    def _compute_neck_point(self, landmarks, w, h):
        """
        Compute the midpoint between left and right shoulder landmarks,
        used as an artificial neck keypoint.
        """
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        if l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5:
            lx, ly = l_shoulder.x * w, l_shoulder.y * h
            rx, ry = r_shoulder.x * w, r_shoulder.y * h
            return (int((lx + rx) / 2), int((ly + ry) / 2))
        return None

    def _get_raw_points(self, landmarks, w, h):
        """
        Convert MediaPipe landmark objects to (x, y) pixel positions
        if the visibility score is high enough.
        """
        points = [None] * self.num_points
        if landmarks[0].visibility > 0.5:
            points[0] = (int(landmarks[0].x * w), int(landmarks[0].y * h))
        for i, mp_idx in enumerate(self.landmark_ids[1:], start=1):
            lm = landmarks[mp_idx]
            if lm.visibility > 0.5:
                points[i] = (int(lm.x * w), int(lm.y * h))
        return points

    def _smooth_points(self, raw_points, buffers):
        """
        Apply Savitzky-Golay filter to each keypoint if visible,
        using history stored in buffers. If the point is missing,
        reuse the last known value.
        """
        smoothed = [None] * self.num_points
        for i, pt in enumerate(raw_points):
            if pt:
                buffers[i].append(pt)
            elif len(buffers[i]) > 0:
                buffers[i].append(buffers[i][-1])

            # Apply smoothing if we have enough data points
            if len(buffers[i]) >= self.window_length:
                xs, ys = zip(*buffers[i])
                sx = int(savgol_filter(xs, self.window_length, self.polyorder)[-1])
                sy = int(savgol_filter(ys, self.window_length, self.polyorder)[-1])
                smoothed[i] = (sx, sy)
            elif len(buffers[i]) > 0:
                smoothed[i] = buffers[i][-1]  # fallback to last known value

        return smoothed

    def _extract_pose_data(self, frame):
        """
        Runs pose estimation on a given frame and extracts:
        - raw landmark positions in image space
        - computed neck point
        """
        h, w = frame.shape[:2]
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return [None] * self.num_points, None

        landmarks = results.pose_landmarks.landmark
        raw_points = self._get_raw_points(landmarks, w, h)
        neck_point = self._compute_neck_point(landmarks, w, h)
        return raw_points, neck_point

    def _draw_pose(self, frame, points, neck_point=None, radius=5, line_thickness=2):
        """
        Draw joints and connecting lines on the frame,
        including a computed neck-to-head/shoulders line.
        """
        # Draw joints
        for pt in points:
            if pt:
                cv2.circle(frame, pt, radius, (0, 0, 255), -1)

        # Draw bone connections
        for a, b in self.pose_pairs:
            if a < len(points) and b < len(points):
                if points[a] and points[b]:
                    cv2.line(frame, points[a], points[b], (0, 255, 255), line_thickness, lineType=cv2.LINE_AA)

        # Draw the neck and its lines to head and shoulders
        if neck_point:
            cv2.circle(frame, neck_point, 6, (255, 0, 0), -1)
            if points[0]:  # HEAD (nose)
                cv2.line(frame, neck_point, points[0], (0, 255, 255), line_thickness, lineType=cv2.LINE_AA)
            if points[1]:  # L_SHOULDER
                cv2.line(frame, neck_point, points[1], (0, 255, 255), line_thickness, lineType=cv2.LINE_AA)
            if points[2]:  # R_SHOULDER
                cv2.line(frame, neck_point, points[2], (0, 255, 255), line_thickness, lineType=cv2.LINE_AA)

    def process_video(self, input_path, output_path):
        """
        Process an entire video, overlaying the pose skeleton and optionally smoothing keypoints.

        Args:
            input_path (str): Path to the input video file.
            output_path (str): Path to save the output video with the overlaid skeleton.
        """
        cap = cv2.VideoCapture(input_path)
        hasFrame, frame = cap.read()
        if not hasFrame:
            print(f"Failed to read video: {input_path}")
            return

        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (w, h))
        buffers = self._init_buffers()

        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                break

            raw_points, neck_point = self._extract_pose_data(frame)
            points = self._smooth_points(raw_points, buffers) if self.smoothing else raw_points
            self._draw_pose(frame, points, neck_point)

            writer.write(frame)
            cv2.imshow("Skeleton", frame)
            if cv2.waitKey(1) == 27:  # ESC key to break
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to {output_path}")

    def generate_pose_overlay_image(self, input_image_path, output_image_path):
        """
        Process a single image and overlay the pose skeleton.

        Args:
            input_image_path (str): Path to input image.
            output_image_path (str): Path to save the image with overlaid skeleton.
        """
        frame = cv2.imread(input_image_path)
        if frame is None:
            print(f"Could not read image: {input_image_path}")
            return

        raw_points, neck_point = self._extract_pose_data(frame)
        self._draw_pose(frame, raw_points, neck_point)

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, frame)
        print(f"Pose overlay saved to {output_image_path}")

    def generate_template_from_images(self, input_glob_pattern, output_name):
        """
        Generate a pose template by averaging pose skeletons across a batch of images.

        Args:
            input_glob_pattern (str): Glob pattern to read multiple pose images.
            output_name (str): Basename for output template files (image + joints file).
        """
        skeleton_dict = {}
        photo_ind = 0

        for f in iglob(input_glob_pattern):
            img = cv2.imread(f)
            if img is None:
                continue
            raw_points, neck_point = self._extract_pose_data(img)
            points = raw_points
            if neck_point:
                points.append(neck_point)  # add synthetic neck point if available

            # Store pairwise joint connections for this image
            skeleton_dict[photo_ind] = []
            for pair in self.pose_pairs:
                partA, partB = pair
                if partA < len(points) and partB < len(points):
                    if points[partA] and points[partB]:
                        skeleton_dict[photo_ind].append((points[partA], points[partB]))
            photo_ind += 1

        if not skeleton_dict:
            print("No valid skeletons detected.")
            return

        # Average all skeletons into one set of joint-pairs
        averages = {i: np.zeros((2, 2)) for i in range(len(skeleton_dict[0]))}
        for s in skeleton_dict:
            for j, (ptA, ptB) in enumerate(skeleton_dict[s]):
                averages[j][0] += ptA
                averages[j][1] += ptB

        for j in averages:
            averages[j][0] = (averages[j][0] / len(skeleton_dict)).astype(int)
            averages[j][1] = (averages[j][1] / len(skeleton_dict)).astype(int)

        # Prepare blank canvas and draw averaged skeleton
        blank_frame = np.zeros((1800, 1400, 3), np.uint8)
        center_point = tuple(averages[1][0])
        x_diff = (blank_frame.shape[1] // 2) - center_point[0]
        y_diff = (blank_frame.shape[0] // 4) - center_point[1]

        average_joints = {}
        for key, value in averages.items():
            pointA = value[0] + np.array([x_diff, y_diff])
            pointB = value[1] + np.array([x_diff, y_diff])
            pointA = tuple(pointA)
            pointB = tuple(pointB)

            cv2.line(blank_frame, pointA, pointB, (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(blank_frame, pointA, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(blank_frame, pointB, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            pose_pair = self.pose_pairs[key]
            average_joints[pose_pair[0]] = pointA
            average_joints[pose_pair[1]] = pointB

        # Save outputs
        os.makedirs('average_joints', exist_ok=True)
        os.makedirs('output_templates', exist_ok=True)
        with open(f'average_joints/average_joints_{output_name}.pickle', 'wb') as handle:
            pickle.dump(average_joints, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cv2.imwrite(f'output_templates/average_{output_name}.jpg', blank_frame)
        print(f"Template '{output_name}' saved.")
