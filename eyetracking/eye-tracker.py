#!/usr/bin/env python
"""
Copyright (c) 2024  Michael Ryan Hunsaker, M.Ed., Ph.D.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# coding=utf-8

import cv2
import numpy as np
import mediapipe as mp
import dlib
import time
import json
import logging
import datetime
import bz2
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from collections import defaultdict
import requests
from screeninfo import get_monitors
import argparse


def download_landmarks_model(
    model_path="./eyetracking/shape_predictor_68_face_landmarks.dat",
):
    """
    Download and extract the facial landmarks model file.
    Assumes code is run from ./Eye_Tracking and not ./Eye_Tracking/eyetracking.
    The landmarks model file will be saved to whatever directory the code is
    launched from

    Parameters
    ----------
    model_path : str
        Path where the model file should be saved.
        Default is "./eyetracking/shape_predictor_68_face_landmarks.dat"

    Returns
    -------
    bool
        True if download successful
        False if download failed

    Notes
    -----
    Downloads the dlib facial landmarks model from the official dlib repository
    and extracts it from the bz2 compressed format into the default or
    designated directory.
    """

    try:
        # Check if the model file already exists
        if Path(model_path).exists():
            # Report that the model file already exists
            print(f"Model file already exists at {model_path}")
            return True
        # if model file does not exist, download it from the official dlib repository
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = f"{model_path}.bz2"
        # Report download the compressed model file
        print("Downloading facial landmarks model...")
        print("This may take a few minutes depending on your internet connection.")
        """
        Download the compressed model file, set stream=True to allow for large files
        Set the block size to 1024 bytes to write the file in 1KB chunks
        """
        response = requests.get(url, stream=True)
        # Get the total size of the compressed model file
        total_size = int(response.headers.get("content-length", 0))
        # Set the block size for writing the file
        block_size = 1024
        # Write the compressed model file to disk
        with open(compressed_path, "wb") as f:
            # Iterate over the response content in 1KB chunks
            for data in response.iter_content(block_size):
                # Write the data to the file
                f.write(data)
                # Calculate the total size downloaded so far
                downloaded = f.tell()
                # Calculate the download progress and display a progress bar
                done = int(50 * downloaded / total_size)
                # Print the download progress bar
                print(
                    f"\rDownloading: [{'=' * done}{' ' * (50 - done)}] {downloaded}/{total_size} bytes",
                    end="",
                    flush=True,
                )

        # Extract the model file from the compressed archive
        print("\nExtracting model file...")
        # Open the compressed file and write the extracted model file
        with bz2.open(compressed_path) as fr, open(model_path, "wb") as fw:
            fw.write(fr.read())
        # Remove the compressed file after extraction
        os.remove(compressed_path)
        print("Download and extraction complete!")
        return True
    # Catch any exceptions that occur during download or extraction
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False


class EyeTrackingSystem:
    """
    A system for tracking eye movements and analyzing gaze patterns.

    This class implements real-time eye tracking using MediaPipe and dlib,
    with capabilities for generating heatmaps, tracking dwell times, and
    recording gaze trajectories across the full screen.

    Parameters
    ----------
    student_initials : str
        The initials of the student being tracked (2-4 letters, any case)
    base_dir : str, optional
        Base directory for storing tracking data
    dwell_threshold : float, optional
        Minimum time in seconds for a fixation to be considered a dwell
    dwell_radius_deg : float, optional
        Radius in degrees of visual angle for dwell detection
    model_path : str, optional
        Path to the shape predictor model file

    Attributes
    ----------
    heatmap : ndarray
        2D array storing gaze intensity data
    trajectory_data : defaultdict
        Dictionary storing gaze trajectory data binned by time
    dwell_events : list
        List of detected dwell events with timing and position information
    screen_width : int
        Width of the primary monitor in pixels
    screen_height : int
        Height of the primary monitor in pixels
    """

    def __init__(
        self,
        student_initials,
        base_dir="eye_tracking_data",
        dwell_threshold=1.0,
        dwell_radius_deg=2.0,
        model_path="./eyetracking/shape_predictor_68_face_landmarks.dat",
    ):
        # Get primary monitor dimensions
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height

        # Set up logging
        now = datetime.datetime.now()
        # Create log file name
        filename = f"eye_tracking_{now:%Y%m%d_%H%M%S}.log"
        # Create logs directory
        log_dir = Path("logs")
        # Create logs directory if it does not exist
        log_dir.mkdir(exist_ok=True)
        # Configure logging to write to file and console
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / filename),
                logging.StreamHandler(),
            ],
        )
        # Create logger
        self.logger = logging.getLogger(__name__)
        # Check if facial landmarks model exists, download if not
        if not Path(model_path).exists():
            self.logger.info(
                "Facial landmarks model not found. Attempting to download..."
            )
            if not download_landmarks_model(model_path):
                raise RuntimeError("Failed to download facial landmarks model")
        # Initialize dlib face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        # Load the facial landmarks model
        self.predictor = dlib.shape_predictor(model_path)
        # Initialize MediaPipe face mesh model
        self.mp_face_mesh = mp.solutions.face_mesh
        # Set up MediaPipe face mesh model with parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Initialize the student directory for storing tracking data
        self.student_dir = Path(base_dir) / student_initials.upper()
        """
        Create subdirectories for heatmaps, trajectories, and dwell times
        if they do not already exist, it also allows for nested directories
        without raising an error if the parent directory already exists
        """
        self.student_dir.mkdir(parents=True, exist_ok=True)
        (self.student_dir / "heatmaps").mkdir(exist_ok=True)
        (self.student_dir / "trajectories").mkdir(exist_ok=True)
        (self.student_dir / "dwell_times").mkdir(exist_ok=True)

        # Initialize tracking data using screen dimensions
        self.heatmap = np.zeros(
            (self.screen_height, self.screen_width), dtype=np.float32
        )
        # Initialize gaze tracking variables
        self.last_head_rotation = None
        # Initialize gaze trajectory data
        self.trajectory_data = defaultdict(list)
        # Initialize trial start time
        self.trial_start_time = time.time()
        # Set dwell time threshold and radius
        self.dwell_threshold = dwell_threshold
        # Convert degrees of visual angle to pixels
        self.dwell_radius_pixels = self._degrees_to_pixels(dwell_radius_deg)
        # Initialize dwell time tracking variables
        self.current_dwell = {"start_time": None, "location": None, "points": []}
        # Initialize dwell events list
        self.dwell_events = []
        # Initialize camera matrix and distortion coefficients
        self.camera_matrix = None
        # Initialize the 3D model points for the eye corners and center
        self.dist_coeffs = np.zeros((4, 1))

        # Define the 3D model points for the eye corners and center
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ])
        # Define the 2D image points corresponding to the 3D model points
        self.logger.info(
            f"Initialized eye tracking system for student: {student_initials}"
        )

    def _degrees_to_pixels(self, degrees, screen_distance_cm=60, screen_width_cm=34.5):
        """
        Convert visual angles in degrees to pixels.

        Parameters
        ----------
        degrees : float
            Visual angle in degrees
        screen_distance_cm : float, optional
            Distance from eyes to screen in centimeters
        screen_width_cm : float, optional
            Width of screen in centimeters

        Returns
        -------
        int
            Number of pixels corresponding to the given visual angle
        """
        # Calculate pixels per centimeter
        pixels_per_cm = self.screen_width / screen_width_cm
        # Calculate pixels per degree
        cm_per_degree = screen_distance_cm * np.tan(np.radians(degrees))
        # Convert to pixels
        return int(cm_per_degree * pixels_per_cm)

    def _pixels_to_degrees(self, pixels, screen_distance_cm=60, screen_width_cm=34.5):
        """
        Convert pixels to visual angles in degrees.

        Parameters
        ----------
        pixels : int
            Number of pixels to convert
        screen_distance_cm : float, optional
            Distance from eyes to screen in centimeters
        screen_width_cm : float, optional
            Width of screen in centimeters

        Returns
        -------
        float
            Visual angle in degrees
        """
        # Calculate visual angle in radians
        cm_per_pixel = screen_width_cm / self.screen_width
        distance_cm = pixels * cm_per_pixel
        return np.degrees(np.arctan(distance_cm / screen_distance_cm))

    def _get_eye_coordinates(self, landmarks, indices):
        """
        Extract eye landmark coordinates from MediaPipe face mesh results.

        Parameters
        ----------
        landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            Face mesh landmarks from MediaPipe.
        indices : list of int
            Indices of landmarks corresponding to eye points.

        Returns
        -------
        ndarray
            Array of (x, y) coordinates for the specified eye landmarks.
        """
        # Extract eye landmark coordinates
        return np.array([
            (landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in indices
        ])

    def _calculate_head_rotation(self, landmarks):
        """
        Calculate head rotation angle from facial landmarks.

        Parameters
        ----------
        landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            Face mesh landmarks from MediaPipe.

        Returns
        -------
        float or None
            Head rotation angle in degrees, or None if calculation fails.
        """
        # Get the 3D model points for the eye corners and center
        if self.camera_matrix is None:
            focal_length = 1280
            center = (640, 360)
            # Define the camera matrix
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype=np.float64,
            )
        # Get the 2D image points for the eye corners and center
        image_points = np.array(
            [
                (landmarks.landmark[30].x * 1280, landmarks.landmark[30].y * 720),
                (landmarks.landmark[152].x * 1280, landmarks.landmark[152].y * 720),
                (landmarks.landmark[226].x * 1280, landmarks.landmark[226].y * 720),
                (landmarks.landmark[446].x * 1280, landmarks.landmark[446].y * 720),
                (landmarks.landmark[57].x * 1280, landmarks.landmark[57].y * 720),
                (landmarks.landmark[287].x * 1280, landmarks.landmark[287].y * 720),
            ],
            dtype=np.float64,
        )
        # Solve for head rotation using PnP algorithm
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        # Calculate head rotation angle from rotation vector
        if not success:
            return None
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        # Calculate Euler angles from rotation matrix
        return np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))

    def _estimate_gaze_point(self, left_eye, right_eye, head_rotation):
        """
        Estimate gaze point from eye positions and head rotation.

        Parameters
        ----------
        left_eye : ndarray
            Array of left eye landmark coordinates.
        right_eye : ndarray
            Array of right eye landmark coordinates.
        head_rotation : float or None
            Head rotation angle in degrees.

        Returns
        -------
        ndarray
            Estimated (x, y) gaze point coordinates.
        """
        # Calculate the center of each eye
        eye_center = np.mean(
            [np.mean(left_eye, axis=0), np.mean(right_eye, axis=0)], axis=0
        )
        # Correct for head rotation
        if head_rotation is not None:
            correction_factor = np.tan(np.radians(head_rotation)) * 100
            eye_center[0] += correction_factor

        return eye_center

    def _update_heatmap(self, x, y):
        """
        Update gaze heatmap with new gaze point.

        Parameters
        ----------
        x : float
            Normalized x-coordinate of gaze point
        y : float
            Normalized y-coordinate of gaze point
        """
        # Convert normalized coordinates to pixel coordinates
        x_int, y_int = int(x * self.screen_width), int(y * self.screen_height)
        # Update heatmap if point is within screen bounds
        if 0 <= x_int < self.screen_width and 0 <= y_int < self.screen_height:
            # Add a Gaussian kernel centered at the gaze point
            self.heatmap[y_int, x_int] += 1
            # Smooth the heatmap
            self.heatmap = gaussian_filter(self.heatmap, sigma=1)

    def process_frame(self, frame, timestamp=None):
        """
        Process a single video frame for eye tracking.

        Parameters
        ----------
        frame : ndarray
            Video frame to process (BGR format).
        timestamp : float, optional
            Frame timestamp in seconds, by default None.

        Returns
        -------
        ndarray or None
            Estimated gaze point coordinates if detection successful, None otherwise.
        """
        # Use current time if timestamp is not provided
        if timestamp is None:
            timestamp = time.time()
        # Convert frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process frame with MediaPipe face mesh
        results = self.face_mesh.process(frame_rgb)
        # Process face mesh results if available
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Extract eye landmarks and calculate head rotation
            left_eye = self._get_eye_coordinates(
                landmarks, [362, 385, 387, 263, 373, 380]
            )
            right_eye = self._get_eye_coordinates(
                landmarks, [33, 160, 158, 133, 153, 144]
            )
            # Calculate head rotation angle
            head_rotation = self._calculate_head_rotation(landmarks)
            # Estimate gaze point from eye positions and head rotation
            if self.last_head_rotation is not None:
                rotation_diff = abs(head_rotation - self.last_head_rotation)
                if rotation_diff > 5:
                    return None
            # Update gaze point if head rotation is stable
            self.last_head_rotation = head_rotation
            gaze_point = self._estimate_gaze_point(left_eye, right_eye, head_rotation)
            # Update gaze heatmap and trajectory
            if gaze_point is not None:
                self._update_heatmap(gaze_point[0], gaze_point[1])
                self._update_trajectory(gaze_point[0], gaze_point[1], timestamp)
                self._update_dwell_time(gaze_point[0], gaze_point[1], timestamp)
            # Draw landmarks on frame
            for landmark in landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            return gaze_point
        return None

    def _update_trajectory(self, x, y, timestamp):
        """
        Update gaze trajectory data with new gaze point.

        Parameters
        ----------
        x : float
            Normalized x-coordinate of gaze point.
        y : float
            Normalized y-coordinate of gaze point.
        timestamp : float
            Time of gaze point in seconds.
        """
        # Calculate current time bin
        bin_number = int((timestamp - self.trial_start_time) / 0.5)  # 500ms bins
        # Create new bin if it doesn't exist
        self.trajectory_data[bin_number].append((x, y, timestamp))

    def _update_dwell_time(self, x, y, timestamp):
        """
        Update dwell time tracking with new gaze point.

        Parameters
        ----------
        x : float
            Normalized x-coordinate of gaze point.
        y : float
            Normalized y-coordinate of gaze point.
        timestamp : float
            Time of gaze point in seconds.
        """
        # Convert normalized coordinates to pixel coordinates
        current_point = np.array([x, y])
        # Check if a dwell event is currently in progress
        if self.current_dwell["start_time"] is None:
            self.current_dwell["start_time"] = timestamp
            self.current_dwell["location"] = current_point
            self.current_dwell["points"] = [(x, y, timestamp)]
        else:
            distance = np.linalg.norm(current_point - self.current_dwell["location"])
            # Update dwell event if point is within dwell radius
            if distance <= self.dwell_radius_pixels:
                self.current_dwell["points"].append((x, y, timestamp))
                dwell_duration = timestamp - self.current_dwell["start_time"]
                # Check if dwell duration exceeds threshold
                if dwell_duration >= self.dwell_threshold:
                    if (
                        len(self.dwell_events) == 0
                        or self.dwell_events[-1]["end_time"] != timestamp
                    ):
                        mean_position = np.mean(
                            np.array(self.current_dwell["points"])[:, :2], axis=0
                        )
                        self.dwell_events.append({
                            "start_time": self.current_dwell["start_time"],
                            "end_time": timestamp,
                            "duration": dwell_duration,
                            "mean_x": float(mean_position[0]),
                            "mean_y": float(mean_position[1]),
                            "n_points": len(self.current_dwell["points"]),
                        })
            else:
                self.current_dwell["start_time"] = timestamp
                self.current_dwell["location"] = current_point
                self.current_dwell["points"] = [(x, y, timestamp)]

    def save_trial_data(self, trial_id=None):
        """
        Save all collected trial data to files.

        Parameters
        ----------
        trial_id : str, optional
            Unique identifier for the trial, by default None.
            If None, generates ID from current timestamp.

        Returns
        -------
        dict
            Dictionary containing paths to saved files:
            - 'heatmap': Path to heatmap image
            - 'trajectory': Path to trajectory data JSON
            - 'dwell': Path to dwell time data JSON
        """
        # Generate trial ID if not provided
        if trial_id is None:
            trial_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create directories for saving data
        self.logger.info(f"Saving trial data for trial_id: {trial_id}")
        # Save heatmap image
        normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        # Apply color map to heatmap
        heatmap_color = cv2.applyColorMap(
            normalized_heatmap.astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_path = self.student_dir / "heatmaps" / f"heatmap_{trial_id}.png"
        cv2.imwrite(str(heatmap_path), heatmap_color)
        # Save gaze trajectory data
        trajectory_path = (
            self.student_dir / "trajectories" / f"trajectory_{trial_id}.json"
        )
        trajectory_data = {
            "trial_id": trial_id,
            "trial_start": self.trial_start_time,
            "trial_end": time.time(),
            "bins": {str(k): v for k, v in self.trajectory_data.items()},
        }
        # Save dwell time data
        with open(trajectory_path, "w") as f:
            json.dump(trajectory_data, f, indent=2)
        dwell_path = self.student_dir / "dwell_times" / f"dwell_{trial_id}.json"
        dwell_data = {
            "trial_id": trial_id,
            "dwell_threshold": self.dwell_threshold,
            "dwell_radius_degrees": self._pixels_to_degrees(self.dwell_radius_pixels),
            "events": self.dwell_events,
        }
        with open(dwell_path, "w") as f:
            json.dump(dwell_data, f, indent=2)
        # Log success
        self.logger.info(f"Successfully saved all trial data for trial_id: {trial_id}")
        return {
            "heatmap": str(heatmap_path),
            "trajectory": str(trajectory_path),
            "dwell": str(dwell_path),
        }


def main():
    """
    Main function to run the eye tracking system in background while student watches a video.

    Command line arguments:
    ----------------------
    student_initials : str
        Student initials (e.g., AaAa for Aaron Aaronson)
    -t, --trial-duration : int
        Trial duration in minutes (default: 5)
    --model-path : str, optional
        Path to facial landmarks model file
    --hide-tracking : bool, optional
        Hide the tracking window (default: False)

    Usage:
    ------
    To run the tracker for 5 minutes with visual feedback for debugging:
    $ python eye-tracker.py RYHU -t 5

    To run the tracker in hidden mdoe for 5 minutes for data collection:
    $ python eye-tracker.py RYHU -t 5 --hide-tracking
    $ python eye-tracker.py RyHu --trial-duration 5 --model-path ./eyetracking/shape_predictor_68_face_landmarks.dat --hide-tracking
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Eye tracking system with student identification"
    )
    # Required argument for student initials
    parser.add_argument(
        "student_initials", type=str, help="Student initials (e.g., JD for John Doe)"
    )
    # Optional argument for trial duration
    parser.add_argument(
        "-t",
        "--trial-duration",
        type=float,
        default=5,
        help="Trial duration in minutes (default: 5)",
    )
    # Optional argument for facial landmarks model path
    parser.add_argument(
        "--model-path",
        type=str,
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to facial landmarks model file",
    )
    # Optional argument to hide tracking window
    parser.add_argument(
        "--hide-tracking",
        action="store_true",
        help="Hide the tracking window (run in background)",
    )
    args = parser.parse_args()
    # Validate student initials
    if not 2 <= len(args.student_initials) <= 4 or not args.student_initials.isalpha():
        raise ValueError("Student initials must be 2-4 letters")

    # Convert minutes to seconds
    trial_duration_seconds = args.trial_duration * 60

    # Get primary monitor dimensions
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set capture resolution to match screen resolution, width and height separately
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
    # Create window to display webcam feed
    if not args.hide_tracking:
        # Create debug window if tracking visualization is enabled
        cv2.namedWindow("Eye Tracking Debug", cv2.WINDOW_NORMAL)
        # Move and resize window
        cv2.moveWindow("Eye Tracking Debug", 0, 0)
        # Resize window to 1/4 of screen resolution
        cv2.resizeWindow("Eye Tracking Debug", screen_width // 4, screen_height // 4)
    # Initialize eye tracking system
    try:
        # Initialize eye tracker
        tracker = EyeTrackingSystem(args.student_initials, model_path=args.model_path)
    except Exception as e:
        print(f"Error initializing eye tracker: {str(e)}")
        # Release webcam and close windows
        cap.release()
        return
    # Start trial timer
    trial_start = time.time()
    print(
        f"\nEye tracking started for {args.trial_duration} minutes. Press 'q' to quit."
    )
    # Main loop to process webcam frames
    try:
        # Initialize trial id
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to match screen resolution
            frame = cv2.resize(frame, (screen_width, screen_height))
            # Process the frame
            current_time = time.time()
            # Get gaze point from eye tracker
            gaze_point = tracker.process_frame(frame, current_time)
            # Display gaze point on screen
            if not args.hide_tracking and gaze_point is not None:
                # Draw gaze point only in debug window
                debug_frame = frame.copy()
                # Draw gaze point on debug frame
                cv2.circle(
                    debug_frame,
                    (
                        int(gaze_point[0] * tracker.screen_width),
                        int(gaze_point[1] * tracker.screen_height),
                    ),
                    5,
                    (0, 255, 0),
                    -1,
                )
                # Show small debug window
                small_frame = cv2.resize(
                    debug_frame, (screen_width // 4, screen_height // 4)
                )
                # Display the frame
                cv2.imshow("Eye Tracking Debug", small_frame)

            # Check for both time limit and quit command
            if current_time - trial_start >= trial_duration_seconds or cv2.waitKey(
                1
            ) & 0xFF == ord("q"):
                saved_files = tracker.save_trial_data()
                print(f"\nTrial data saved for student {args.student_initials}:")
                for key, path in saved_files.items():
                    print(f"{key.capitalize()}: {path}")
                break
    # Graceful shutdown
    except Exception as e:
        tracker.logger.error(f"Error during tracking: {str(e)}")
    # Release webcam and close windows
    finally:
        # Release webcam and close windows
        cap.release()
        # Close debug window if open
        cv2.destroyAllWindows()


# Run the main function
if __name__ == "__main__":
    main()
