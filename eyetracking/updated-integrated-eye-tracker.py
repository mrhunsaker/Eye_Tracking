#!/usr/bin/env python
# coding: utf-8
"""
copyright (c) 2024  michael ryan hunsaker, m.ed., ph.d.
licensed under the apache license, version 2.0 (the "license");
you may not use this file except in compliance with the license.
you may obtain a copy of the license at

    https://www.apache.org/licenses/license-2.0

unless required by applicable law or agreed to in writing, software
distributed under the license is distributed on an "as is" basis,
without warranties or conditions of any kind, either express or implied.
see the license for the specific language governing permissions and
limitations under the license.
"""

# import the necessary packages
import argparse
import bz2
import datetime
import json
import logging
import os
from pathlib import Path
import time
import tkinter as tk

import cv2
import dlib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import requests
from screeninfo import get_monitors
from sklearn.svm import SVR


def download_landmarks_model(
    model_choice,
    model_path_5="./eyetracking/shape_predictor_5_face_landmarks.dat",
    model_path_68="./eyetracking/shape_predictor_68_face_landmarks.dat",
):
    """
    Downloads and extracts the facial landmarks model file from dlib.net based on the model_choice argument.

    Args:
        model_choice (str): "5" or "68" indicating the desired number of facial landmarks.
        model_path_5 (str, optional): Path to the 5-point model file. Defaults to "./eyetracking/shape_predictor_5_face_landmarks.dat".
        model_path_68 (str, optional): Path to the 68-point model file. Defaults to "./eyetracking/shape_predictor_68_face_landmarks.dat".

    Returns:
        str: Path to the downloaded model file if successful, None otherwise.
    """
    try:
        # Ensure model_choice is a string and strip any path information
        model_choice = str(model_choice).strip()
        if not model_choice in ["5", "68"]:
            raise ValueError(
                f"Invalid model choice: {model_choice}. Must be '5' or '68' (No quotation marks)"
            )

        # Determine which model to download based on choice
        if model_choice == "5":
            download_path = model_path_5
            url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
        else:  # model_choice == "68"
            download_path = model_path_68
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(download_path), exist_ok=True)

        # Check if the model file already exists
        if Path(download_path).exists():
            print(f"Model file already exists at {download_path}")
            return download_path

        # Download the model file
        compressed_path = f"{download_path}.bz2"

        print("Downloading facial landmarks model...")
        print("This may take a few minutes depending on your internet connection.")

        # Download the file using requests
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with open(compressed_path, "wb") as f:
            # Iterate over the response content and write it to the file
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded = f.tell()
                done = int(50 * downloaded / total_size)
                print(
                    f"\rDownloading: [{'=' * done}{' ' * (50 - done)}] {downloaded}/{total_size} bytes",
                    end="",
                    flush=True,
                )

        print("\nExtracting model file...")
        # Extract the model file using bz2.open()
        with bz2.open(compressed_path) as fr, open(download_path, "wb") as fw:
            fw.write(fr.read())

        # Remove the compressed file once the model file is extracted
        os.remove(compressed_path)
        print("Download and extraction complete!")
        return download_path

    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None


class IntegratedEyeTrackingSystem:
    """
    integrated eye tracking system combining basic and advanced features.
    """

    def __init__(
        self,
        student_initials,
        base_dir="eye_tracking_data",
        dwell_threshold=0.5,
        dwell_radius_deg=1.5,
        model_choice="68",
        calibration_points=9,
    ):
        if model_choice == "5":
            self.model_path = "./eyetracking/shape_predictor_5_face_landmarks.dat"
        elif model_choice == "68":
            self.model_path = "./eyetracking/shape_predictor_68_face_landmarks.dat"
        else:
            raise ValueError(f"Invalid model choice: {args.model}")

        # get primary monitor dimensions
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height

        # set up logging
        now = datetime.datetime.now()
        filename = f"eye_tracking_{now:%y%m%d_%h%m%s}.log"
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / filename),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # initialize face detection and tracking
        if not Path(self.model_path).exists():
            self.logger.info(
                "facial landmarks model not found. attempting to download..."
            )
            if not download_landmarks_model(self.model_path):
                raise RuntimeError("failed to download facial landmarks model")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.model_path)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # initialize data directories
        self.student_dir = Path(base_dir) / student_initials.upper()
        self.student_dir.mkdir(parents=True, exist_ok=True)
        (self.student_dir / "heatmaps").mkdir(exist_ok=True)
        (self.student_dir / "trajectories").mkdir(exist_ok=True)
        (self.student_dir / "dwell_times").mkdir(exist_ok=True)

        # initialize tracking data
        self.heatmap = np.zeros(
            (self.screen_height, self.screen_width), dtype=np.float32
        )
        self.last_head_rotation = None
        self.trajectory_data = {}  # change from defaultdict to regular dict
        self.trial_start_time = time.time()
        self.dwell_threshold = dwell_threshold
        self.dwell_radius_pixels = self._degrees_to_pixels(dwell_radius_deg)
        self.current_dwell = {"start_time": None, "location": None, "points": []}
        self.dwell_events = []

        # initialize camera parameters
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ])

        # advanced features initialization
        self.session_duration = None
        self.session_start_time = None
        self.session_active = False
        self.calibration_points = calibration_points
        self.calibration_data = []
        self.gaze_model_x = SVR(kernel="rbf", C=5000)
        self.gaze_model_y = SVR(kernel="rbf", C=5000)
        self.is_calibrated = False
        self.ear_threshold = 0.2
        self.min_ear_frames = 2
        self.smoothing_window = 5
        self.gaze_history = []

        # initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("could not open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)

    def _degrees_to_pixels(self, degrees, screen_distance_cm=60, screen_width_cm=34.5):
        pixels_per_cm = self.screen_width / screen_width_cm
        cm_per_degree = screen_distance_cm * np.tan(np.radians(degrees))
        return int(cm_per_degree * pixels_per_cm)

    def _pixels_to_degrees(self, pixels, screen_distance_cm=60, screen_width_cm=34.5):
        """
        convert pixels to visual angles in degrees.

        parameters
        ----------
        pixels : int
            number of pixels to convert
        screen_distance_cm : float, optional
            distance from eyes to screen in centimeters
        screen_width_cm : float, optional
            width of screen in centimeters

        returns
        -------
        float
            visual angle in degrees
        """
        # calculate visual angle in radians
        cm_per_pixel = screen_width_cm / self.screen_width
        distance_cm = pixels * cm_per_pixel
        return np.degrees(np.arctan(distance_cm / screen_distance_cm))

    def _get_eye_coordinates(self, landmarks, indices):
        """
        extract eye landmark coordinates from mediapipe face mesh results.

        parameters
        ----------
        landmarks : mediapipe.framework.formats.landmark_pb2.normalizedlandmarklist
            face mesh landmarks from mediapipe.
        indices : list of int
            indices of landmarks corresponding to eye points.

        returns
        -------
        ndarray
            array of (x, y) coordinates for the specified eye landmarks.
        """
        # extract eye landmark coordinates
        return np.array([
            (landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in indices
        ])

    def _calculate_head_rotation(self, landmarks):
        """
        calculate head rotation angle from facial landmarks.

        parameters
        ----------
        landmarks : mediapipe.framework.formats.landmark_pb2.normalizedlandmarklist
            face mesh landmarks from mediapipe.

        returns
        -------
        float or None
            head rotation angle in degrees, or None if calculation fails.
        """
        # get the 3d model points for the eye corners and center
        if self.camera_matrix is None:
            focal_length = 1280
            center = (640, 360)
            # define the camera matrix
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype=np.float64,
            )
        # get the 2d image points for the eye corners and center
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
        # solve for head rotation using pnp algorithm
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs
        )
        # calculate head rotation angle from rotation vector
        if not success:
            return None
        # convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        # calculate euler angles from rotation matrix
        return np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))

    def _estimate_gaze_point(self, left_eye, right_eye, head_rotation):
        """
        estimate gaze point from eye positions and head rotation.

        parameters
        ----------
        left_eye : ndarray
            array of left eye landmark coordinates.
        right_eye : ndarray
            array of right eye landmark coordinates.
        head_rotation : float or None
            head rotation angle in degrees.

        returns
        -------
        ndarray
            estimated (x, y) gaze point coordinates.
        """
        # calculate the center of each eye
        eye_center = np.mean(
            [np.mean(left_eye, axis=0), np.mean(right_eye, axis=0)], axis=0
        )
        # correct for head rotation
        if head_rotation is not None:
            correction_factor = np.tan(np.radians(head_rotation)) * 100
            eye_center[0] += correction_factor

        return eye_center

    def _update_heatmap(self, x, y):
        """
        Update gaze heatmap with new gaze point with increased sensitivity.

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
            # Create a more pronounced Gaussian kernel for each point
            kernel_size = 50  # Increased kernel size for better visibility
            sigma = 10.0  # Increased sigma for smoother spread

            # Calculate kernel bounds
            x_min = max(0, x_int - kernel_size // 2)
            x_max = min(self.screen_width, x_int + kernel_size // 2)
            y_min = max(0, y_int - kernel_size // 2)
            y_max = min(self.screen_height, y_int + kernel_size // 2)

            # Create kernel coordinates
            X, Y = np.meshgrid(
                np.arange(x_min - x_int, x_max - x_int),
                np.arange(y_min - y_int, y_max - y_int),
            )

            # Calculate Gaussian kernel
            kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
            kernel = kernel / kernel.max()  # Normalize kernel

            # Add kernel to heatmap
            self.heatmap[y_min:y_max, x_min:x_max] += (
                kernel * 0.3
            )  # Reduced multiplication factor for finer control

    def process_frame(self, frame, timestamp=None):
        """
                process a single video frame for eye tracking.
        self.trajectory_data = {}  # change from defaultdict to regular dict
                parameters
                ----------
                frame : ndarray
                    video frame to process (bgr format).
                timestamp : float, optional
                    frame timestamp in seconds, by default None.

                returns
                -------
                ndarray or None
                    estimated gaze point coordinates if detection successful, None otherwise.
        """
        # use current time if timestamp is not provided
        if timestamp is None:
            timestamp = time.time()
        # convert frame to rgb format
        frame_flipped = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        # process frame with mediapipe face mesh
        results = self.face_mesh.process(frame_rgb)
        # process face mesh results if available
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # extract eye landmarks and calculate head rotation
            left_eye = self._get_eye_coordinates(
                landmarks, [362, 385, 387, 263, 373, 380]
            )
            right_eye = self._get_eye_coordinates(
                landmarks, [33, 160, 158, 133, 153, 144]
            )
            # calculate head rotation angle
            head_rotation = self._calculate_head_rotation(landmarks)
            # estimate gaze point from eye positions and head rotation
            if self.last_head_rotation is not None:
                rotation_diff = abs(head_rotation - self.last_head_rotation)
                if rotation_diff > 5:
                    return None
            # update gaze point if head rotation is stable
            self.last_head_rotation = head_rotation
            gaze_point = self._estimate_gaze_point(left_eye, right_eye, head_rotation)
            # update gaze heatmap and trajectory
            if gaze_point is not None:
                self._update_heatmap(gaze_point[0], gaze_point[1])
                self._update_trajectory(gaze_point[0], gaze_point[1], timestamp)
                self._update_dwell_time(gaze_point[0], gaze_point[1], timestamp)
            # draw landmarks on frame
            for landmark in landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            return gaze_point
        return None

    def _update_trajectory(self, x, y, timestamp):
        """
        update gaze trajectory data with new gaze point at 0.5 second intervals.

        parameters
        ----------
        x : float
            normalized x-coordinate of gaze point
        y : float
            normalized y-coordinate of gaze point
        timestamp : float
            time of gaze point in seconds
        """
        # calculate which 0.5s bin this timestamp belongs to
        elapsed_time = timestamp - self.trial_start_time
        current_bin = int(elapsed_time * 8)  # multiply by 2 to get 0.5s bins
        bin_timestamp = self.trial_start_time + (
            current_bin * 0.125
        )  # get exact bin timestamp

        # initialize the bin if it doesn't exist
        if current_bin not in self.trajectory_data:
            self.trajectory_data[current_bin] = {
                "bin_timestamp": bin_timestamp,
                "gaze_points": [],
                "mean_position": None,
            }

        # add the gaze point to the current bin
        self.trajectory_data[current_bin]["gaze_points"].append({
            "x": x,
            "y": y,
            "timestamp": timestamp,
        })

        # calculate and update mean position for the bin
        points = self.trajectory_data[current_bin]["gaze_points"]
        mean_x = sum(p["x"] for p in points) / len(points)
        mean_y = sum(p["y"] for p in points) / len(points)
        self.trajectory_data[current_bin]["mean_position"] = {"x": mean_x, "y": mean_y}

    def _update_dwell_time(self, x, y, timestamp):
        """
        update dwell time tracking with new gaze point.

        parameters
        ----------
        x : float
            normalized x-coordinate of gaze point.
        y : float
            normalized y-coordinate of gaze point.
        timestamp : float
            time of gaze point in seconds.
        """
        # convert normalized coordinates to pixel coordinates
        current_point = np.array([x, y])
        # check if a dwell event is currently in progress
        if self.current_dwell["start_time"] is None:
            self.current_dwell["start_time"] = timestamp
            self.current_dwell["location"] = current_point
            self.current_dwell["points"] = [(x, y, timestamp)]
        else:
            distance = np.linalg.norm(current_point - self.current_dwell["location"])
            # update dwell event if point is within dwell radius
            if distance <= self.dwell_radius_pixels:
                self.current_dwell["points"].append((x, y, timestamp))
                dwell_duration = timestamp - self.current_dwell["start_time"]
                # check if dwell duration exceeds threshold
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

    def plot_trajectory(self, trial_id, trajectory_data):
        """
        Create a visualization of the gaze trajectory.

        Parameters
        ----------
        trial_id : str
            Unique identifier for the trial
        trajectory_data : dict
            Dictionary containing trajectory data

        Returns
        -------
        str
            Path to the saved trajectory plot
        """
        # Set up the plot with black background
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10.5, 8))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # Convert trajectory data points to plottable format
        x_coords = []
        y_coords = []
        timestamps = []

        for bin_data in trajectory_data["trajectory_bins"].values():
            for point in bin_data["gaze_points"]:
                x_coords.append(point["x"])
                y_coords.append(point["y"])
                timestamps.append(point["timestamp"] - trajectory_data["trial_start"])

        # Create scatter plot with high-contrast orange dots
        scatter = ax.scatter(
            x_coords,
            y_coords,
            s=25,  # 0.25 cm dots (approximately)
            c=timestamps,  # Color by timestamp for temporal information
            cmap="viridis",  # Use blue-to-yellow colormap
        )

        # Customize plot appearance
        ax.set_xlim(min(x_coords), max(x_coords))
        ax.set_ylim(min(y_coords), max(y_coords))
        ax.set_xticks([min(x_coords), max(x_coords)])
        ax.set_yticks([min(y_coords), max(y_coords)])

        ax.set_xticklabels(["left", "right"])
        ax.set_yticklabels(["bottom", "top"])

        ax.set_xlabel("Horizontal Position", color="white")
        ax.set_ylabel("Vertical Position", color="white")
        ax.set_title(f"Gaze Trajectory - Trial {trial_id}", color="white", pad=20)

        # Add colorbar to show temporal progression
        cbar = plt.colorbar(scatter)
        cbar.set_label("Time (seconds)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        # Add grid for better readability
        ax.grid(False)

        # Adjust tick colors
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")

        # Save plot
        trajectory_plot_path = (
            self.student_dir / "trajectories" / f"trajectory_plot_{trial_id}.png"
        )
        plt.savefig(
            trajectory_plot_path,
            facecolor="black",
            edgecolor="None",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        return str(trajectory_plot_path)

    def save_trial_data(self, trial_id=None):
        """
        Save all collected trial data to files with enhanced heatmap visualization.

        Parameters
        ----------
        trial_id : str, optional
            Unique identifier for the trial, by default None.
            If None, generates ID from current timestamp.

        Returns
        -------
        dict
            Dictionary containing Paths to saved files
        """
        # Generate trial ID if not provided
        if trial_id is None:
            trial_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.logger.info(f"Saving trial data for trial_id: {trial_id}")

        """  commenting out heatmap until I fix the code
        # Enhanced heatmap visualization
        # Normalize heatmap with increased sensitivity
        normalized_heatmap = self.heatmap.copy()

        # Apply non-linear scaling to make low values more visible
        normalized_heatmap = np.power(
            normalized_heatmap, 0.5
        )  # Square root for non-linear scaling

        # Normalize to 0-255 range with improved minimum visibility
        normalized_heatmap = cv2.normalize(
            normalized_heatmap,
            None,
            alpha=50,  # Minimum value (increased from 0)
            beta=255,  # Maximum value
            norm_type=cv2.NORM_MINMAX,
        )

        # Apply colormap with enhanced contrast
        heatmap_color = cv2.applyColorMap(
            normalized_heatmap.astype(np.uint8), cv2.COLORMAP_JET
        )

        # Blend with a dark background for better visibility
        background = np.zeros_like(heatmap_color)
        alpha = 0.7  # Opacity of heatmap
        heatmap_blend = cv2.addWeighted(heatmap_color, alpha, background, 1 - alpha, 0)

        # Save heatmap image
        heatmap_path = self.student_dir / "heatmaps" / f"heatmap_{trial_id}.png"
        cv2.imwrite(str(heatmap_path), heatmap_blend)
        """
        # Save trajectory data
        trajectory_path = (
            self.student_dir / "trajectories" / f"trajectory_{trial_id}.json"
        )
        trajectory_data = {
            "trial_id": trial_id,
            "trial_start": self.trial_start_time,
            "trial_end": time.time(),
            "bin_size_seconds": 0.125,
            "trajectory_bins": {
                str(bin_num): {
                    "bin_timestamp": bin_data["bin_timestamp"],
                    "mean_position": bin_data["mean_position"],
                    "sample_count": len(bin_data["gaze_points"]),
                    "gaze_points": bin_data["gaze_points"],
                }
                for bin_num, bin_data in sorted(self.trajectory_data.items())
            },
        }

        with open(trajectory_path, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        # Generate and save trajectory plot
        trajectory_plot_path = self.plot_trajectory(trial_id, trajectory_data)

        # Save dwell time data
        dwell_path = self.student_dir / "dwell_times" / f"dwell_{trial_id}.json"
        dwell_data = {
            "trial_id": trial_id,
            "dwell_threshold": self.dwell_threshold,
            "dwell_radius_degrees": self._pixels_to_degrees(self.dwell_radius_pixels),
            "events": self.dwell_events,
        }

        with open(dwell_path, "w") as f:
            json.dump(dwell_data, f, indent=2)

        self.logger.info(f"Successfully saved all trial data for trial_id: {trial_id}")

        # Generate dwell time plot
        dwell_plot_path = self.student_dir / "dwell_times" / f"dwell_{trial_id}.png"
        dwell_data = self.dwell_events  # assuming 'dwell_events' holds dwell data

        # Create the plot
        plt.figure(figsize=(8, 6))  # adjust figure size as needed

        # Plot points with size based on n_points and color based on duration
        for event in dwell_data:
            size = np.sqrt(event["n_points"]) * 5  # adjust scaling factor as needed
            color = self._map_duration_to_color(
                event["duration"]
            )  # define color mapping function
            plt.plot(
                event["mean_x"],
                event["mean_y"],
                marker="o",
                markersize=size,
                linestyle="",
                color=color,
            )
        mean_x = []
        mean_y = []
        if event in dwell_data:
            if "mean_x" in event and "mean_y" in event:
                mean_x.append(event["mean_x"])
                mean_y.append(event["mean_y"])
            # Customize plot (optional)
            plt.xlabel("X-coordinate (Normalized)")
            plt.ylabel("Y-coordinate (Normalized)")
            plt.title("Dwell Time Plot")
            plt.xlim(min(mean_x), max(mean_x))
            plt.ylim(min(mean_y), max(mean_y))
            plt.xticks([min(mean_x), max(mean_x)], ["LEFT", "RIGHT"])
            plt.yticks([min(mean_y), max(mean_y)], ["BOTTOM", "TOP"])
            plt.savefig(dwell_plot_path)
            plt.close()  # avoid figure stacking
        else:
            print(f"Error: Missing 'mean_x' or 'mean_y' in event: {event}")

        return {
            # "heatmap": str(heatmap_path),
            "trajectory": str(trajectory_path),
            "trajectory_plot": trajectory_plot_path,
            "dwell": str(dwell_path),
        }

    # Helper function to map dwell duration to color
    def _map_duration_to_color(self, duration):
        # Define colormap and normalization
        cmap = plt.cm.viridis
        # Adjust vmin and vmax as needed
        norm = plt.Normalize(vmin=0, vmax=self.dwell_threshold)
        # Map duration to color
        color = cmap(norm(duration))
        return color

    def calibrate(self, screen_points):
        """
        Perform gaze calibration using known screen points.
        """
        self.calibration_data = []

        # Create a window and set it to full screen
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
        cv2.moveWindow("calibration", 0, 0)
        cv2.resizeWindow("calibration", screen_width, screen_height)

        print("Calibration phase: Look at each yellow dot")
        for point in screen_points:
            point_data = []
            start_time = time.time()

            while time.time() - start_time < 5:
                # Create a fresh frame for each update
                calibration_frame = np.zeros(
                    (self.screen_height, self.screen_width, 3), dtype=np.uint8
                )

                # Draw the target point (yellow circle)
                # Flip x-coordinate for display
                point_px = (
                    int((1 - point[0]) * self.screen_width),
                    int(point[1] * self.screen_height),
                )
                cv2.circle(calibration_frame, point_px, 50, (0, 255, 255), -1)

                # Get and process the current frame
                ret, frame = self.cap.read()
                if not ret:
                    continue

                gaze_point = self.process_frame(frame)
                if gaze_point is not None:
                    # Draw the current gaze point (green circle)
                    gaze_px = (
                        int(gaze_point[0] * self.screen_width),
                        int(gaze_point[1] * self.screen_height),
                    )
                    cv2.circle(calibration_frame, gaze_px, 10, (0, 255, 0), -1)

                    # Draw a line between target and gaze point
                    cv2.line(calibration_frame, point_px, gaze_px, (255, 0, 0), 1)
                    point_data.append(gaze_point)

                # Show the frame with both points
                cv2.imshow("calibration", calibration_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyWindow("calibration")
                    return False

                time.sleep(0.1)

            if point_data:
                avg_point = np.mean(point_data, axis=0)
                self.calibration_data.append((avg_point, point))

        # fit calibration models
        if len(self.calibration_data) >= 6:
            features, targets = zip(*self.calibration_data)
            x = np.array(features)
            y_x = np.array([t[0] for t in targets])
            y_y = np.array([t[1] for t in targets])
            self.gaze_model_x.fit(x, y_x)
            self.gaze_model_y.fit(x, y_y)
            self.is_calibrated = True
            print("calibration completed successfully!")

            # validation phase
            print("validation phase: look at each point to check accuracy")
            for point in screen_points:
                calibration_frame = np.zeros(
                    (self.screen_height, self.screen_width, 3), dtype=np.uint8
                )
                point_px = (
                    int(point[0] * self.screen_width),
                    int(point[1] * self.screen_height),
                )
                # show target point
                cv2.circle(calibration_frame, point_px, 50, (0, 255, 255), -1)

                start_time = time.time()
                while time.time() - start_time < 2:  # 2 seconds validation per point
                    ret, frame = self.cap.read()
                    if not ret:
                        continue

                    gaze_point = self.process_frame(frame)
                    if gaze_point is not None:
                        # apply calibration transformation
                        calibrated_x = self.gaze_model_x.predict([gaze_point])[0]
                        calibrated_y = self.gaze_model_y.predict([gaze_point])[0]

                        # show calibrated gaze point
                        cal_px = (
                            int(calibrated_x * self.screen_width),
                            int(calibrated_y * self.screen_height),
                        )
                        # draw calibrated gaze point (green)
                        cv2.circle(calibration_frame, cal_px, 10, (0, 255, 0), -1)
                        # draw line between target and calibrated gaze
                        cv2.line(
                            calibration_frame, point_px, cal_px, (255, 255, 255), 1
                        )

                        # calculate and display error distance
                        error_distance = np.sqrt(
                            (point_px[0] - cal_px[0]) ** 2
                            + (point_px[1] - cal_px[1]) ** 2
                        )
                        error_text = f"error: {error_distance:.1f}px"
                        cv2.putText(
                            calibration_frame,
                            error_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )

                    cv2.imshow("calibration", calibration_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                time.sleep(0.1)

        else:
            print("insufficient calibration data collected.")
            return False

        cv2.destroyWindow("calibration")
        return True

    def run_session(self, duration, show_tracking=False):
        """
        Run a gaze tracking session for a specified duration.
        :param duration: duration of the session in seconds
        :param show_tracking: whether to display the gaze tracking window with the lanmark fit shown in real-time

        """
        self.session_duration = duration
        self.session_start_time = time.time()
        self.session_active = True
        # Open a new window to display the tracking
        if show_tracking:
            cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
        # Start the webcam capture
        while self.session_active:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Process the current frame
            gaze_point = self.process_frame(frame)
            # Display the gaze point on the frame
            if gaze_point is not None:
                if show_tracking:
                    display_frame = frame.copy()
                    cv2.circle(
                        display_frame,
                        (
                            int(gaze_point[0] * self.screen_width),
                            int(gaze_point[1] * self.screen_height),
                        ),
                        5,
                        (0, 255, 0),
                        -1,
                    )
                    cv2.imshow("tracking", display_frame)
            # Check if the session is complete
            if cv2.waitKey(1) & 0xFF == 27 or self.is_session_complete():
                break
        # Close the tracking window
        if show_tracking:
            cv2.destroyWindow("tracking")
        # Save the trial data
        self.save_trial_data()
        # End the session
        self.session_active = False

    # Save the trial data
    def is_session_complete(self):
        """
        check if the current session has exceeded its duration.
        :return: True if the session is complete, False otherwise
        """
        if not self.session_active:
            return False
        # Check if the session duration has elapsed
        return time.time() - self.session_start_time >= self.session_duration

    def preview_camera(self):
        """
        Open a preview window to ensure the student is properly positioned
        before starting calibration. Press spacebar to continue or ESC to exit.
        """
        print(
            "Position yourself in front of the camera and press SPACEBAR when ready. Press ESC to exit."
        )

        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Camera Preview", 0, 0)
        cv2.resizeWindow("Camera Preview", self.screen_width, self.screen_height)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                return False

            # Process frame to show face landmarks
            gaze_point = self.process_frame(frame)

            # Create a copy of the frame to draw on
            preview_frame = frame.copy()

            # Add instruction text
            cv2.putText(
                preview_frame,
                "Press SPACEBAR when ready, ESC to exit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show the processed frame
            cv2.imshow("Camera Preview", preview_frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Spacebar
                cv2.destroyWindow("Camera Preview")
                return True
            elif key == 27:  # ESC
                cv2.destroyWindow("Camera Preview")
                return False


def main():
    """
    Main function to run the integrated eye tracking system.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Integrated Eye Tracking System")
    parser.add_argument(
        "-s",
        "--student-initials",
        type=str,
        default="TEST",
        help="Student initials (e.g., JoDo for John Doe; default: TEST)",
    )
    parser.add_argument(
        "-t",
        "--trial-duration",
        type=float,
        default=5,
        help="Trial duration in minutes (default: 5)",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["5", "68"],
        default="68",
        help="facial landmark model: 5 points (faster) or 68 points (more accurate)",
    )
    parser.add_argument(
        "-ht",
        "--hide-tracking",
        action="store_true",
        help="Hide the tracking window (default: show tracking)",
    )

    args = parser.parse_args()

    if not 2 <= len(args.student_initials) <= 4 or not args.student_initials.isalpha():
        raise ValueError("Student initials must be 2-4 letters")

    try:
        tracker = IntegratedEyeTrackingSystem(
            args.student_initials,
            model_choice=args.model,
        )

        # Add preview phase
        print("\nStarting camera preview...")
        if not tracker.preview_camera():
            print("Preview cancelled or interrupted. Exiting...")
            return

        # Generate calibration points (3x3 grid)
        calibration_points = [(x, y) for x in [0.1, 0.5, 0.9] for y in [0.1, 0.5, 0.9]]

        print("\nStarting calibration sequence...")
        if tracker.calibrate(calibration_points):
            print(f"\nStarting tracking session for {args.trial_duration} minutes...")
            tracker.run_session(
                duration=args.trial_duration * 60, show_tracking=not args.hide_tracking
            )
        else:
            print("Calibration failed or was interrupted. Exiting...")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if "tracker" in locals() and hasattr(tracker, "cap"):
            tracker.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
