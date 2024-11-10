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
from scipy.spatial import distance
from sklearn.svm import SVR
from scipy.signal import savgol_filter
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
    """
    try:
        if Path(model_path).exists():
            print(f"Model file already exists at {model_path}")
            return True

        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = f"{model_path}.bz2"
        
        print("Downloading facial landmarks model...")
        print("This may take a few minutes depending on your internet connection.")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        
        with open(compressed_path, "wb") as f:
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
        with bz2.open(compressed_path) as fr, open(model_path, "wb") as fw:
            fw.write(fr.read())
        os.remove(compressed_path)
        print("Download and extraction complete!")
        return True
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

class IntegratedEyeTrackingSystem:
    """
    Integrated eye tracking system combining basic and advanced features.
    """
    def __init__(
        self,
        student_initials,
        base_dir="eye_tracking_data",
        dwell_threshold=1.0,
        dwell_radius_deg=2.0,
        model_path="./eyetracking/shape_predictor_68_face_landmarks.dat",
        calibration_points=9
    ):
        # Get primary monitor dimensions
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height

        # Set up logging
        now = datetime.datetime.now()
        filename = f"eye_tracking_{now:%Y%m%d_%H%M%S}.log"
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

        # Initialize face detection and tracking
        if not Path(model_path).exists():
            self.logger.info("Facial landmarks model not found. Attempting to download...")
            if not download_landmarks_model(model_path):
                raise RuntimeError("Failed to download facial landmarks model")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Initialize data directories
        self.student_dir = Path(base_dir) / student_initials.upper()
        self.student_dir.mkdir(parents=True, exist_ok=True)
        (self.student_dir / "heatmaps").mkdir(exist_ok=True)
        (self.student_dir / "trajectories").mkdir(exist_ok=True)
        (self.student_dir / "dwell_times").mkdir(exist_ok=True)

        # Initialize tracking data
        self.heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
        self.last_head_rotation = None
        self.trajectory_data = defaultdict(list)
        self.trial_start_time = time.time()
        self.dwell_threshold = dwell_threshold
        self.dwell_radius_pixels = self._degrees_to_pixels(dwell_radius_deg)
        self.current_dwell = {"start_time": None, "location": None, "points": []}
        self.dwell_events = []

        # Initialize camera parameters
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

        # Advanced features initialization
        self.session_duration = None
        self.session_start_time = None
        self.session_active = False
        self.calibration_points = calibration_points
        self.calibration_data = []
        self.gaze_model_x = SVR(kernel='rbf', C=1000)
        self.gaze_model_y = SVR(kernel='rbf', C=1000)
        self.is_calibrated = False
        self.EAR_THRESHOLD = 0.2
        self.MIN_EAR_FRAMES = 2
        self.smoothing_window = 5
        self.gaze_history = []

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)

    # [Previous helper methods remain the same]
    def _degrees_to_pixels(self, degrees, screen_distance_cm=60, screen_width_cm=34.5):
        pixels_per_cm = self.screen_width / screen_width_cm
        cm_per_degree = screen_distance_cm * np.tan(np.radians(degrees))
        return int(cm_per_degree * pixels_per_cm)

    def calibrate(self, screen_points):
        """
        Perform gaze calibration using known screen points.
        """
        self.calibration_data = []
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        
        for point in screen_points:
            calibration_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            point_px = (
                int(point[0] * self.screen_width),
                int(point[1] * self.screen_height)
            )
            cv2.circle(calibration_frame, point_px, 10, (0, 255, 0), -1)
            cv2.imshow("Calibration", calibration_frame)
            
            point_data = []
            start_time = time.time()
            
            while time.time() - start_time < 3:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                gaze_point = self.process_frame(frame)
                if gaze_point is not None:
                    point_data.append(gaze_point)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyWindow("Calibration")
                    return False
                
                time.sleep(0.1)
            
            if point_data:
                avg_point = np.mean(point_data, axis=0)
                self.calibration_data.append((avg_point, point))
        
        if len(self.calibration_data) >= 6:
            features, targets = zip(*self.calibration_data)
            X = np.array(features)
            y_x = np.array([t[0] for t in targets])
            y_y = np.array([t[1] for t in targets])
            
            self.gaze_model_x.fit(X, y_x)
            self.gaze_model_y.fit(X, y_y)
            self.is_calibrated = True
            print("Calibration completed successfully!")
        else:
            print("Insufficient calibration data collected.")
            return False
        
        cv2.destroyWindow("Calibration")
        return True

    def run_session(self, duration, show_tracking=False):
        """
        Run a complete eye tracking session.
        """
        self.session_duration = duration
        self.session_start_time = time.time()
        self.session_active = True
        
        if show_tracking:
            cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        
        while self.session_active:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            gaze_point = self.process_frame(frame)
            
            if gaze_point is not None:
                if show_tracking:
                    display_frame = frame.copy()
                    cv2.circle(
                        display_frame,
                        (int(gaze_point[0] * self.screen_width),
                         int(gaze_point[1] * self.screen_height)),
                        5, (0, 255, 0), -1
                    )
                    cv2.imshow("Tracking", display_frame)
            
            if cv2.waitKey(1) & 0xFF == 27 or self.is_session_complete():
                break
        
        if show_tracking:
            cv2.destroyWindow("Tracking")
        
        self.save_trial_data()
        self.session_active = False

    def is_session_complete(self):
        """Check if the current session has exceeded its duration."""
        if not self.session_active:
            return False
        return time.time() - self.session_start_time >= self.session_duration

    # [Rest of the methods from both classes remain the same]

def main():
    """
    Main function to run the integrated eye tracking system.
    """
    parser = argparse.ArgumentParser(description="Integrated Eye Tracking System")
    parser.add_argument(
        "student_initials",
        type=str,
        help="Student initials (e.g., JD for John Doe)"
    )
    parser.add_argument(
        "-t",
        "--trial-duration",
        type=float,
        default=5,
        help="Trial duration in minutes (default: 5)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./eyetracking/shape_predictor_68_face_landmarks.dat",
        help="Path to facial landmarks model file"
    )
    parser.add_argument(
        "--hide-tracking",
        action="store_true",
        help="Hide the tracking window"
    )
    args = parser.parse_args()

    if not 2 <= len(args.student_initials) <= 4 or not args.student_initials.isalpha():
        raise ValueError("Student initials must be 2-4 letters")

    try:
        tracker = IntegratedEyeTrackingSystem(
            args.student_initials,
            model_path=args.model_path
        )
        
        # Generate calibration points (3x3 grid)
        calibration_points = [
            (x, y) for x in [0.1, 0.5, 0.9]
            for y in [0.1, 0.5, 0.9]
        ]
        
        print("Starting calibration sequence...")
        if tracker.calibrate(calibration_points):
            print(f"\nStarting tracking session for {args.trial_duration} minutes...")
            tracker.run_session(
                duration=args.trial_duration * 60,
                show_tracking=not args.hide_tracking
            )
        else:
            print("Calibration failed or was interrupted. Exiting...")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if hasattr(tracker, 'cap'):
            tracker.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
