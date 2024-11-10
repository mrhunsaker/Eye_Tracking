import pytest
import numpy as np
import cv2
import mediapipe as mp
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
import time
from screeninfo import Monitor

# Mock classes and fixtures
@pytest.fixture
def mock_monitor():
    return Monitor(x=0, y=0, width=1920, height=1080)

@pytest.fixture
def mock_webcam():
    mock = Mock()
    mock.isOpened.return_value = True
    mock.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
    return mock

@pytest.fixture
def mock_face_mesh():
    mock = Mock()
    mock.process.return_value = Mock(
        multi_face_landmarks=[
            Mock(
                landmark=[
                    Mock(x=0.5, y=0.5),  # Add more landmarks as needed
                    Mock(x=0.51, y=0.51),
                    Mock(x=0.52, y=0.52),
                ] * 23  # To make 69 landmarks
            )
        ]
    )
    return mock

@pytest.fixture
def tracker(mock_monitor, mock_webcam, mock_face_mesh, tmp_path):
    with patch('screeninfo.get_monitors', return_value=[mock_monitor]), \
         patch('cv2.VideoCapture', return_value=mock_webcam), \
         patch('mediapipe.solutions.face_mesh.FaceMesh', return_value=mock_face_mesh):
        
        tracker = IntegratedEyeTrackingSystem(
            student_initials="TEST",
            base_dir=str(tmp_path),
            model_path=str(tmp_path / "mock_model.dat")
        )
        return tracker

# Test initialization
def test_init(tracker, tmp_path):
    """Test initialization of the eye tracking system"""
    assert tracker.screen_width == 1920
    assert tracker.screen_height == 1080
    assert tracker.student_dir == tmp_path / "TEST"
    assert (tracker.student_dir / "heatmaps").exists()
    assert (tracker.student_dir / "trajectories").exists()
    assert (tracker.student_dir / "dwell_times").exists()

# Test degree conversion functions
def test_degrees_to_pixels_conversion(tracker):
    """Test conversion between degrees and pixels"""
    # Test with default values
    pixels = tracker._degrees_to_pixels(1.0)  # 1 degree
    assert isinstance(pixels, int)
    assert pixels > 0
    
    # Test reverse conversion
    degrees = tracker._pixels_to_degrees(pixels)
    assert pytest.approx(degrees, rel=0.1) == 1.0

# Test gaze point estimation
def test_estimate_gaze_point(tracker):
    """Test gaze point estimation from eye positions"""
    left_eye = np.array([[0.4, 0.4], [0.45, 0.45]])
    right_eye = np.array([[0.5, 0.5], [0.55, 0.55]])
    head_rotation = 0.0
    
    gaze_point = tracker._estimate_gaze_point(left_eye, right_eye, head_rotation)
    assert isinstance(gaze_point, np.ndarray)
    assert len(gaze_point) == 2
    assert 0 <= gaze_point[0] <= 1
    assert 0 <= gaze_point[1] <= 1

# Test heatmap updates
def test_update_heatmap(tracker):
    """Test heatmap updating with gaze points"""
    initial_sum = np.sum(tracker.heatmap)
    tracker._update_heatmap(0.5, 0.5)  # Add point at center
    
    assert np.sum(tracker.heatmap) > initial_sum
    assert np.max(tracker.heatmap) <= 1.0
    assert np.min(tracker.heatmap) >= 0.0

# Test trajectory updates
def test_update_trajectory(tracker):
    """Test trajectory data updates"""
    x, y = 0.5, 0.5
    timestamp = time.time()
    
    tracker._update_trajectory(x, y, timestamp)
    
    # Get the bin number for this timestamp
    elapsed_time = timestamp - tracker.trial_start_time
    current_bin = int(elapsed_time * 2)
    
    assert current_bin in tracker.trajectory_data
    assert len(tracker.trajectory_data[current_bin]["gaze_points"]) == 1
    assert tracker.trajectory_data[current_bin]["mean_position"] is not None

# Test dwell time updates
def test_update_dwell_time(tracker):
    """Test dwell time tracking updates"""
    x, y = 0.5, 0.5
    timestamp = time.time()
    
    # Simulate dwelling at one point
    for i in range(10):
        tracker._update_dwell_time(x, y, timestamp + i * 0.1)
    
    assert len(tracker.dwell_events) > 0
    assert tracker.dwell_events[0]["duration"] >= tracker.dwell_threshold

# Test data saving
def test_save_trial_data(tracker, tmp_path):
    """Test saving of trial data"""
    # Add some sample data
    tracker._update_heatmap(0.5, 0.5)
    tracker._update_trajectory(0.5, 0.5, time.time())
    
    trial_id = "test_trial"
    saved_paths = tracker.save_trial_data(trial_id)
    
    assert Path(saved_paths["heatmap"]).exists()
    assert Path(saved_paths["trajectory"]).exists()
    assert Path(saved_paths["trajectory_plot"]).exists()
    assert Path(saved_paths["dwell"]).exists()
    
    # Check JSON data structure
    with open(saved_paths["trajectory"]) as f:
        trajectory_data = json.load(f)
        assert "trial_id" in trajectory_data
        assert "trajectory_bins" in trajectory_data

# Test frame processing
def test_process_frame(tracker):
    """Test processing of video frames"""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    gaze_point = tracker.process_frame(frame)
    
    assert gaze_point is not None
    assert len(gaze_point) == 2
    assert 0 <= gaze_point[0] <= 1
    assert 0 <= gaze_point[1] <= 1

# Test session completion check
def test_is_session_complete(tracker):
    """Test session completion checking"""
    tracker.session_active = True
    tracker.session_start_time = time.time() - 10  # 10 seconds ago
    tracker.session_duration = 5  # 5 seconds
    
    assert tracker.is_session_complete()
    
    tracker.session_duration = 15  # 15 seconds
    assert not tracker.is_session_complete()

# Test calibration data collection
@patch('cv2.namedWindow')
@patch('cv2.moveWindow')
@patch('cv2.resizeWindow')
@patch('cv2.imshow')
@patch('cv2.waitKey', return_value=0)
def test_calibration(mock_wait, mock_imshow, mock_resize, mock_move, mock_window, tracker):
    """Test calibration process"""
    screen_points = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
    
    with patch('time.sleep'):  # Skip actual waiting
        result = tracker.calibrate(screen_points)
    
    assert result is True
    assert tracker.is_calibrated
    assert len(tracker.calibration_data) >= len(screen_points)

# Test error handling
def test_error_handling(tracker):
    """Test error handling for invalid inputs"""
    with pytest.raises(ValueError):
        tracker._pixels_to_degrees(-1)
    
    with pytest.raises(ValueError):
        tracker._degrees_to_pixels(-1)

if __name__ == "__main__":
    pytest.main([__file__])
