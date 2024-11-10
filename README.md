# Eye Tracking: A Python Program for Analyzing Student Gaze Patterns

This program is designed to track the eye gaze of students with Cerebral Palsy in a classroom setting. It utilizes MediaPipe and dlib libraries to achieve real-time eye tracking and analyze gaze patterns. It is capable of tracking eye gaze with nonstandard head postures.

## Features:

-   Tracks student eye movements using MediaPipe's face mesh model.
-   estimates gaze point location on the screen.
-   Generates heatmaps to visualize dwell times and areas of focus.
-   Records gaze trajectory data for further analysis.
-   Saves data in JSON format for each trial.

## Installation:

0. Ensure you have the development version of the Tcl/Tk library installed on your system (required for matplotlib and mediapipe and python is not automatically installed with the python-tk package whe installed using PyEnv and/or many package managers):

```bash
sudo apt-get install tk-dev # debian/ubuntu-based systems
	# or
sudo dnf install tk-devel # Red Hat/Alma Linux/Rocky Linux
    # or
sudo yum install tk-devel # Red Hat/Alma Linux/Rocky Linux
    # or
sudo dnf5 install tk-devel # Fedora-based systems
    # or
sudo zypper install tk-devel # OpenSuse-based systems
    # or
sudo pacman -S tk-dev # Arch-based systems
    # or
sudo emerge tk-dev # Gentoo-based systems
    # or
sudo apk add tk-dev # Alpine-based systems
```

1. Ensure you have Python 3.12.7+ installed (I use PyEnv to manage Python versions on Fedora). You may have to install additional dependencies for Python 3.12.7 as well as build dependencies:

```bash
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.12.7
```

2. Install necessary libraries using pip (pypoetry and jupyter):

```bash
python -m pip install poetry jupyter ruff
```

3. Clone the repository and move to the project directory:

```bash
git clone git@github.com:mrhunsaker/Eye_Tracking.git # by ssh
	# or
git clone https://github.com/mrhunsaker/Eye_Tracking.git # using https
	# or
https://github.com/mrhunsaker/Eye_Tracking/archive/refs/heads/main.zip # download the zip file
```

4. Install dependencies using Poetry:

```bash
cd ./Eye_Tracking # cd into the folder where the pyproject.toml file is located. The program will be run from this directory as well
poetry install
```

## Usage:

1. Run from the ./Eye_Tracking directory.

```bash
cd ./Eye_Tracking
```

2. Run the eye tracker for student 'XxXx' and a 5-minute trial:

```bash
poetry run python ./eyetracking/eye-tracker.py XxXx --trial-duration 5  #  visual tracking window ON with visual feedback of shape landmarks ON
poetry run python ./eyetracking/eye-tracker.py XxXx --trial-duration 5 --model-path ./eyetracking/shape_predictor_68_face_landmarks.dat --hide-tracking  # Track student 'RyHu' for 5 minutes, specify model path, hide tracking window to avoid visual distractions
```

## Explanation of Command-Line Arguments:

-   `student_initials`: Initials of the student being tracked (2-4 letters).
-   `-t`, `--trial-duration`: Duration of the tracking session in minutes (default: 5 minutes).
-   `--model-path`: Path to the facial landmarks model file (default: ./Eye_Tracking/eyetracking/shape_predictor_68_face_landmarks.dat).
-   `--hide-tracking`: Hides the window displaying facial landmarks (useful for data collection by mitigating student distraction).

## Data Output:

The program saves tracking data for each session in a directory named after the student's initials ('XxXx'). This directory includes:

```bash
./Eye_Tracking/eye_tracking_data/XxXx/
```

The files are saved with a timestamp in the filename to avoid overwriting data from previous trials. The following files are saved:

-   Heatmap image (visualizes gaze intensity). <- This is still in development and not useful for times under 10 minutes
-   Trajectory data (JSON file containing gaze point timestamps and coordinates).
-   Dwell time data (JSON file recording durations of fixations).

## Jupyter Notebook-Based Plotting

The program includes a Jupyter notebook for plotting gaze trajectories and heatmaps. To use it:

```bash
cd ./Eye_Tracking/jupyter_plotting
jupyter lab # for using Jupyter Lab
    # or
jupyter notebook # for using Jupyter Notebook
```

-   Open the `eye_tracking.ipynb` notebook.
-   Change the Following line from 'YyYy' to the student's initials ('XxXx') and the trial timestamp:

```python
"data_dir = "../eye_tracking_data/YyYy/trajectories"\n" # original
"data_dir = "../eye_tracking_data/XxXx/trajectories"\n" # modified for student XxXx
```

-   Press the 'Run' button on thej top of the cell to execute the code blocks and visualize the data.
-   This will generate a plot of the gaze trajectory and a heatmap of the student's gaze pattern.
-   The plot is also saved in the same directory as the data files.

## Development

### Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

3. Create a new branch for your feature or bug fix:

```bash
git checkout -b feature-or-bugfix-branch
```

4. Make your changes and test thoroughly.

### Submitting Changes

1. Commit your changes:

```bash
git commit -m "Your descriptive commit message"
```

2. Push to your fork:

```bash
git push origin feature-or-bugfix-branch
```

3. Submit a pull request:
    - Go to the Pull Requests tab on GitHub.
    - Click the "New Pull Request" button.
    - Select the branch with your changes.
4. Describe your changes in detail, providing context and reasons for the changes

#### Code Style

I try my best to follow the styles enforced by the [Black](https://black.readthedocs.io/en/stable/) code formatter [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black). Please do your best to follow our coding style guidelines to maintain consistency across the project.

### Reporting Issues

If you encounter any issues or have suggestions, please [open an issue on GitHub](https://github.com/mrhunsaker/StudentDataGUI/issues). Provide as much detail as possible, including your operating system and relevant configuration.

### Development Workflow

-   Before starting to work on an issue, make sure it's not already assigned or being worked on.
-   If you plan major changes, it's a good idea to open an issue for discussion first.

### Code of Conduct

Everyone participating in the Black project, and in particular in the issue tracker, pull requests, and social media activity, is expected to treat other people with respect and more generally to follow the guidelines articulated in the [Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).
