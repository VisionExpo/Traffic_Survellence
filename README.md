# Traffic Surveillance System

A real-time traffic surveillance application that detects and counts vehicles and pedestrians in video streams. The system provides both a desktop GUI and a web interface for monitoring traffic.

## Features

- Real-time vehicle and pedestrian detection
- Live counting of traffic objects
- Information panel with date, time, and statistics
- Simulated real-time processing with FPS control
- Both desktop GUI and web interface options
- Sample video generation for testing

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Flask (for web interface)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/VisionExpo/Traffic_Survellence.git
   cd Traffic_Survellence
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Desktop Application

Run the desktop application with:

```bash
python traffic_surveillance.py [options]
```

Options:

- `--source`, `-s`: Path to video file (default: generates a sample video)
- `--confidence`, `-c`: Minimum confidence threshold for detections (0-1)
- `--fps`, `-f`: Target FPS for display
- `--width`, `-W`: Display window width
- `--height`, `-H`: Display window height

Press 'q' to quit the application.

### Web Application

Run the web application with:

```bash
python web_app.py [options]
```

Options:

- `--source`, `-s`: Path to video file (default: generates a sample video)
- `--confidence`, `-c`: Minimum confidence threshold for detections (0-1)
- `--fps`, `-f`: Target FPS for processing
- `--port`, `-p`: Port for the web server (default: 5000)

Then open your browser and navigate to `http://localhost:5000`.

## How It Works

The system uses computer vision techniques to detect and track vehicles and pedestrians in video streams:

1. Video frames are captured from a file or camera
2. Each frame is processed by the object detector
3. Detected objects are annotated with bounding boxes
4. Statistics are calculated and displayed
5. Processed frames are shown in real-time

The application uses OpenCV's built-in detection methods (HOG for pedestrians and Haar cascades for vehicles) when pre-trained models are not available.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
