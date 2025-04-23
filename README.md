# Advanced Traffic Surveillance System

A comprehensive traffic surveillance application that detects and tracks vehicles and pedestrians, recognizes license plates, detects helmets, and monitors speed violations. The system provides both a desktop GUI and a web dashboard for real-time monitoring and analysis.

## Features

- **Object Detection**: Real-time detection of vehicles, pedestrians, license plates, and helmets
- **Object Tracking**: Track objects across frames with unique IDs
- **License Plate Recognition**: Detect and recognize license plates on vehicles
- **Helmet Detection**: Identify whether motorcycle riders are wearing helmets
- **Speed Estimation**: Calculate and monitor vehicle speeds
- **Violation Detection**: Identify traffic violations (speeding, no helmet)
- **Database Storage**: Store violations and tracks for later analysis
- **Vector Database**: Search for similar images using vector embeddings
- **Web Dashboard**: Monitor traffic and analyze violations through a web interface
- **Real-time Visualization**: Display processed frames with annotations and statistics

## Requirements

- Python 3.7+
- PyTorch and Torchvision
- OpenCV
- NumPy
- Ultralytics YOLOv8
- EasyOCR or Tesseract (for license plate recognition)
- Dash and Plotly (for web dashboard)
- FAISS (for vector database)
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

### Main Application

Run the main application with:

```bash
python main.py [options]
```

Options:

- `--config`: Path to configuration file (default: config.yaml)
- `--source`: Path to video file or camera index
- `--output`: Path to output directory
- `--dashboard`: Launch the web dashboard
- `--debug`: Enable debug mode

Press 'q' to quit the application.

### Configuration

The system is configured through the `config.yaml` file, which includes settings for:

- Detection models and parameters
- Tracking algorithms
- OCR engines
- Speed estimation calibration
- Database settings
- Visualization options
- Dashboard configuration

### Web Dashboard

The web dashboard is automatically launched when using the `--dashboard` option. You can access it by opening your browser and navigating to `http://localhost:8050` (or the port specified in the configuration).

The dashboard provides:

- Live video feed with annotations
- List of recent violations with details
- Statistics and charts for analysis
- Search functionality for finding similar violations

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
