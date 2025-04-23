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

The system uses advanced computer vision and deep learning techniques to monitor traffic and detect violations:

1. **Video Processing**: Frames are captured from a file or camera source
2. **Object Detection**: YOLOv8 detects vehicles, pedestrians, license plates, and helmets
3. **Object Tracking**: Detected objects are tracked across frames using ByteTrack or other algorithms
4. **License Plate Recognition**: License plates are detected and recognized using OCR
5. **Speed Estimation**: Vehicle speeds are calculated based on tracking and calibration
6. **Violation Detection**: The system identifies speeding vehicles and riders without helmets
7. **Database Storage**: Violations and tracks are stored in a database for later analysis
8. **Visualization**: Processed frames are annotated and displayed in real-time

The system uses YOLOv8 for object detection, but falls back to OpenCV's built-in methods (HOG for pedestrians and Haar cascades for vehicles) when pre-trained models are not available.

## Project Structure

```text
├── config.yaml           # Main configuration file
├── main.py               # Main application entry point
├── requirements.txt      # Dependencies
├── data/                 # Data directory
│   ├── datasets/         # Training datasets
│   ├── models/           # Trained models
│   └── output/           # Output files and database
└── src/                  # Source code
    ├── detection/        # Object detection modules
    ├── tracking/         # Object tracking modules
    ├── ocr/              # License plate recognition
    ├── utils/            # Utility functions
    ├── visualization/    # Visualization modules
    ├── database/         # Database management
    └── dashboard/        # Web dashboard
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
