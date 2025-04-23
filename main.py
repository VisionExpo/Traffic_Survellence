#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Advanced Traffic Surveillance System.
"""

import os
import argparse
import yaml
import logging
from datetime import datetime

# Import modules
from src.detection.detector import ObjectDetector
from src.tracking.tracker import ObjectTracker
from src.ocr.license_plate import LicensePlateRecognizer
from src.utils.video import VideoProcessor
from src.visualization.visualizer import VisualizationEngine
from src.database.db_manager import DatabaseManager
from src.dashboard.app import create_dashboard

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Traffic Surveillance System")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to video file or camera index")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output directory")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch the web dashboard")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    return parser.parse_args()

def setup_logging(config, debug=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if debug else getattr(logging, config["logging"]["level"])
    log_file = config["logging"]["file"]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("traffic_surveillance")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_logging(config, args.debug)
    logger.info("Starting Advanced Traffic Surveillance System")
    
    # Override config with command line arguments
    if args.source:
        config["video_source"] = args.source
    if args.output:
        config["paths"]["output"] = args.output
    
    # Create output directory
    os.makedirs(config["paths"]["output"], exist_ok=True)
    
    # Initialize components
    logger.info("Initializing system components")
    
    # Initialize detector
    detector = ObjectDetector(
        model_type=config["detection"]["model_type"],
        model_size=config["detection"]["model_size"],
        confidence_threshold=config["detection"]["confidence_threshold"],
        classes=config["detection"]["classes"]
    )
    
    # Initialize tracker
    tracker = ObjectTracker(
        algorithm=config["tracking"]["algorithm"],
        max_age=config["tracking"]["max_age"],
        min_hits=config["tracking"]["min_hits"],
        iou_threshold=config["tracking"]["iou_threshold"]
    )
    
    # Initialize license plate recognizer
    lpr = LicensePlateRecognizer(
        engine=config["ocr"]["engine"],
        languages=config["ocr"]["languages"]
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(
        db_type=config["database"]["type"],
        db_path=config["database"]["path"]
    )
    
    # Initialize visualization engine
    visualizer = VisualizationEngine(
        window_name=config["visualization"]["window_name"],
        width=config["visualization"]["width"],
        height=config["visualization"]["height"],
        show_detections=config["visualization"]["show_detections"],
        show_tracks=config["visualization"]["show_tracks"],
        show_speed=config["visualization"]["show_speed"],
        show_license_plates=config["visualization"]["show_license_plates"],
        show_helmet_status=config["visualization"]["show_helmet_status"]
    )
    
    # Initialize video processor
    video_processor = VideoProcessor(
        source=config.get("video_source", 0),
        detector=detector,
        tracker=tracker,
        lpr=lpr,
        visualizer=visualizer,
        db_manager=db_manager,
        speed_limit=config["speed"]["speed_limit"],
        calibration_factor=config["speed"]["calibration_factor"],
        output_path=os.path.join(config["paths"]["output"], f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    )
    
    # Start processing
    if args.dashboard and config["dashboard"]["enabled"]:
        # Start dashboard
        logger.info("Starting web dashboard")
        create_dashboard(
            video_processor=video_processor,
            db_manager=db_manager,
            host=config["dashboard"]["host"],
            port=config["dashboard"]["port"],
            theme=config["dashboard"]["theme"]
        )
    else:
        # Process video
        logger.info("Starting video processing")
        video_processor.process()
    
    logger.info("Advanced Traffic Surveillance System stopped")

if __name__ == "__main__":
    main()
