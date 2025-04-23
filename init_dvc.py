#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to initialize DVC for dataset management.
"""

import os
import argparse
import subprocess
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Initialize DVC for dataset management")
    parser.add_argument("--remote-type", choices=["gdrive", "s3", "azure", "local"], default="gdrive",
                        help="Remote storage type (default: gdrive)")
    parser.add_argument("--remote-url", default=None,
                        help="Remote storage URL (default: auto-generate)")
    parser.add_argument("--remote-name", default="storage",
                        help="Remote storage name (default: storage)")
    parser.add_argument("--datasets-dir", default="data/datasets",
                        help="Datasets directory (default: data/datasets)")
    parser.add_argument("--models-dir", default="data/models",
                        help="Models directory (default: data/models)")
    return parser.parse_args()

def check_dvc_installed():
    """Check if DVC is installed."""
    try:
        subprocess.run(["dvc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def init_dvc():
    """Initialize DVC if not already initialized."""
    if os.path.exists(".dvc"):
        print("DVC already initialized")
        return True
    
    try:
        subprocess.run(["dvc", "init"], check=True)
        print("DVC initialized")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error initializing DVC: {e}")
        return False

def setup_remote(remote_type, remote_url, remote_name):
    """Set up DVC remote storage."""
    try:
        # Generate remote URL if not provided
        if remote_url is None:
            if remote_type == "gdrive":
                # Create a unique folder name for this project
                remote_url = "gdrive://traffic_surveillance_data"
            elif remote_type == "local":
                # Use a local directory
                remote_dir = os.path.abspath("dvc_remote")
                os.makedirs(remote_dir, exist_ok=True)
                remote_url = f"local://{remote_dir}"
            else:
                print(f"Error: Remote URL must be provided for {remote_type}")
                return False
        
        # Add remote storage
        subprocess.run(["dvc", "remote", "add", "--default", remote_name, remote_url], check=True)
        print(f"Added {remote_type} remote storage: {remote_url}")
        
        # Configure remote storage
        if remote_type == "gdrive":
            subprocess.run(["dvc", "remote", "modify", remote_name, "gdrive_acknowledge_abuse", "true"], check=True)
        
        return True
    
    except subprocess.SubprocessError as e:
        print(f"Error setting up remote storage: {e}")
        return False

def add_directories_to_dvc(directories):
    """Add directories to DVC."""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        try:
            # Check if directory is already tracked
            dvc_file = f"{directory}.dvc"
            if os.path.exists(dvc_file):
                print(f"Directory already tracked: {directory}")
                continue
            
            # Add directory to DVC
            subprocess.run(["dvc", "add", directory], check=True)
            print(f"Added directory to DVC: {directory}")
        
        except subprocess.SubprocessError as e:
            print(f"Error adding directory to DVC: {e}")

def create_gitignore():
    """Create .gitignore file for DVC."""
    gitignore_content = """
# DVC
/data
*.dvc
/dvc_remote

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo
"""
    
    with open(".gitignore", "a") as f:
        f.write(gitignore_content)
    
    print("Updated .gitignore file")

def main():
    """Main function."""
    args = parse_args()
    
    # Check if DVC is installed
    if not check_dvc_installed():
        print("Error: DVC is not installed")
        print("Please install DVC with: pip install dvc dvc-gdrive")
        return
    
    # Initialize DVC
    if not init_dvc():
        return
    
    # Set up remote storage
    if not setup_remote(args.remote_type, args.remote_url, args.remote_name):
        return
    
    # Add directories to DVC
    directories_to_add = [
        args.datasets_dir,
        args.models_dir
    ]
    add_directories_to_dvc(directories_to_add)
    
    # Create .gitignore file
    create_gitignore()
    
    print("\nDVC initialization complete!")
    print("\nTo push data to remote storage, run:")
    print("dvc push")
    
    if args.remote_type == "gdrive":
        print("\nThis will open a browser window for Google authentication.")

if __name__ == "__main__":
    main()
