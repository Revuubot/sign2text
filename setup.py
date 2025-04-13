#!/usr/bin/env python3
"""
Setup script for ASL Sign Language to Text Converter
This script helps set up the environment for running the application locally.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible."""
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("Error: This application requires Python 3.8 or higher.")
        sys.exit(1)
    print(f"✅ Python version {py_version.major}.{py_version.minor}.{py_version.micro} is compatible.")

def install_dependencies():
    """Install required Python dependencies."""
    print("Installing required dependencies...")
    packages = [
        "streamlit",
        "opencv-python",
        "numpy",
        "scikit-learn",
        "joblib",
        "pandas",
        "matplotlib",
        "sqlalchemy"
    ]
    
    # Optionally install PostgreSQL connector if available
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
        print("✅ PostgreSQL connector installed.")
    except subprocess.CalledProcessError:
        print("⚠️ Could not install PostgreSQL connector. History features will use SQLite instead.")
    
    # Install main dependencies
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}. Please install it manually.")
            return False
    
    return True

def check_camera():
    """Check if OpenCV can access the camera."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️ Could not access webcam. The application will run in mock mode.")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not read from webcam. The application will run in mock mode.")
            return False
        
        cap.release()
        print("✅ Webcam is accessible.")
        return True
    except Exception as e:
        print(f"⚠️ Error accessing webcam: {e}. The application will run in mock mode.")
        return False

def setup_database():
    """Set up database environment variables for local testing."""
    db_type = input("Do you want to use PostgreSQL? (y/n, default: n): ").lower()
    
    if db_type == 'y':
        print("Setting up for PostgreSQL...")
        host = input("Database host (default: localhost): ") or "localhost"
        port = input("Database port (default: 5432): ") or "5432"
        user = input("Database user: ")
        password = input("Database password: ")
        db_name = input("Database name: ")
        
        if user and password and db_name:
            db_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            
            # Save as environment variable
            if platform.system() == "Windows":
                with open("env_setup.bat", "w") as f:
                    f.write(f"set DATABASE_URL={db_url}\n")
                print("✅ Created env_setup.bat - Run this file before starting the application.")
            else:
                with open("env_setup.sh", "w") as f:
                    f.write("#!/bin/bash\n")
                    f.write(f"export DATABASE_URL={db_url}\n")
                os.chmod("env_setup.sh", 0o755)
                print("✅ Created env_setup.sh - Run 'source env_setup.sh' before starting the application.")
        else:
            print("⚠️ Missing database credentials. Will use SQLite by default.")
    else:
        print("✅ SQLite will be used for local database storage.")

def main():
    """Main setup function."""
    print("=" * 60)
    print("ASL Sign Language to Text Converter - Setup")
    print("=" * 60)
    
    check_python_version()
    
    if install_dependencies():
        print("✅ All dependencies installed successfully.")
    else:
        print("⚠️ Some dependencies could not be installed.")
    
    check_camera()
    setup_database()
    
    print("\nSetup complete! You can now run the application with:")
    print("  streamlit run app.py")
    print("\nMake sure to set environment variables if you're using PostgreSQL.")
    print("=" * 60)

if __name__ == "__main__":
    main()