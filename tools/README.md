# Skin Tone Detection System

A real-time skin tone detection application using OpenCV and MediaPipe for facial landmark detection.

## Features

- Real-time webcam face detection
- Facial landmark extraction using MediaPipe
- Skin region extraction (cheeks, forehead)
- Average RGB and HSV color calculation
- Skin tone classification into 6 categories:
  - Fair
  - Light
  - Medium
  - Olive
  - Brown
  - Dark
- Real-time display with overlays
- Console output with color values

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

No additional files needed! MediaPipe includes all required models.

## Usage

Run the skin tone detection script:
```bash
python skin_tone_detector.py
```

- Press 'q' to quit the application
- Make sure your face is well-lit and visible to the webcam
- The application will display real-time results in both the console and video window

## How it Works

1. **Face Detection**: Uses MediaPipe's face mesh to locate faces and extract 468 facial landmarks
2. **Landmark Detection**: Extracts detailed facial landmarks using MediaPipe's face mesh model
3. **Skin Region Extraction**: Defines regions of interest (cheeks, forehead) based on landmark positions
4. **Color Analysis**: Extracts pixel values from skin regions and filters for skin-like colors using HSV ranges
5. **Classification**: Classifies skin tone based on HSV values using predefined thresholds
6. **Display**: Shows real-time results with facial landmarks, skin regions, and classification

## Requirements

- Python 3.7+
- Webcam
- Good lighting conditions

## Notes

- The application uses a buffer system to smooth out skin tone detection results
- Skin tone classification is based on HSV color space analysis
- The system handles cases where no face is detected gracefully
- Results are displayed both in the console and as overlays on the video feed
