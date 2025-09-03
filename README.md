# Football Tracking System

Football analysis system using YOLO and OpenCV for ball possession tracking and speed analysis.

## Features

- **Object Detection & Tracking**: YOLO11x + ByteTrack algorithm
- **Team Assignment**: K-Means++ clustering based on colors
- **Ball Possession**: Distance-based assignment with cubic interpolation
- **Speed**: Pixel-to-meter conversion with perspective transformation
- **Camera Movement**: Lucas-Kanade optical flow 
- **View Transformation**: Homography mapping for field coordinates

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and setup model weights (see `training/readme.md` for details)

## Usage

Run the main script:
```bash
python main.py
```

Enter the path to your football video when prompted. The processed video will be saved in the `output_videos/` directory.




## Short video demo

![Football Tracking Demo](demo/demo_01.gif)



## DATA FOR TESTING

You can download video in [Bundesliga 460 MP4 Videos](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv/data)  to demo.