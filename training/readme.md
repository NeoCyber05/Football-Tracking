# Model Weights

This directory contains the trained model weights for the football tracking system.


### 1. YOLO11x Base Model
- **File**: `yolo11x.pt`
- **Download**: [YOLO11x Official](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)
- **Size**: ~68MB
- **Purpose**: Base object detection model

### 2. Custom Trained Model
- **File**: `best_2.pt` (train in 2 dataset - currently using)
- **File**: `best.pt` (train in 1 dataset)

## Dataset

The custom models were trained on football-specific datasets:

### Dataset 1
- **Source**: [Roboflow Dataset 1](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)
- **Used for**: `best.pt` model
- **Classes**: Player, Ball, Goalkeeper, Referee

### Dataset 2  
- **Source**: [Roboflow Dataset 2](https://universe.roboflow.com/football-gozni/football-player-detection-bfswn/dataset/1)
- **Used for**: `best_2.pt` model
- **Classes**: Player, Ball, Goalkeeper, Referee
- **Note**: Combined dataset for improved accuracy

*Replace the Roboflow links above with your actual dataset URLs*
## Usage

The main script (`main.py`) loads the custom model:
```python
tracker = Tracker('training/best_2.pt')
```
