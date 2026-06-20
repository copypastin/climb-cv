# climb-cv | climbing with computer vision


![GitHub last commit](https://img.shields.io/github/last-commit/copypastin/climb-cv?style=for-the-badge) ![GitHub License](https://img.shields.io/github/license/copypastin/climb-cv?style=for-the-badge) ![GitHub contributors](https://img.shields.io/github/contributors/copypastin/climb-cv?style=for-the-badge)



<img width="800" height="400" alt="climb-cv" src="assets/header.gif" />

read the [journal](./JOURNAL.md) for ongoing updates and development!

## concept
the main concept of this project is to create a package of computer vision tools for climbing.

- this project aims to explore the potential of computer vision in climbing and provide a platform for climbers to leverage visual data to enhance the climbing technology. 
- by analyzing visual data, we can gain insights into climbing techniques, developing consistent routesetting, and even predict climbing routes based on visual cues.

## implementations / tech

stack is compartmentalized into 3 distinct modules:
1. **Video streaming and processing**
>  **OpenCV** for image processing and analysis
2. **Pose estimation and analysis**
>  [MediaPipe](https://github.com/google/mediapipe) for pose estimation
> Data points are extracted and filtered to remove noise and improve accuracy
> Results are rendered in 3d space using **matplotlib**    
3. **Climb hold prediction and visualization**
>  Implements [YOLO model](https://docs.ultralytics.com/models/yolo26#overview) for object detection
> 


<img width="800" height="354" alt="climb2" src="assets/climb1.gif" />


## requirements
- python 3.8+
- macbook M2 or newer (required for lid angle sensor)
    - swift compiler

## additional tech & references
- [Crux Lab Reconstruct](https://github.com/cruxbetalabs/reconstruct)
- [LidAngleSensor](https://github.com/samhenrigold/LidAngleSensor)
