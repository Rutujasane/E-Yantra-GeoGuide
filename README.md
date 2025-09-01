# Autonomous Bot Navigation

This project is an autonomous robot that navigates a custom arena using real-time computer vision and a machine learning model. The system identifies various "events" on the arena floor, plans a priority-based route, and guides the robot to each location.

![Project Demo](https://drive.google.com/file/d/1FSJBo8isUng6pxaYMCx-LnSWOEKhXnYs/view?usp=sharing)


---

## How It Works

The project operates through a complete perception-to-action pipeline:

1.  **See & Understand:** An overhead camera captures the arena. OpenCV processes the video feed, corrects the perspective, and tracks the robot's position using ArUco markers.

2.  **Classify Events:** A pre-trained PyTorch (InceptionV3) model analyzes the camera feed to identify and classify 5 different types of events on the arena floor.

3.  **Plan the Route:** Based on the importance of the identified events, a path-planning algorithm dynamically creates the most efficient route for the robot to follow.

4.  **Execute the Mission:** The main Python script sends movement commands over WiFi to the ESP32-powered robot, which navigates the arena grid to complete its objectives.

## Key Features

-   **Real-time Video Processing:** Live environment perception using OpenCV.
-   **ArUco Marker Detection:** Precise perspective warping and robot localization.
-   **Machine Learning:** Employs a PyTorch model to classify events on the arena, including:
    -   🔥 Fire
    -   💣 Combat
    -   🏚️ Destroyed Buildings
    -   🚑 Humanitarian Aid
    -   🚛 Military Vehicles
-   **Priority-Based Path Planning:** Dynamically creates a navigation path based on event severity.
-   **Wireless Control:** A Python script on a host computer commands the ESP32 robot over WiFi.

## Technology Stack

-   **Main Language:** Python 3
-   **Computer Vision:** OpenCV
-   **Machine Learning:** PyTorch, Torchvision
-   **Firmware:** C++ (Arduino Framework)
-   **Microcontroller:** ESP32

## Codebase Overview
    .
    ├── esp32_code/
    │ └── esp32_code.ino # Firmware for the ESP32 microcontroller
    ├── src.py # Main Python script for vision and control
    ├── best_model.pth # The pre-trained PyTorch classification model
    └── requirements.txt # Python libraries used in the project