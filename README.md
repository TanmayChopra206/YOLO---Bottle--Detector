# Custom YOLOv8 Object Detection Project

This project documents the end-to-end process of building a custom real-time object detector using YOLOv8. The goal was to train a model to recognize bottles in a live webcam feed.

## Learning Journey

This repository contains the full workflow, including:
- Preparing and annotating a custom dataset on Roboflow.
- Training a YOLOv8 model locally.
- Building a Python application with OpenCV to run the model on a live webcam.

**Note:** The final model's performance is limited due to a small training dataset, and it currently struggles with real-time detection. However, the scripts and structure represent a complete, functional pipeline for a computer vision project.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python main.py`

## Technologies Used
- Python
- YOLOv8
- PyTorch
- OpenCV
- Roboflow