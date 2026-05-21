# Road Pothole Detection using YOLOv8

A Streamlit-based AI web application for real-time road pothole detection using YOLOv8 Instance Segmentation and OpenCV.

The system detects potholes from images and videos, performs instance segmentation, calculates pothole area, and provides confidence analytics through an interactive and mobile-friendly interface.

--------------------------------------------------------------------------------

LIVE DEMO

https://your-streamlit-app-link.streamlit.app

--------------------------------------------------------------------------------

GITHUB REPOSITORY

https://github.com/PankajRandhawa1/yolov8-pothole-instance-segmentation

--------------------------------------------------------------------------------

FEATURES

- Real-time pothole detection
- YOLOv8 instance segmentation
- Image pothole analysis
- Video pothole analysis
- Pothole area calculation
- Confidence score analytics
- Download processed outputs
- Interactive Streamlit dashboard
- Mobile-friendly responsive UI
- Streamlit cloud deployment support

--------------------------------------------------------------------------------

TECH STACK

Programming Language
- Python

Frameworks & Libraries
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- NumPy
- PyTorch

AI/ML Concepts
- Computer Vision
- Deep Learning
- Instance Segmentation
- Object Detection

--------------------------------------------------------------------------------

PROJECT STRUCTURE

yolov8-pothole-instance-segmentation/
│
├── app.py
├── README.md
├── requirements.txt
├── packages.txt
├── yolov8segn.pt
├── road_img.jpg
│
├── notebooks/
│   └── yolov8_instance_segmentation_training.ipynb
│
├── sample_outputs/
│   └── detected_output.jpg
│
├── training_results/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── MaskF1_curve.png
│   ├── MaskPR_curve.png
│   ├── MaskP_curve.png
│   └── MaskR_curve.png
│
└── .streamlit/
    └── config.toml

--------------------------------------------------------------------------------

INSTALLATION

1. Clone Repository

git clone https://github.com/PankajRandhawa1/yolov8-pothole-instance-segmentation.git

2. Open Project Folder

cd yolov8-pothole-instance-segmentation

3. Install Dependencies

pip install -r requirements.txt

4. Run Streamlit App

streamlit run app.py

--------------------------------------------------------------------------------

SUPPORTED INPUT FORMATS

Images
- JPG
- JPEG
- PNG

Videos
- MP4

--------------------------------------------------------------------------------

FUNCTIONALITIES

IMAGE DETECTION
- Upload road images
- Detect potholes
- View segmentation masks
- Calculate pothole area
- Download processed image

VIDEO DETECTION
- Upload road videos
- Frame-by-frame pothole detection
- Instance segmentation
- Analytics generation
- Download processed video

--------------------------------------------------------------------------------

MODEL INFORMATION

- Model: YOLOv8 Instance Segmentation
- Framework: Ultralytics
- Task: Road Pothole Detection

OUTPUTS
- Segmentation Masks
- Bounding Boxes
- Area Estimation
- Confidence Scores

--------------------------------------------------------------------------------

TRAINING NOTEBOOK

The complete model training and experimentation process is available in:

notebooks/yolov8_instance_segmentation_training.ipynb

--------------------------------------------------------------------------------

FUTURE IMPROVEMENTS

- Real-time webcam detection
- GPS-based pothole mapping
- Severity classification
- Multi-road damage detection
- Database integration
- Smart city integration

--------------------------------------------------------------------------------

DEVELOPED BY

- Pankaj Randhawa
- Rohit Rattu
- Rohit Saroya

--------------------------------------------------------------------------------

TECHNOLOGIES USED

- YOLOv8 Instance Segmentation
- Streamlit Web Framework
- OpenCV Image Processing
- Deep Learning
- Computer Vision
- PyTorch

--------------------------------------------------------------------------------

LICENSE

This project is developed for educational and research purposes.

--------------------------------------------------------------------------------

ACKNOWLEDGEMENTS

- Ultralytics YOLOv8
- Streamlit Community Cloud
- OpenCV
- Python Community