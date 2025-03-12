# Wet Wipe Object Detection Using YOLO  

## Overview  
This project focuses on training a **YOLO (You Only Look Once)** model for object detection on **wet wipe** product images. The dataset used is from Kaggle and contains labeled images of wet wipe packages. The trained model aims to accurately detect and classify wet wipe products in images.  

## Dataset  
The dataset is available on Kaggle:  
🔗 **[Wet Wipe Dataset](https://www.kaggle.com/datasets/erhanbaran/wet-wipe)**  

It consists of:  
- **Train Set:** Images and annotations for model training  
- **Validation Set:** Images for model evaluation  
- **Test Set:** Separate images for final testing  

## Model & Training  
The project utilizes **YOLO (Ultralytics YOLOv8)** for object detection. The training process includes:  
1. **Preprocessing:**  
   - Image resizing and augmentation  
   - Converting annotations into YOLO format  

2. **Model Training:**  
   - Using the **YOLOv8** model  
   - Fine-tuning with transfer learning  
   - Evaluating performance with validation data  

3. **Inference & Testing:**  
   - Running the model on test images  
   - Visualizing object detection results  

## Installation & Requirements  
Ensure you have the required dependencies installed:  
```bash
pip install ultralytics opencv-python numpy matplotlib
```

## Training the Model  
Run the following script to train the YOLO model:  
```python
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano version

# Train the model on the dataset
model.train(data="path/to/dataset.yaml", epochs=50, imgsz=640)
```

## Running Inference  
To test the trained model:  
```python
results = model("path/to/test_image.jpg")  
results.show()
```

## Results  
The model outputs bounding boxes for wet wipe products in images with confidence scores.  

## Future Improvements  
- Fine-tuning the model for better accuracy  
- Expanding the dataset with more variations  
- Deploying the model as a web application or mobile app  
