import cv2
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO

def load_image(image_path):
    """Load an image from the given path."""
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def display_image(image, detections, model):
    """Display the image with bounding boxes."""
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class label
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def detect_objects(image_path):
    """Detect objects in an image using YOLOv8."""
    # Load the YOLO model
    global model
    model = YOLO("yolov8n.pt")  # Replace 'yolov8n.pt' with a larger model if needed

    # Load the image
    image = load_image(image_path)

    # Run object detection
    results = model(image)

    # Extract detections
    detections = results[0].boxes  # Access the Boxes object

    # Display results
    display_image(image, detections, model)


# Specify the path to your image
image_path = r"C:\Users\DELL\Pictures\Screenshots\Screenshot 2024-11-06 144109.png" # Replace with your image path
detect_objects(image_path)
