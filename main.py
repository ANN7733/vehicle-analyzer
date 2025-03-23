import logging

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a FastAPI application
app = FastAPI()

# Determine the device (GPU or CPU) for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model for car detection
yolo_model = YOLO("yolo11m.pt").to(device)

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)


def count_red_cars(image: np.ndarray, car_boxes: list, red_threshold: float = 0.15) -> int:
    """
    Counts the number of red cars in the image based on HSV color space.

    Args:
        image (np.ndarray): Input image in BGR format.
        car_boxes (list): List of bounding boxes for detected cars.
        red_threshold (float): Threshold for red pixel ratio to classify a car as red.

    Returns:
        int: Number of red cars.
    """
    red_cars = 0

    for box in car_boxes:
        x1, y1, x2, y2 = box
        car_image = image[y1:y2, x1:x2]

        # Convert to HSV for color analysis
        hsv_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)

        # Define red color ranges in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 100, 100])
        upper_red2 = np.array([185, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Calculate red pixel ratio
        red_pixel_count = cv2.countNonZero(mask)
        total_pixel_count = car_image.shape[0] * car_image.shape[1]
        red_ratio = red_pixel_count / total_pixel_count

        # Classify as red if ratio exceeds threshold
        if red_ratio > red_threshold:
            red_cars += 1

    return red_cars


def detect_cars(image: np.ndarray, imgsz: int = 640, conf: float = 0.2) -> tuple[int, int, list]:
    """
    Detects cars in the image using YOLO and counts red cars.

    Args:
        image (np.ndarray): Input image in BGR format.
        imgsz (int): Size of the image for YOLO inference. Default is 640.
        conf (float): Confidence threshold for YOLO detection. Default is 0.2.

    Returns:
        tuple: Total cars, red cars, and bounding boxes.
    """
    # Run YOLO model inference on the image
    results = yolo_model(image, imgsz=imgsz, conf=conf)
    car_boxes = []

    # Extract bounding boxes for detected cars
    for result in results:
        for box in result.boxes:
            if box.cls == 2:  # Class 2 corresponds to cars in YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_boxes.append((x1, y1, x2, y2))

    # Count the number of red cars
    red_cars = count_red_cars(image, car_boxes)
    total_cars = len(car_boxes)

    return total_cars, red_cars, car_boxes


def generate_description(image: np.ndarray) -> str:
    """
    Generates a textual description of the image using the BLIP-2 model.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        str: Textual description of the image.
    """
    # Convert the image from BGR to RGB and then to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Preprocess the image and generate a description using the BLIP model
    inputs = processor(image_pil, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    description = "The image depicts " + description

    return description

@app.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...)) -> dict:
    """
    API endpoint to analyze an image and return car counts and description.

    Args:
        image (UploadFile): Image file uploaded by the user.

    Returns:
        dict: JSON response with total cars, red cars, and description.
    """
    # Check if the uploaded file is an image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read and decode the image
    image_data = await image.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Analyze the image: detect cars and generate a description
    total_cars, red_cars, _ = detect_cars(image_cv)
    description = generate_description(image_cv)

    # Return the results as a JSON response
    return {
        "total_cars": total_cars,
        "red_cars": red_cars,
        "description": description,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)