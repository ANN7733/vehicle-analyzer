# Vehicle Image Analyzer

This project provides a RESTful API for analyzing images of urban streets. It detects cars, identifies red cars, and generates a textual description of the image using state-of-the-art machine learning models.

## Features

- **Car Detection**: Detects all cars in the image using YOLOv11.
- **Red Car Identification**: Identifies and counts red cars using color thresholding in the HSV color space.
- **Image Description**: Generates a textual description of the image using the BLIP model.

---

## Project Setup

### Prerequisites

- Python 3.8 or higher
- Poetry (for dependency management)
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ANN7733/vehicle-analyzer.git
   cd vehicle-analyzer
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Download YOLOv11 weights**:
   - The YOLOv11 model weights (`yolo11m.pt`) are automatically downloaded when you run the script for the first time.

4. **Set up environment variables** (optional):
   - If you want to use a GPU, ensure CUDA is installed and configured.

---

## Usage

### Running the API

1. Start the FastAPI server:
   ```bash
   poetry run python main.py
   ```

2. The API will be available at `http://0.0.0.0:8000`.

### Making a Request

Send a `POST` request to the `/analyze-image` endpoint with an image file:

```bash
curl -X POST -F "image=@example.jpg" http://localhost:8000/analyze-image
```

### Example Response

```json
{
  "total_cars": 17,
  "red_cars": 2,
  "description": "The image depicts a parking lot full of cars"
}
```
---

## Approach

### Car Detection

- **Model**: YOLOv11 (You Only Look Once) is used for object detection.
- **Process**:
  - The image is passed through the YOLOv11 model to detect objects.
  - Only objects classified as "cars" (class ID 2) are considered.
  - Bounding boxes for detected cars are extracted.

### Red Car Identification

- **Process**:
  - For each detected car, the bounding box region is extracted from the image.
  - The region is converted to the HSV color space for better color segmentation.
  - A mask is created to identify red pixels using predefined HSV ranges for red.
  - If the ratio of red pixels exceeds a threshold, the car is classified as red.

### Image Description Generation

- **Model**: BLIP (Bootstrapped Language-Image Pretraining, base version) is used for image captioning.
- **Process**:
  - The image is passed through the BLIP model.
  - The model generates a textual description summarizing the content of the image.

---

## Directory Structure

```
vehicle-analyzer/
├── main.py                # Main script for the FastAPI application
├── pyproject.toml         # Poetry dependencies and project configuration
├── README.md              # Project documentation
└── example.jpg            # Example image for testing the API
```

---

## Dependencies

- **FastAPI**: Web framework for building the API.
- **OpenCV**: Image processing and color space conversion.
- **YOLOv11**: Object detection model for car detection.
- **BLIP**: Image captioning model for generating descriptions.
- **PyTorch**: Deep learning framework for running models on CPU/GPU.

---

## Notes

- **Performance**: For faster inference, use a GPU. The script automatically detects and uses CUDA if available.
- **Customization**: You can modify the `red_threshold` in the `count_red_cars` function to adjust the sensitivity for red car detection.
- **Model Weights**: YOLOv11 weights are downloaded automatically. BLIP weights are loaded from Hugging Face's model hub.

---
