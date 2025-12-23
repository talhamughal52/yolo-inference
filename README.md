# YOLO Model Inference and ONNX Conversion

This project demonstrates how to:
1. Run inference using a YOLO PyTorch model
2. Convert the PyTorch model to ONNX format
3. Run inference using the ONNX model
4. Compare results between PyTorch and ONNX models using IoU (Intersection over Union)

## Environment Setup

### Prerequisites
- **Python 3.8, 3.9, 3.10, 3.11, or 3.12** (recommended: 3.11 or 3.12)
  - ⚠️ **Note**: Python 3.13+ is not yet fully supported by PyTorch. If you have Python 3.13, please use Python 3.12 instead.
- pip (Python package manager)

**Installing Python 3.12 on macOS:**
```bash
brew install python@3.12
```

### Installation Steps

1. **Create a virtual environment (recommended):**
   ```bash
   # If you have Python 3.13+, use Python 3.12 instead:
   python3.12 -m venv venv  # Recommended
   # or
   python3 -m venv venv     # If using Python 3.8-3.12
   
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Simply run the main script:
```bash
python yolo_inference.py
```

The script will:
- Load the `yolo11n.pt` model
- Run inference on the available image file (`image.png`, `image-2.png`, etc.)
- Convert the model to ONNX format
- Run inference using the ONNX model
- Generate comparison visualizations

### Output Files

After running the script, you'll get:
- `yolo11n.onnx` - The converted ONNX model
- `pytorch_result.jpg` - Annotated image with PyTorch model detections
- `onnx_result.jpg` - Annotated image with ONNX model detections
- `comparison.png` - Side-by-side comparison with IoU statistics

## Project Structure

```
.
├── yolo11n.pt              # Input PyTorch model
├── image.png               # Input image (or image-2.png, etc.)
├── requirements.txt        # Python dependencies
├── yolo_inference.py       # Main inference script
├── README.md              # This file
└── [output files]         # Generated after running the script
```

## Features

### 1. PyTorch Inference
- Loads the YOLO model using Ultralytics library
- Runs inference on the input image
- Displays bounding boxes, labels, and confidence scores

### 2. ONNX Conversion
- Converts the PyTorch model to ONNX format
- Ensures compatibility and optimization

### 3. ONNX Inference
- Uses ONNX Runtime for inference
- Processes the same image and generates detections

### 4. IoU Comparison
- Matches detections between PyTorch and ONNX models
- Calculates Intersection over Union (IoU) for matched pairs
- Visualizes comparison with statistics

## Technical Details

### Dependencies
- **torch**: PyTorch framework for model loading
- **ultralytics**: YOLO model library
- **onnx**: ONNX format support
- **onnxruntime**: ONNX model inference
- **opencv-python**: Image processing
- **numpy**: Numerical operations
- **matplotlib**: Visualization

### Model Information
- Model: YOLO11n (nano variant)
- Input size: 640x640 pixels
- Output: Bounding boxes, class predictions, confidence scores

## Notes

- The script automatically detects available image files
- IoU threshold for matching detections is set to 0.5
- Confidence threshold for detections is 0.25
- The ONNX model uses CPU execution provider by default

## Troubleshooting

### Common Issues

1. **PyTorch installation fails / "No matching distribution found"**:
   - **Cause**: Python 3.13+ is not yet supported by PyTorch
   - **Solution**: Use Python 3.11 or 3.12:
     ```bash
     # Install Python 3.12
     brew install python@3.12
     
     # Remove old venv and create new one
     rm -rf venv
     python3.12 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

2. **RuntimeError: Numpy is not available / NumPy compatibility issues**:
   - **Cause**: PyTorch 2.2.2 requires NumPy 1.x, but newer opencv-python versions require NumPy 2.x
   - **Solution**: The requirements.txt is configured to use compatible versions:
     ```bash
     # Reinstall with correct versions
     pip install "numpy>=1.24.0,<2.0.0" "opencv-python>=4.8.0,<4.10.0" --force-reinstall
     ```

3. **Import errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

4. **CUDA/GPU issues**: The script uses CPU by default. For GPU support, ensure CUDA is properly installed and modify the ONNX Runtime provider if needed.

3. **Image not found**: Place your image file in the project directory with one of these names:
   - `image.png`
   - `image-2.png`
   - `image (34).png`
   - `image (35).png`

## License

This project is provided as-is for educational and demonstration purposes.

