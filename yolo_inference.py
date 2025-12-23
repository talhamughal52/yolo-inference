"""
YOLO Model Inference and ONNX Conversion Script

This script:
1. Loads a YOLO PyTorch model and runs inference on an image
2. Converts the model to ONNX format
3. Runs inference using the ONNX model
4. Compares results using IoU (Intersection over Union)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_detections(pytorch_dets: List[Dict], onnx_dets: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Match detections between PyTorch and ONNX models based on IoU.
    
    Args:
        pytorch_dets: List of detections from PyTorch model
        onnx_dets: List of detections from ONNX model
        iou_threshold: Minimum IoU to consider a match
    
    Returns:
        List of matched detections with IoU scores
    """
    matches = []
    used_onnx = set()
    
    for pt_idx, pt_det in enumerate(pytorch_dets):
        best_match = None
        best_iou = 0.0
        best_onnx_idx = -1
        
        pt_box = pt_det['box']
        
        for onnx_idx, onnx_det in enumerate(onnx_dets):
            if onnx_idx in used_onnx:
                continue
            
            onnx_box = onnx_det['box']
            iou = calculate_iou(pt_box, onnx_box)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = onnx_det
                best_onnx_idx = onnx_idx
        
        if best_match is not None:
            used_onnx.add(best_onnx_idx)
            matches.append({
                'pytorch': pt_det,
                'onnx': best_match,
                'iou': best_iou
            })
    
    return matches


def run_pytorch_inference(model_path: str, image_path: str) -> Tuple[List[Dict], np.ndarray]:
    """
    Run inference using PyTorch YOLO model.
    
    Args:
        model_path: Path to the .pt model file
        image_path: Path to the input image
    
    Returns:
        Tuple of (detections list, annotated image)
    """
    print(f"\n{'='*60}")
    print("Running PyTorch Inference")
    print(f"{'='*60}")
    
    # Load model
    model = YOLO(model_path)
    print(f"Loaded PyTorch model from: {model_path}")
    
    # Run inference
    results = model(image_path)
    
    # Extract detections
    detections = []
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            label = result.names[cls]
            
            detections.append({
                'box': box,
                'confidence': conf,
                'class': cls,
                'label': label
            })
            
            print(f"  Detection {i+1}: {label} (conf: {conf:.3f}) at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Create annotated image
    annotated_img = results[0].plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    print(f"\nTotal detections: {len(detections)}")
    
    return detections, annotated_img_rgb


def convert_to_onnx(model_path: str, output_path: str = None) -> str:
    """
    Convert PyTorch YOLO model to ONNX format.
    
    Args:
        model_path: Path to the .pt model file
        output_path: Optional path for output ONNX file
    
    Returns:
        Path to the converted ONNX model
    """
    print(f"\n{'='*60}")
    print("Converting Model to ONNX")
    print(f"{'='*60}")
    
    if output_path is None:
        output_path = str(Path(model_path).with_suffix('.onnx'))
    
    # Load model
    model = YOLO(model_path)
    
    # Export to ONNX
    print(f"Exporting model to ONNX format...")
    model.export(format='onnx', imgsz=640, simplify=True)
    
    # The export method saves the file, get the actual path
    onnx_path = str(Path(model_path).with_suffix('.onnx'))
    
    if os.path.exists(onnx_path):
        print(f"✓ ONNX model saved to: {onnx_path}")
        return onnx_path
    else:
        raise FileNotFoundError(f"ONNX conversion failed. Expected file at: {onnx_path}")


def run_onnx_inference(onnx_path: str, image_path: str) -> Tuple[List[Dict], np.ndarray]:
    """
    Run inference using ONNX model.
    Uses Ultralytics YOLO to load ONNX model for proper preprocessing/postprocessing.
    
    Args:
        onnx_path: Path to the .onnx model file
        image_path: Path to the input image
    
    Returns:
        Tuple of (detections list, annotated image)
    """
    print(f"\n{'='*60}")
    print("Running ONNX Inference")
    print(f"{'='*60}")
    
    # Load ONNX model using Ultralytics (handles preprocessing/postprocessing)
    print(f"Loading ONNX model from: {onnx_path}")
    model = YOLO(onnx_path)
    
    # Run inference
    results = model(image_path)
    
    # Extract detections
    detections = []
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            label = result.names[cls]
            
            detections.append({
                'box': box,
                'confidence': conf,
                'class': cls,
                'label': label
            })
            
            print(f"  Detection {i+1}: {label} (conf: {conf:.3f}) at [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Create annotated image
    annotated_img = results[0].plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    print(f"\nTotal detections: {len(detections)}")
    
    return detections, annotated_img_rgb


def visualize_comparison(pytorch_img: np.ndarray, onnx_img: np.ndarray, 
                        pytorch_dets: List[Dict], onnx_dets: List[Dict],
                        matches: List[Dict], output_path: str = "comparison.png"):
    """
    Visualize and compare PyTorch and ONNX model predictions.
    
    Args:
        pytorch_img: Annotated image from PyTorch model
        onnx_img: Annotated image from ONNX model
        pytorch_dets: Detections from PyTorch model
        onnx_dets: Detections from ONNX model
        matches: Matched detections with IoU scores
        output_path: Path to save the comparison visualization
    """
    print(f"\n{'='*60}")
    print("Creating Comparison Visualization")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PyTorch results
    axes[0].imshow(pytorch_img)
    axes[0].set_title(f'PyTorch Model\n({len(pytorch_dets)} detections)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # ONNX results
    axes[1].imshow(onnx_img)
    axes[1].set_title(f'ONNX Model\n({len(onnx_dets)} detections)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # IoU comparison chart
    if matches:
        iou_scores = [m['iou'] for m in matches]
        axes[2].bar(range(len(iou_scores)), iou_scores, color='steelblue', alpha=0.7)
        axes[2].axhline(y=0.5, color='r', linestyle='--', label='IoU Threshold (0.5)')
        axes[2].set_xlabel('Matched Detection Pair', fontsize=10)
        axes[2].set_ylabel('IoU Score', fontsize=10)
        axes[2].set_title(f'IoU Comparison\n(Mean IoU: {np.mean(iou_scores):.3f})', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.1])
    else:
        axes[2].text(0.5, 0.5, 'No matches found\n(IoU < 0.5)', 
                    ha='center', va='center', fontsize=12)
        axes[2].set_title('IoU Comparison', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to: {output_path}")
    
    # Print IoU statistics
    if matches:
        iou_scores = [m['iou'] for m in matches]
        print(f"\nIoU Statistics:")
        print(f"  Mean IoU: {np.mean(iou_scores):.3f}")
        print(f"  Median IoU: {np.median(iou_scores):.3f}")
        print(f"  Min IoU: {np.min(iou_scores):.3f}")
        print(f"  Max IoU: {np.max(iou_scores):.3f}")
        print(f"  Matched pairs: {len(matches)}")
        print(f"  PyTorch-only detections: {len(pytorch_dets) - len(matches)}")
        print(f"  ONNX-only detections: {len(onnx_dets) - len(matches)}")


def main():
    """Main function to run the complete pipeline."""
    # Configuration
    model_path = "yolo11n.pt"
    
    # Try to find an image file (prioritize image.png, then use first available)
    image_files = ["image.png", "image-2.png", "image (34).png", "image (35).png"]
    image_path = None
    
    for img_file in image_files:
        if os.path.exists(img_file):
            image_path = img_file
            break
    
    if image_path is None:
        raise FileNotFoundError("No image file found. Please provide image.png or similar.")
    
    print(f"Using image: {image_path}")
    print(f"Using model: {model_path}")
    
    # Step 1: PyTorch Inference
    pytorch_detections, pytorch_img = run_pytorch_inference(model_path, image_path)
    
    # Step 2: Convert to ONNX
    onnx_path = convert_to_onnx(model_path)
    
    # Step 3: ONNX Inference
    onnx_detections, onnx_img = run_onnx_inference(onnx_path, image_path)
    
    # Step 4: Compare results
    matches = match_detections(pytorch_detections, onnx_detections, iou_threshold=0.5)
    
    # Step 5: Visualize comparison
    visualize_comparison(pytorch_img, onnx_img, pytorch_detections, 
                        onnx_detections, matches, "comparison.png")
    
    # Save individual results
    cv2.imwrite("pytorch_result.jpg", cv2.cvtColor(pytorch_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("onnx_result.jpg", cv2.cvtColor(onnx_img, cv2.COLOR_RGB2BGR))
    
    print(f"\n{'='*60}")
    print("✓ All tasks completed successfully!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - ONNX model: {onnx_path}")
    print(f"  - PyTorch result: pytorch_result.jpg")
    print(f"  - ONNX result: onnx_result.jpg")
    print(f"  - Comparison: comparison.png")


if __name__ == "__main__":
    main()

