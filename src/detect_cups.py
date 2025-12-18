#! /usr/bin/env python3
# -*- coding: utf8 -*-

"""
YOLO-based cup detection module.
Provides pure function to detect cups in an image.
"""

import argparse
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO


def detect_cups(image: np.ndarray, model_path: str, confidence_threshold: float = 0.5, imgsz: int = 640, device: str = "0") -> List[Tuple[List[float], int, float]]:
    """
    Detect cups in an image using YOLO.
    
    Args:
        image: Input image as numpy array (BGR format)
        model_path: Path to YOLO model weights
        confidence_threshold: Minimum confidence score to keep detection
        imgsz: Image size for inference
        device: Device to run inference on ('0' for GPU, 'cpu' for CPU)
    
    Returns:
        List of (bbox, class_id, confidence) where bbox is [x1, y1, x2, y2]
    """
    model = YOLO(model_path)
    results = model.predict(image, imgsz=imgsz, conf=confidence_threshold, device=device, verbose=False)
    
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            if conf < confidence_threshold:
                continue
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            xyxy = b.xyxy[0].tolist()
            detections.append((xyxy, cls_id, conf))
    
    return detections

def detect_empty_cups(image: np.ndarray, model: str, confidence_threshold: float = 0.5, imgsz: int = 640, device: str = "0") -> List[Tuple[List[float], int, float]]:
    """
    Detect cups in an image using YOLO.
    
    Args:
        image: Input image as numpy array (BGR format)
        model_path: Path to YOLO model weights
        confidence_threshold: Minimum confidence score to keep detection
        imgsz: Image size for inference
        device: Device to run inference on ('0' for GPU, 'cpu' for CPU)
    
    Returns:
        List of (bbox, class_id, confidence) where bbox is [x1, y1, x2, y2]
    """
    # model = YOLO(model_path)
    results = model.predict(image, imgsz=imgsz, conf=confidence_threshold, device=device, verbose=False)
    
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            if conf < confidence_threshold:
                continue
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            xyxy = b.xyxy[0].tolist()
            detections.append((xyxy, cls_id, conf))
    
    return detections


if __name__ == "__main__":
    import cv2
    import json
    
    parser = argparse.ArgumentParser(description="Detect cups in an image using YOLO")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image: {args.image}")
        exit(1)
    
    detections = detect_cups(img, args.model, args.conf)
    
    if args.json:
        output = [{"bbox": bbox, "class": cls_id, "confidence": conf} for bbox, cls_id, conf in detections]
        print(json.dumps(output, indent=2))
    else:
        print(f"Found {len(detections)} cup(s):")
        for i, (bbox, cls_id, conf) in enumerate(detections):
            print(f"  Cup {i}: class={cls_id} conf={conf:.2f} bbox={bbox}")
