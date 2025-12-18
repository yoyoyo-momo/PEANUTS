#! /usr/bin/env python3
# -*- coding: utf8 -*-

"""
Train YOLO model for cup detection.
Requires dataset in YOLO format with data.yaml config.
"""

import argparse
from ultralytics import YOLO


def train_yolo(data_yaml, model='yolov8n.pt', epochs=50, imgsz=640, batch=16, device='0'):
    """
    Train YOLO model on custom dataset.
    
    Args:
        data_yaml: Path to data.yaml config file
        model: Pretrained model to start from (yolov8n/s/m/l/x.pt)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to train on ('0' for GPU, 'cpu' for CPU)
    """
    # Load pretrained model
    model = YOLO(model)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project='runs/detect',
        name='train',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save=True,
        save_period=-1,
        cache=False,
        workers=8,
        patience=50,
    )
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model saved to: runs/detect/train/weights/best.pt")
    print(f"Last model saved to: runs/detect/train/weights/last.pt")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for cup detection")
    parser.add_argument("data", help="Path to data.yaml config file")
    parser.add_argument("--model", default="yolo11n.pt", 
                        help="Base model (yolo11n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, 
                        help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, 
                        help="Batch size (reduce if GPU memory issues)")
    parser.add_argument("--device", default="0", 
                        help="Device to train on (0 for GPU, cpu for CPU)")
    
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
