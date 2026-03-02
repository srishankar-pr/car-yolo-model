"""
YOLO Object Detection Script
Runs inference using trained YOLOv8 or YOLOv11 model weights.

Usage:
    python detect.py --source image.jpg
    python detect.py --source image.jpg --model yolo_v11
    python detect.py --source video.mp4 --conf 0.5
    python detect.py --source 0  (webcam)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, video, or '0' for webcam",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo_v8", "yolo_v11"],
        default="yolo_v11",
        help="Model version to use (default: yolo_v11)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to runs/detect/",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Display results in a window (default: True)",
    )
    args = parser.parse_args()

    # Select model weights based on chosen version
    model_paths = {
        "yolo_v8": Path("yolo_v8/my_model_v8.pt"),
        "yolo_v11": Path("yolo_v11/my_model_v11.pt"),
    }
    weights = model_paths[args.model]

    if not weights.exists():
        print(f"Error: Model weights not found at '{weights}'")
        return

    # Load model
    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    # Handle webcam input
    source = 0 if args.source == "0" else args.source

    # Run inference
    results = model.predict(
        source=source,
        conf=args.conf,
        save=args.save,
        show=args.show,
    )

    # Print summary
    for result in results:
        boxes = result.boxes
        print(f"\nDetected {len(boxes)} object(s):")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            confidence = float(box.conf[0])
            print(f"  - {cls_name}: {confidence:.2f}")


if __name__ == "__main__":
    main()
