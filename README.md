# YOLO Object Detection for Hotwheels cars using custom Dataset

Custom-trained YOLO models (v8 & v11) for object detection using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework.

## Sample

![Sample Input](sample.jpeg)

## Project Structure

```
yolo-push/
├── yolo_v8/              # YOLOv8 training results
│   ├── my_model_v8.pt    # Trained weights
│   ├── results.png       # Training metrics
│   ├── confusion_matrix.png
│   └── ...               # Curves & batch samples
├── yolo_v11/             # YOLOv11 training results
│   ├── my_model_v11.pt   # Trained weights
│   ├── results.png       # Training metrics
│   ├── confusion_matrix.png
│   └── ...               # Curves & batch samples
├── data/                 # Training dataset
│   ├── images/           # 199 annotated images
│   ├── labels/           # YOLO-format annotations
│   ├── classes.txt       # Class names
│   └── notes.json        # Dataset metadata
├── sample.jpeg           # Sample input image
├── detect.py             # Inference script
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run detection on an image:
```bash
python detect.py --source sample.jpeg
```

Use a specific model version:
```bash
python detect.py --source image.jpg --model yolo_v8
python detect.py --source image.jpg --model yolo_v11
```

Adjust confidence threshold:
```bash
python detect.py --source image.jpg --conf 0.5
```

Save results:
```bash
python detect.py --source image.jpg --save
```

Webcam:
```bash
python detect.py --source 0
```

## Dataset

Custom dataset of **199 images** across **10 car classes**, manually annotated using [Label Studio](https://labelstud.io/):

| # | Class |
|---|-------|
| 0 | Aston Martin Valkyrie |
| 1 | Audi Quattro |
| 2 | Camaro |
| 3 | Honda Civic |
| 4 | Hyundai I20 WRC |
| 5 | Mercedes-Benz 560 SEC AMG |
| 6 | Mitsubishi Pajero Evolution |
| 7 | Porsche Carrera |
| 8 | Tesla Model S Plaid |
| 9 | Toyota GR Supra |

## Training Results

Both models were trained and evaluated. See the training output directories (`yolo_v8/`, `yolo_v11/`) for:

- **Confusion Matrices** — Classification accuracy per class
- **PR / F1 Curves** — Precision-Recall and F1 score curves
- **Training Batches** — Sample training and validation batches
- **Results Summary** — Overall training metrics

## Built With

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Label Studio](https://labelstud.io/) — Annotation
- Python 3.8+
- tutorial referred to -(https://youtu.be/r0RspiLG260?si=FbCmOFXcuKIiC_67)
