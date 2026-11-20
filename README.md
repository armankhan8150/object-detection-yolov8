# 🔍 YOLOv8 Object Detection — Streamlit Web App

A production-ready object detection web app built with **YOLOv8n** fine-tuned on the COCO128 dataset, deployed on Streamlit Community Cloud.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL_HERE)

---

## Project Overview

| Item | Detail |
|------|--------|
| **Model** | YOLOv8n (pretrained on COCO, fine-tuned on COCO128) |
| **Dataset** | COCO128 — 128 labeled images, 80 object classes |
| **Training** | 25 epochs, `imgsz=640`, `batch=16` |
| **Framework** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| **UI** | Streamlit |
| **Classes** | 80 (people, vehicles, animals, food, everyday objects) |

---

## App Features

- **📂 Image Upload**: drag and drop JPG, PNG, BMP, WEBP
- **🎚️ Confidence Threshold Slider**: adjust detection sensitivity (0.05 – 1.00)
- **🔲 IOU Threshold Slider**: control non-maximum suppression
- **🖼️ Annotated Output**: bounding boxes with class labels drawn on the image
- **📋 Detection List**: every detected object with confidence score + colour-coded progress bar
- **📊 Class Count Table**: summary of how many of each class were found
- **⬇️ Download Button**: save the annotated image as JPEG

---

## Training Details

**Dataset:** [MS COCO 128](https://github.com/ultralytics/assets): 128 real-world images covering 80 object categories, the standard YOLO benchmark subset.

**Fine-tuning command (from the notebook):**
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")          # pretrained backbone
results = model.train(
    data="datasets/coco128.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    patience=0,                     # run all 25 epochs
)
```

**Metrics (see `training/yolo_training.ipynb` for full per-epoch table):**

| Metric | Value |
|--------|-------|
| Final mAP50 | _see notebook_ |
| Final mAP50-95 | _see notebook_ |
| Model size | ~6 MB (YOLOv8n) |

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/yolo-object-detection.git
cd yolo-object-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ensure the model is present
The fine-tuned weights are committed at `models/best.pt`. If you want to retrain:
```bash
# Download COCO128 dataset first, place under datasets/coco128/
# Then open training/yolo_training.ipynb and run all cells
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub (make sure `models/best.pt` is committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, main file `app.py`
4. Click **Deploy**

The app auto-installs `requirements.txt` on each deploy.

---

## Dependencies

```
ultralytics==8.4.216   # YOLOv8 training + inference
streamlit>=1.35.0      # web UI
Pillow>=10.0.0         # image I/O
pandas>=2.0.0          # detection table
numpy>=1.24.0          # array ops
```

---

## Detectable Classes (80 COCO)

`person` · `bicycle` · `car` · `motorcycle` · `airplane` · `bus` · `train` · `truck` · `boat` · `traffic light` · `fire hydrant` · `stop sign` · `bench` · `bird` · `cat` · `dog` · `horse` · `sheep` · `cow` · `elephant` · `bear` · `zebra` · `giraffe` · `backpack` · `umbrella` · `handbag` · `tie` · `suitcase` · `frisbee` · `skis` · `snowboard` · `sports ball` · `kite` · `baseball bat` · `baseball glove` · `skateboard` · `surfboard` · `tennis racket` · `bottle` · `wine glass` · `cup` · `fork` · `knife` · `spoon` · `bowl` · `banana` · `apple` · `sandwich` · `orange` · `broccoli` · `carrot` · `hot dog` · `pizza` · `donut` · `cake` · `chair` · `couch` · `potted plant` · `bed` · `dining table` · `toilet` · `tv` · `laptop` · `mouse` · `remote` · `keyboard` · `cell phone` · `microwave` · `oven` · `toaster` · `sink` · `refrigerator` · `book` · `clock` · `vase` · `scissors` · `teddy bear` · `hair drier` · `toothbrush`

---

## Author

Built as part of an AI Engineering portfolio — demonstrating fine-tuning, object detection, and production Streamlit deployment.

> **GitHub:** [armankhan8150](https://github.com/armankhan8150)
