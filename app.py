"""
YOLOv8 Object Detection — Streamlit Web App
Fine-tuned on COCO128 (80 classes: people, vehicles, animals, everyday objects)
"""
import io
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLOv8 Object Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Load model (cached so it only loads once) ──────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Only show detections above this confidence score"
    )

    iou_thresh = st.slider(
        "IOU Threshold (NMS)",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Non-maximum suppression threshold — lower = fewer overlapping boxes"
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "**Model:** YOLOv8n fine-tuned on COCO128  \n"
        "**Classes:** 80 (people, vehicles, animals, everyday objects)  \n"
        "**Framework:** Ultralytics YOLOv8  \n"
        "**Dataset:** MS COCO 128-image subset"
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔍 YOLOv8 Object Detection")
st.markdown(
    "Upload any image to detect objects with a **YOLOv8n** model fine-tuned on COCO128.  \n"
    "Detects **80 object classes** including people, vehicles, animals, food, and everyday items."
)

uploaded_file = st.file_uploader(
    "📂 Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP, WEBP"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Load model
    with st.spinner("Loading model..."):
        model = load_model()

    # Run inference
    with st.spinner("Detecting objects..."):
        start = time.time()
        results = model.predict(
            image,
            conf=confidence,
            iou=iou_thresh,
            verbose=False,
        )
        elapsed = time.time() - start

    result = results[0]
    boxes  = result.boxes

    # ── Layout: image left, stats right ──────────────────────────────────────
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("🖼️ Detection Results")

        if boxes is not None and len(boxes) > 0:
            # Annotated image (RGB numpy array)
            annotated_bgr = result.plot(line_width=2, font_size=12)
            annotated_rgb = annotated_bgr[..., ::-1]   # BGR → RGB
            st.image(annotated_rgb, use_container_width=True)
        else:
            st.image(image, use_container_width=True)
            st.info("No objects detected above the confidence threshold. Try lowering the slider.")

        st.caption(f"⏱️ Inference time: {elapsed*1000:.0f} ms  |  "
                   f"Image size: {image.width}×{image.height} px")

    with col2:
        n_det = len(boxes) if boxes is not None else 0
        st.subheader(f"📋 Detected Objects ({n_det})")

        if n_det > 0:
            # Build detection table
            rows = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                name = model.names[cls_id]
                rows.append({
                    "Object": name.capitalize(),
                    "Confidence": f"{conf_val*100:.1f}%",
                    "Conf Score": conf_val,
                })

            # Sort by confidence (highest first)
            rows.sort(key=lambda r: r["Conf Score"], reverse=True)

            # Show each detection as a card row
            for i, row in enumerate(rows):
                conf_pct = row["Conf Score"]
                bar_color = (
                    "🟢" if conf_pct >= 0.7 else
                    "🟡" if conf_pct >= 0.4 else "🔴"
                )
                st.markdown(
                    f"{bar_color} **{row['Object']}** — `{row['Confidence']}`"
                )
                st.progress(conf_pct)

            st.markdown("---")

            # Count by class
            st.markdown("**Object counts:**")
            class_counts = {}
            for row in rows:
                class_counts[row["Object"]] = class_counts.get(row["Object"], 0) + 1

            df_counts = pd.DataFrame(
                list(class_counts.items()), columns=["Class", "Count"]
            ).sort_values("Count", ascending=False).reset_index(drop=True)
            st.dataframe(df_counts, use_container_width=True, hide_index=True)

        else:
            st.info("No detections yet.")

        # Download annotated image
        if n_det > 0:
            st.markdown("---")
            annotated_pil = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            annotated_pil.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name="detection_result.jpg",
                mime="image/jpeg",
            )

else:
    # Placeholder / instructions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 **Step 1**: Upload any image using the file uploader above")
    with col2:
        st.info("🎚️ **Step 2**: Adjust the confidence threshold in the sidebar")
    with col3:
        st.info("🔍 **Step 3**: See bounding boxes, labels and confidence scores")

    st.markdown("### 🏷️ Detectable Object Classes (80)")
    coco_classes = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
        "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
        "potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush",
    ]
    st.markdown(
        " · ".join([f"`{c}`" for c in coco_classes])
    )
