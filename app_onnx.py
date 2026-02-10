import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# --- Your ONNX model path ---
ONNX_PATH = r"G:\cla_projects\NOVA\onnx\yolo8l_1k_320\best.onnx"

st.set_page_config(page_title="ONNX Object Detection (Images)", layout="wide")
st.title("ONNX Object Detection (Images Only)")
st.caption("Upload image(s) → adjust confidence → run detection → save annotated outputs.")

# ---- Sidebar controls ----
st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence threshold", 0.01, 1.00, 0.25, 0.01)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.95, 0.45, 0.01)
imgsz = st.sidebar.selectbox("Inference image size", [320, 416, 512, 640, 768, 1024], index=0)
show_labels = st.sidebar.checkbox("Show labels", value=True)
show_conf = st.sidebar.checkbox("Show confidence", value=True)
line_width = st.sidebar.slider("Box line width", 1, 8, 2, 1)

@st.cache_resource
def load_model():
    return YOLO(ONNX_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load ONNX model at:\n{ONNX_PATH}\n\nError: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

run = st.button("Run Detection", type="primary", disabled=(not uploaded))

def annotate(image_bgr: np.ndarray, result) -> np.ndarray:
    annotated = image_bgr.copy()
    names = result.names if hasattr(result, "names") else {}

    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c, p in zip(boxes, clss, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), line_width)

        if show_labels:
            label = names.get(c, str(c))
            if show_conf:
                label = f"{label} {p:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text_top = max(0, y1 - th - 8)
            cv2.rectangle(annotated, (x1, y_text_top), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(
                annotated, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )

    return annotated

all_conf_rows = []


if run:
    for i, f in enumerate(uploaded):
        # ---- Separator + header per image ----
        st.markdown("---")  # separator line
        st.subheader(f"Image {i+1}: {f.name}")

        pil = Image.open(f).convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        results = model.predict(
            source=img_rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )
        r = results[0]
                # ---- Confidence summary per image (min/mean/max) ----
                # ---- Confidence summary per image (min/mean/max) ----
        if r.boxes is not None and len(r.boxes) > 0:
            confs_arr = r.boxes.conf.cpu().numpy().astype(float)

            row = {
                "image": f.name,
                "num_detections": int(len(confs_arr)),
                "conf_min": float(np.min(confs_arr)),
                "conf_mean": float(np.mean(confs_arr)),
                "conf_max": float(np.max(confs_arr)),
            }
            all_conf_rows.append(row)

            conf_summary = pd.DataFrame([row]).copy()
            conf_summary["conf_min"] = conf_summary["conf_min"].map(lambda x: f"{x:.3f}")
            conf_summary["conf_mean"] = conf_summary["conf_mean"].map(lambda x: f"{x:.3f}")
            conf_summary["conf_max"] = conf_summary["conf_max"].map(lambda x: f"{x:.3f}")

            st.write("Confidence summary (this image):")
            st.table(conf_summary)

        else:
            row = {
                "image": f.name,
                "num_detections": 0,
                "conf_min": np.nan,
                "conf_mean": np.nan,
                "conf_max": np.nan,
            }
            all_conf_rows.append(row)

            st.write("Confidence summary (this image):")
            st.table(pd.DataFrame([{
                "image": f.name,
                "num_detections": 0,
                "conf_min": "-",
                "conf_mean": "-",
                "conf_max": "-",
            }]))

        annotated_bgr = annotate(img_bgr, r)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        c1, c2 = st.columns(2)
        with c1:
            st.image(pil, caption="Original", use_container_width=True)
        with c2:
            st.image(annotated_rgb, caption=f"Detected (conf ≥ {conf:.2f})", use_container_width=True)

        # Download annotated image
        out_pil = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")

        st.download_button(
            label=f"Download annotated: {Path(f.name).stem}_detected.png",
            data=buf.getvalue(),
            file_name=f"{Path(f.name).stem}_detected.png",
            mime="image/png",
            key=f"dl_{Path(f.name).stem}_{i}_{conf:.2f}_{iou:.2f}"
        )

        # Optional: show raw detections table
        if r.boxes is not None and len(r.boxes) > 0:
            st.write("Detections:")
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            names = r.names
            rows = []
            for (x1, y1, x2, y2), c, p in zip(boxes, clss, confs):
                rows.append({
                    "class_id": int(c),
                    "class_name": names.get(int(c), str(int(c))),
                    "confidence": float(p),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)
                })
            st.dataframe(rows, use_container_width=True)

        st.markdown("---")  # bottom separator (optional)
    
    st.markdown("## Confidence Summary (All Images)")
    all_conf_df = pd.DataFrame(all_conf_rows)

    # Show formatted view in the app
    display_df = all_conf_df.copy()
    for col in ["conf_min", "conf_mean", "conf_max"]:
        display_df[col] = display_df[col].map(lambda x: "-" if pd.isna(x) else f"{x:.3f}")
    st.dataframe(display_df, use_container_width=True)

    # CSV bytes for download (keep raw numeric values)
    csv_bytes = all_conf_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download confidence summary CSV (all images)",
        data=csv_bytes,
        file_name="confidence_summary_all_images.csv",
        mime="text/csv",
        key=f"dl_conf_summary_{len(all_conf_df)}_{conf:.2f}_{iou:.2f}"
    )

st.info(f"Loaded ONNX model: {ONNX_PATH}")
