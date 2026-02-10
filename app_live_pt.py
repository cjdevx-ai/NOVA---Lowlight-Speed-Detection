import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="YOLO ONNX Live Dashboard (Cam 1)", layout="wide")
st.title("YOLO ONNX Live Dashboard")
st.caption("Live inference using OpenCV VideoCapture(1) + Ultralytics YOLO ONNX.")


# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)

def open_camera(index: int, width: int | None, height: int | None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Windows-friendly
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    # Try reduce latency (not supported on all drivers)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def draw_boxes_fast(frame_bgr: np.ndarray, r0, names, show_labels=True, show_conf=True):
    out = frame_bgr.copy()
    if r0.boxes is None or len(r0.boxes) == 0:
        return out, 0

    xyxy = r0.boxes.xyxy.cpu().numpy()
    cls = r0.boxes.cls.cpu().numpy().astype(int)
    conf = r0.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_labels:
            label = names.get(c, str(c)) if isinstance(names, dict) else str(c)
            if show_conf:
                label = f"{label} {p:.2f}"
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    return out, len(xyxy)


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Model (ONNX)")
DEFAULT_ONNX = r"G:\cla_projects\NOVA\onnx\yolo8l_1k_320\best.onnx"
model_path = st.sidebar.text_input("ONNX path", value=DEFAULT_ONNX)

st.sidebar.header("Camera")
cam_index = st.sidebar.number_input("Camera index", min_value=0, max_value=10, value=1, step=1)
req_w = st.sidebar.selectbox("Request width", [None, 640, 960, 1280, 1920], index=1)
req_h = st.sidebar.selectbox("Request height", [None, 480, 540, 720, 1080], index=1)

st.sidebar.header("Inference")
conf = st.sidebar.slider("Confidence (conf)", 0.01, 1.00, 0.25, 0.01)
iou = st.sidebar.slider("IoU (NMS)", 0.10, 0.95, 0.45, 0.01)
imgsz = st.sidebar.selectbox("Image size (imgsz)", [320, 416, 512, 640, 768, 896, 1024], index=0)
max_det = st.sidebar.slider("Max detections", 1, 300, 50, 1)
show_labels = st.sidebar.checkbox("Show labels", True)
show_conf = st.sidebar.checkbox("Show confidence", True)

st.sidebar.header("Device")
device_choice = st.sidebar.selectbox("Device", ["cpu", "0 (cuda if available)"], index=0)
device = "cpu" if device_choice == "cpu" else 0

st.sidebar.header("Performance")
target_fps = st.sidebar.slider("Target FPS (throttle)", 5, 60, 30, 1)

st.sidebar.header("Snapshots")
save_snapshots = st.sidebar.checkbox("Enable snapshot saving", value=True)
snap_dir = Path(st.sidebar.text_input("Snapshot folder", value="snapshots"))
snap_prefix = st.sidebar.text_input("Snapshot prefix", value="frame")
if save_snapshots:
    snap_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([2, 1], gap="large")

with right:
    st.subheader("Controls")
    start = st.button("▶ Start", use_container_width=True)
    stop = st.button("⏹ Stop", use_container_width=True)
    snap = st.button("📸 Snapshot", use_container_width=True, disabled=not save_snapshots)

    st.divider()
    st.subheader("Stats")
    fps_box = st.empty()
    res_box = st.empty()
    det_box = st.empty()
    err_box = st.empty()

with left:
    st.subheader("Live View")
    frame_slot = st.empty()


# ----------------------------
# Session state
# ----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "last_frame_bgr" not in st.session_state:
    st.session_state.last_frame_bgr = None

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

if snap and st.session_state.last_frame_bgr is not None and save_snapshots:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = snap_dir / f"{snap_prefix}_{ts}.jpg"
    cv2.imwrite(str(out_path), st.session_state.last_frame_bgr)
    st.success(f"Saved: {out_path.as_posix()}")


# ----------------------------
# Live Loop
# ----------------------------
if st.session_state.running:
    # Load ONNX model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.session_state.running = False
        err_box.error(f"Model load error: {e}")
        st.stop()

    cap = open_camera(int(cam_index), req_w, req_h)
    if not cap.isOpened():
        st.session_state.running = False
        err_box.error(f"Could not open camera index {cam_index}. Try 0/2 or close apps using the camera.")
        st.stop()

    min_dt = 1.0 / float(target_fps)
    prev_t = time.time()
    frames = 0
    last_fps_update = time.time()

    try:
        while st.session_state.running:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                err_box.error("Camera read failed.")
                break

            st.session_state.last_frame_bgr = frame

            # Predict
            try:
                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    max_det=int(max_det),
                    device=device,
                    verbose=False
                )
            except Exception as e:
                err_box.error(f"Inference error: {e}")
                break

            r0 = results[0]
            annotated_bgr, n_det = draw_boxes_fast(frame, r0, model.names, show_labels, show_conf)

            # Render
            frame_slot.image(bgr_to_rgb(annotated_bgr), channels="RGB", use_container_width=True)

            h, w = frame.shape[:2]
            res_box.write(f"Resolution: **{w}×{h}** | Camera index: **{cam_index}** | Device: **{device_choice}**")
            det_box.write(f"Detections: **{n_det}**")

            # FPS
            frames += 1
            now = time.time()
            if (now - last_fps_update) >= 0.5:
                dt = now - prev_t
                fps = frames / dt if dt > 0 else 0.0
                fps_box.write(f"FPS: **{fps:.1f}**")
                prev_t = now
                frames = 0
                last_fps_update = now

            # Throttle
            elapsed = time.time() - t0
            sleep_t = min_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        cap.release()
        st.session_state.running = False
else:
    err_box.info("Click **Start** to begin (camera index 1).")
