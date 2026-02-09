import cv2, time, math, threading
import numpy as np
import supervision as sv
import onnxruntime as ort
from collections import defaultdict, deque

# =========================
# Local-scale Speedometer (OLD MATH)
# =========================
class LocalScaleSpeedometer:
    """
    Computes speed per-frame using centroid displacement (pixels) converted to meters via a mapper.
    Uses median smoothing over the last few speed estimates.
    """
    def __init__(self, mapper, fps, unit=3.6, window=5, max_kph=300):
        self.mapper = mapper
        self.fps = float(fps)
        self.unit = float(unit)
        self.window = int(window)
        self.max_kph = float(max_kph)
        self.pos_hist = defaultdict(lambda: deque(maxlen=self.window))
        self.speed_hist = defaultdict(lambda: deque(maxlen=8))

    def _local_mpp(self, point):
        cx, cy = int(round(point[0])), int(round(point[1]))
        img_pts = np.array([[cx, cy], [cx + 1, cy], [cx, cy + 1]], dtype=np.float32)
        try:
            world_pts = self.mapper.map(img_pts)
        except Exception:
            return (1e-6, 1e-6)

        w00, wx, wy = world_pts[0], world_pts[1], world_pts[2]
        mpp_x = float(np.linalg.norm(wx - w00)) or 1e-6
        mpp_y = float(np.linalg.norm(wy - w00)) or 1e-6
        return (mpp_x, mpp_y)

    def update_with_centroid(self, frame_idx: int, track_id: int, centroid: tuple):
        tid = int(track_id)
        cx, cy = int(round(centroid[0])), int(round(centroid[1]))
        self.pos_hist[tid].append((cx, cy))

        if len(self.pos_hist[tid]) < 2:
            return

        (x_prev, y_prev), (x_curr, y_curr) = self.pos_hist[tid][-2], self.pos_hist[tid][-1]
        dx_px = float(x_curr - x_prev)
        dy_px = float(y_curr - y_prev)

        mid = ((x_prev + x_curr) / 2.0, (y_prev + y_curr) / 2.0)
        mpp_x, mpp_y = self._local_mpp(mid)

        dx_m = dx_px * mpp_x
        dy_m = dy_px * mpp_y
        ds_m = math.hypot(dx_m, dy_m)

        m_s = ds_m * self.fps
        kph = m_s * self.unit

        if kph < 0:
            kph = 0.0

        if kph > self.max_kph:
            if self.speed_hist[tid]:
                kph = float(self.speed_hist[tid][-1])
            else:
                kph = float(min(kph, self.max_kph))

        self.speed_hist[tid].append(kph)

    def get_speed(self, track_id: int):
        tid = int(track_id)
        if not self.speed_hist[tid]:
            return 0
        arr = np.array(self.speed_hist[tid], dtype=float)
        return int(round(float(np.median(arr))))

    def reset(self, track_id: int):
        tid = int(track_id)
        if tid in self.pos_hist:
            self.pos_hist[tid].clear()
        if tid in self.speed_hist:
            self.speed_hist[tid].clear()

# =========================================
#  Simple mapper using 8m / gate pixel dist
# =========================================
class SimpleScaleMapper:
    """
    Minimal mapper with .map() compatible with LocalScaleSpeedometer.
    Uses constant meters-per-pixel derived from the 8m calibration.
    """
    def __init__(self, meters_per_pixel: float):
        self.mpp = float(meters_per_pixel)

    def map(self, image_pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 2)
        return pts * self.mpp

# ---------------- CONFIG ----------------
ONNX_PATH = r"G:\cla_projects\NOVA\yolo11nano_5k_320\best.onnx"
CAMERA_INDEX = 1

# Detection thresholds
CONF_TH = 0.55
IOU_TH = 0.50
TOPK_PRE_NMS = 120

# Inference resolution
IMG_SIZE = 320

# Run detection every N frames (tracker uses cached detections between)
INFER_EVERY = 2

STALE_RESET_FRAMES = 60

# Gates/ROI (must match your live camera framing)
GATE_A = ((140, 230), (661, 132))
GATE_B = ((1207, 367), (370, 715))
DISTANCE_M = 10.0

POLYGON = np.array([
    [140, 230],
    [661, 132],
    [1207, 367],
    [370, 715]
], dtype=np.int32)

USE_ROI = True
# ---------------------------------------

# ======================
# DirectML ORT Session
# ======================
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1

sess = ort.InferenceSession(
    ONNX_PATH,
    sess_options=so,
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)

inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name

# Some broken installs shadow ort.get_available_providers; keep prints guarded
try:
    print("Available providers:", ort.get_available_providers())
except Exception:
    print("Available providers: (could not query)")

print("Session providers:", sess.get_providers())
if "DmlExecutionProvider" not in sess.get_providers():
    raise RuntimeError("DirectML provider not active. This run is CPU.")

# ======================
# Frame buffer thread
# ======================
class LatestFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.ts = 0.0

    def set(self, frame):
        with self.lock:
            self.frame = frame
            self.ts = time.time()

    def get(self):
        with self.lock:
            return self.frame, self.ts

latest = LatestFrame()
stop = threading.Event()

def capture_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {CAMERA_INDEX}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Force 1280x720 (OBS virtual camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while not stop.is_set():
        ok, frame = cap.read()
        if ok:
            latest.set(frame)
        else:
            time.sleep(0.01)

    cap.release()

threading.Thread(target=capture_loop, daemon=True).start()

# ======================
# Geometry helpers
# ======================
def centroid_xy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (0.5*(x1+x2), 0.5*(y1+y2))

def inside_roi(pt, polygon):
    return cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0

def midpoint(p, q):
    return np.array([(p[0]+q[0])/2.0, (p[1]+q[1])/2.0], dtype=np.float32)

# ======================
# Calibrate meters-per-pixel from gates (OLD METHOD)
# ======================
A_mid = midpoint(*GATE_A)
B_mid = midpoint(*GATE_B)
gate_px_dist = float(np.linalg.norm(B_mid - A_mid))
if gate_px_dist < 1e-6:
    raise RuntimeError("Gate midpoints are identical; cannot calibrate scale.")

mpp = DISTANCE_M / gate_px_dist
mapper = SimpleScaleMapper(mpp)

# Speedometer uses runtime FPS estimate (updated every second)
speedometer = LocalScaleSpeedometer(mapper, fps=30.0, window=5, max_kph=300)
print(f"[calibration] gate_px_dist={gate_px_dist:.2f}px, mpp={mpp:.6f} m/px")

# ======================
# YOLO ONNX utils
# ======================
def letterbox(im, new_shape, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_shape - nw, new_shape - nh
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    im_padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return im_padded, r, left, top

def iou_xyxy(a, b):
    xx1 = np.maximum(a[0], b[:, 0])
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    return inter / (area_a + area_b - inter + 1e-9)

def nms(boxes, scores, iou_th=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_xyxy(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_th]
    return keep

def onnx_detect(frame_bgr):
    """
    Returns boxes_xyxy (float32 Nx4) and conf (float32 N) in ORIGINAL frame coords.
    """
    img, r, pad_x, pad_y = letterbox(frame_bgr, IMG_SIZE)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]

    y = sess.run([out], {inp: x})[0]
    y = np.squeeze(y)
    if y.ndim == 2 and y.shape[0] < y.shape[1]:
        y = y.T  # (N,C)

    xywh = y[:, 0:4]
    scores = y[:, 4:]
    conf = scores.max(axis=1)

    m = conf >= CONF_TH
    if not np.any(m):
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    xywh = xywh[m]
    conf = conf[m]

    cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = cx - w/2; y1 = cy - h/2
    x2 = cx + w/2; y2 = cy + h/2
    boxes = np.stack([x1,y1,x2,y2], axis=1)

    # undo letterbox
    boxes[:, [0,2]] -= pad_x
    boxes[:, [1,3]] -= pad_y
    boxes /= r

    H, W = frame_bgr.shape[:2]
    boxes[:,0] = np.clip(boxes[:,0], 0, W-1)
    boxes[:,2] = np.clip(boxes[:,2], 0, W-1)
    boxes[:,1] = np.clip(boxes[:,1], 0, H-1)
    boxes[:,3] = np.clip(boxes[:,3], 0, H-1)

    # top-k before NMS
    if conf.size > TOPK_PRE_NMS:
        idx = np.argsort(conf)[::-1][:TOPK_PRE_NMS]
        boxes = boxesավորմ०ละ= boxes[idx]
        conf = conf[idx]

    keep = nms(boxes, conf, IOU_TH)
    boxes = boxes[keep].astype(np.float32)
    conf = conf[keep].astype(np.float32)
    return boxes, conf

def boxes_to_sv_detections(boxes_xyxy, conf):
    if boxes_xyxy.shape[0] == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=boxes_xyxy.astype(np.float32),
        confidence=conf.astype(np.float32),
        class_id=np.zeros((boxes_xyxy.shape[0],), dtype=int),
    )

# ======================
# Tracker
# ======================
tracker = sv.ByteTrack(frame_rate=30)

# Runtime FPS estimate (updates speedometer.fps so the old math stays correct)
fps_est = 30.0
t_last_fps = time.time()
fps_counter = 0

last_seen = {}
frame_idx = 0

# Cache detections between infer frames
last_det = sv.Detections.empty()
last_inf_ms = 0.0

try:
    while True:
        frame, ts = latest.get()
        if frame is None:
            time.sleep(0.005)
            continue

        # FPS estimate (and update speedometer fps)
        fps_counter += 1
        tnow = time.time()
        if tnow - t_last_fps >= 1.0:
            fps_est = fps_counter / (tnow - t_last_fps)
            fps_counter = 0
            t_last_fps = tnow
            speedometer.fps = max(1.0, float(fps_est))

        # Inference gate
        did_infer = (frame_idx % INFER_EVERY == 0)

        if did_infer:
            t0 = time.perf_counter()
            boxes, conf = onnx_detect(frame)
            last_inf_ms = (time.perf_counter() - t0) * 1000
            last_det = boxes_to_sv_detections(boxes, conf)

        # Track every frame using cached detections
        detections = tracker.update_with_detections(last_det)

        # Draw ROI + gates
        cv2.polylines(frame, [POLYGON], True, (255, 0, 0), 2)
        cv2.line(frame, GATE_A[0], GATE_A[1], (0, 255, 255), 3)
        cv2.line(frame, GATE_B[0], GATE_B[1], (0, 165, 255), 3)

        if detections.tracker_id is not None and len(detections) > 0:
            for i, tid in enumerate(detections.tracker_id):
                tid = int(tid)
                xyxy = detections.xyxy[i]
                cx, cy = centroid_xy(xyxy)

                # Reset stale IDs (ID reuse protection)
                if tid in last_seen and (frame_idx - last_seen[tid] > STALE_RESET_FRAMES):
                    speedometer.reset(tid)
                last_seen[tid] = frame_idx

                # ROI gate
                if USE_ROI and not inside_roi((cx, cy), POLYGON):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
                    cv2.putText(frame, f"VHCL {tid}", (x1, max(y1-10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                    continue

                # OLD speed math: update every frame (not only when did_infer)
                speedometer.update_with_centroid(frame_idx, tid, (cx, cy))
                spd = speedometer.get_speed(tid)

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
                cv2.putText(frame, f"VHCL {tid}, SPD: {spd} km/h",
                            (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        cv2.putText(frame,
                    f"FPS: {fps_est:.1f} | infer(ms): {last_inf_ms:.1f} | every {INFER_EVERY} | imgsz:{IMG_SIZE}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

        cv2.imshow("NOVA Live SpeedTracking (Old speed math, DirectML) - Q to quit", frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

finally:
    stop.set()
    cv2.destroyAllWindows()
