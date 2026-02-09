import os, cv2, time, math, threading
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
from collections import defaultdict, deque

# =========================
# Speedometer (same)
# =========================
class LocalScaleSpeedometer:
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
        ds_m = math.hypot(dx_px * mpp_x, dy_px * mpp_y)
        kph = (ds_m * self.fps) * self.unit
        if kph < 0: kph = 0.0
        if kph > self.max_kph:
            kph = float(self.speed_hist[tid][-1]) if self.speed_hist[tid] else float(self.max_kph)
        self.speed_hist[tid].append(kph)

    def get_speed(self, track_id: int):
        tid = int(track_id)
        if not self.speed_hist[tid]:
            return 0
        return int(round(float(np.median(np.array(self.speed_hist[tid], dtype=float)))))

    def reset(self, track_id: int):
        tid = int(track_id)
        if tid in self.pos_hist: self.pos_hist[tid].clear()
        if tid in self.speed_hist: self.speed_hist[tid].clear()

class SimpleScaleMapper:
    def __init__(self, meters_per_pixel: float):
        self.mpp = float(meters_per_pixel)

    def map(self, image_pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 2)
        return pts * self.mpp

# ---------------- CONFIG ----------------
API_URL = "https://serverless.roboflow.com"
API_KEY = "UMn9vdgXLxL0XGiK66mz"  # rotate
MODEL_ID = "nova-v2-5000/3"

CAMERA_INDEX = 1
CONF_TH = 0.40

# Performance knobs
MAX_SIDE = 960          # try 640 / 800 / 960
INFER_EVERY = 3         # infer every N frames (2-5 typical)
STALE_RESET_FRAMES = 60

# Gates/ROI (must match your live camera framing)
GATE_A = ((72, 156), (333, 80))
GATE_B = ((608, 238), (187, 478))
DISTANCE_M = 8.0

#OBS Source
POLYGON = np.array([
    [72, 156],    # left-top
    [333, 80],    # right-top
    [608, 238],   # right-bottom
    [187, 478]     # left-bottom
], dtype=np.int32)

USE_ROI = True
# ---------------------------------------

assert API_KEY, "API_KEY empty"
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

# Latest-frame buffer to avoid lag
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

    # Reduce internal buffering if supported
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    # Optional: force smaller camera res to reduce CPU cost too
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while not stop.is_set():
        ok, frame = cap.read()
        if ok:
            latest.set(frame)
        else:
            time.sleep(0.01)
    cap.release()

# ByteTrack (use a sane frame_rate; speedometer uses fps param below)
tracker = sv.ByteTrack(frame_rate=30)

def rf_to_sv_detections(result, conf_th=0.4):
    xyxy, confs, class_ids = [], [], []
    for p in result.get("predictions", []):
        if float(p.get("confidence", 0.0)) < conf_th:
            continue
        x, y = float(p["x"]), float(p["y"])
        w, h = float(p["width"]), float(p["height"])
        xyxy.append([x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0])
        confs.append(float(p["confidence"]))
        class_ids.append(0)
    if not xyxy:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )

def centroid_xy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (0.5*(x1+x2), 0.5*(y1+y2))

def inside_roi(pt, polygon):
    return cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0

def midpoint(p, q):
    return np.array([(p[0]+q[0])/2.0, (p[1]+q[1])/2.0], dtype=np.float32)

# Calibrate scale once (still pixel-based; if your frame size changes, update these coords!)
A_mid = midpoint(*GATE_A)
B_mid = midpoint(*GATE_B)
gate_px_dist = float(np.linalg.norm(B_mid - A_mid))
if gate_px_dist < 1e-6:
    raise RuntimeError("Gate midpoints identical; cannot calibrate.")
mpp = DISTANCE_M / gate_px_dist
mapper = SimpleScaleMapper(mpp)

# IMPORTANT: do NOT trust CAP_PROP_FPS for webcams; track a runtime FPS estimate
speedometer = LocalScaleSpeedometer(mapper, fps=30.0, window=5, max_kph=300)

print(f"[calibration] gate_px_dist={gate_px_dist:.2f}px, mpp={mpp:.6f} m/px")

# Start capture thread
threading.Thread(target=capture_loop, daemon=True).start()

# Runtime FPS estimate (used to update speedometer.fps so speed doesn't break when FPS changes)
fps_est = 30.0
t_last_fps = time.time()
fps_counter = 0

last_seen = {}
frame_idx = 0

# Store last detections to reuse between inference steps
last_det = sv.Detections.empty()
last_scale = 1.0

try:
    while True:
        frame, ts = latest.get()
        if frame is None:
            time.sleep(0.005)
            continue

        # FPS estimate
        fps_counter += 1
        tnow = time.time()
        if tnow - t_last_fps >= 1.0:
            fps_est = fps_counter / (tnow - t_last_fps)
            fps_counter = 0
            t_last_fps = tnow
            speedometer.fps = max(1.0, float(fps_est))

        # Inference every N frames on resized frame
        do_infer = (frame_idx % INFER_EVERY == 0)

        if do_infer:
            h, w = frame.shape[:2]
            scale = min(MAX_SIDE / max(h, w), 1.0)
            if scale != 1.0:
                small = cv2.resize(frame, (int(w*scale), int(h*scale)))
            else:
                small = frame

            result = client.infer(small, model_id=MODEL_ID)
            det = rf_to_sv_detections(result, conf_th=CONF_TH)

            # Scale boxes back up to full-res coordinates for tracker + drawing
            if len(det) > 0 and scale != 1.0:
                det.xyxy = det.xyxy / scale

            last_det = det
            last_scale = scale

        # Tracker runs every frame using last detections (cheat mode, but works for live)
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

                if tid in last_seen and (frame_idx - last_seen[tid] > STALE_RESET_FRAMES):
                    speedometer.reset(tid)
                last_seen[tid] = frame_idx

                if USE_ROI and not inside_roi((cx, cy), POLYGON):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
                    cv2.putText(frame, f"VHCL {tid}", (x1, max(y1-10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                    continue

                speedometer.update_with_centroid(frame_idx, tid, (cx, cy))
                spd = speedometer.get_speed(tid)

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
                cv2.putText(frame, f"VHCL {tid}, SPD: {spd} km/h", (x1, max(y1-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

        cv2.putText(frame, f"FPS: {fps_est:.1f} | infer every {INFER_EVERY}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow("NOVA Live (Q to quit)", frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

finally:
    stop.set()
    cv2.destroyAllWindows()
