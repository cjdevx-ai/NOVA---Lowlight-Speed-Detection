import os
import cv2
import math
import numpy as np
from tqdm import tqdm
import supervision as sv
from inference_sdk import InferenceHTTPClient
from collections import defaultdict, deque

# =========================
#  Speedometer (unchanged)
# =========================
class LocalScaleSpeedometer:
    # Computes speed using local pixel -> meter scale computed from homography-like mapper
    # Uses update_with_centroid(frame_idx, track_id, (cx, cy)) every frame per tracked object
    # Uses get_speed(track_id) to read smoothed speed (kph int)
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
            world_pts = self.mapper.map(img_pts)  # shape (3,2)
        except Exception:
            return (1e-6, 1e-6)

        w00 = world_pts[0]
        wx = world_pts[1]
        wy = world_pts[2]
        mpp_x = float(np.linalg.norm(wx - w00))
        mpp_y = float(np.linalg.norm(wy - w00))
        if mpp_x == 0: mpp_x = 1e-6
        if mpp_y == 0: mpp_y = 1e-6
        return (mpp_x, mpp_y)

    def update_with_centroid(self, frame_idx:int, track_id:int, centroid:tuple):
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

    def get_speed(self, track_id:int):
        tid = int(track_id)
        if not self.speed_hist[tid]:
            return 0
        arr = np.array(self.speed_hist[tid], dtype=float)
        return int(round(float(np.median(arr))))

    def reset(self, track_id:int):
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
        # image_pts: (N,2) -> world_pts: (N,2) in meters
        pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 2)
        return pts * self.mpp

# ---------------- CONFIG ----------------
API_URL = "https://serverless.roboflow.com"
API_KEY = "UMn9vdgXLxL0XGiK66mz"  # rotate later
MODEL_ID = "nova-v2-5000/3"

INPUT_VIDEO  = "snippets/4.mp4"
OUTPUT_VIDEO = "snippets/speed_output.mp4"

CONF_TH = 0.40

# Gates (pixel coords)
GATE_A = ((140, 230), (661, 132))
GATE_B = ((1207, 367), (370, 715))
DISTANCE_M = 8.0

# Optional ROI polygon (only show speed if centroid inside)
POLYGON = np.array([
    [140, 230],
    [661, 132],
    [1207, 367],
    [370, 715]
], dtype=np.int32)

USE_ROI = True
STALE_RESET_FRAMES = 60  # reset speed history if track disappears this long
# ---------------------------------------

assert os.path.exists(INPUT_VIDEO), f"Input video not found: {INPUT_VIDEO}"
assert API_KEY, "API_KEY is empty"

client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError("Failed to open video")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
os.makedirs(os.path.dirname(OUTPUT_VIDEO) or ".", exist_ok=True)
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# ByteTrack
tracker = sv.ByteTrack(frame_rate=int(round(fps)))

def rf_to_sv_detections(result, conf_th=0.4):
    xyxy, confs, class_ids = [], [], []
    for p in result.get("predictions", []):
        if float(p.get("confidence", 0.0)) < conf_th:
            continue

        x, y = float(p["x"]), float(p["y"])
        w, h = float(p["width"]), float(p["height"])
        x1, y1 = x - w / 2.0, y - h / 2.0
        x2, y2 = x + w / 2.0, y + h / 2.0

        xyxy.append([x1, y1, x2, y2])
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
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def inside_roi(pt, polygon):
    return cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0

# ---- Calibrate meters-per-pixel from gates ----
def midpoint(p, q):
    return np.array([(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0], dtype=np.float32)

A_mid = midpoint(*GATE_A)
B_mid = midpoint(*GATE_B)
gate_px_dist = float(np.linalg.norm(B_mid - A_mid))
if gate_px_dist < 1e-6:
    raise RuntimeError("Gate midpoints are identical; cannot calibrate scale.")

mpp = DISTANCE_M / gate_px_dist  # meters per pixel
mapper = SimpleScaleMapper(mpp)
speedometer = LocalScaleSpeedometer(mapper, fps=fps, window=5, max_kph=300)

print(f"[calibration] gate_px_dist={gate_px_dist:.2f}px, mpp={mpp:.6f} m/px")

# Track last seen frames to reset stale IDs (ByteTrack can reuse IDs)
last_seen = {}

pbar = tqdm(total=(frames if frames > 0 else None), desc="Detect+Track+Speed")
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = client.infer(frame, model_id=MODEL_ID)
    detections = rf_to_sv_detections(result, conf_th=CONF_TH)
    detections = tracker.update_with_detections(detections)

    # Visualize ROI + gates
    cv2.polylines(frame, [POLYGON], True, (255, 0, 0), 2)
    cv2.line(frame, GATE_A[0], GATE_A[1], (0, 255, 255), 3)
    cv2.line(frame, GATE_B[0], GATE_B[1], (0, 165, 255), 3)
    cv2.putText(frame, "Gate A", (GATE_A[0][0], max(GATE_A[0][1]-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Gate B", (GATE_B[0][0], max(GATE_B[0][1]-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    if detections.tracker_id is not None and len(detections) > 0:
        for i, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            xyxy = detections.xyxy[i]
            cx, cy = centroid_xy(xyxy)

            # Reset if ID was stale too long (ID reuse protection)
            if tid in last_seen and (frame_idx - last_seen[tid] > STALE_RESET_FRAMES):
                speedometer.reset(tid)
            last_seen[tid] = frame_idx

            # Optional ROI filter (use centroid; less fragile than bottom-center)
            if USE_ROI and not inside_roi((cx, cy), POLYGON):
                # Still draw bbox + id, but skip speed update
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"VHCL {tid}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                continue

            # Update speedometer with centroid every frame
            speedometer.update_with_centroid(frame_idx, tid, (cx, cy))
            spd = speedometer.get_speed(tid)

            # Draw bbox + formatted label
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"VHCL {tid}, SPD: {spd} km/h"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    writer.write(frame)
    frame_idx += 1
    pbar.update(1)

pbar.close()
cap.release()
writer.release()
print("Saved:", OUTPUT_VIDEO)
