import os
import cv2
import numpy as np
from tqdm import tqdm
import supervision as sv
from inference_sdk import InferenceHTTPClient

# ---------------- CONFIG ----------------
API_URL = "https://serverless.roboflow.com"
API_KEY = "UMn9vdgXLxL0XGiK66mz"  # set this in Colab env
MODEL_ID = "nova-v2-5000/3"

INPUT_VIDEO  = "snippets/4.mp4"
OUTPUT_VIDEO = "snippets/tracking_output.mp4"

CONF_TH = 0.40

# Gates (pixel coords)
GATE_A = ((140, 230), (661, 132))        # start
GATE_B = ((1207, 367), (370, 715))       # end
DISTANCE_M = 8.0

# Optional ROI polygon (only compute speed for vehicles whose bottom-center is inside)
POLYGON = np.array([
    [140, 230],
    [661, 132],
    [1207, 367],
    [370, 715]
], dtype=np.int32)

DEBOUNCE_FRAMES = 5          # avoid double-trigger
MIN_MOVE_PX = 2.0            # ignore jitter
DIR_MIN_DOT = 0.20           # direction filter strictness
# ---------------------------------------

assert os.path.exists(INPUT_VIDEO), f"Input video not found: {INPUT_VIDEO}"
assert API_KEY, "ROBOFLOW_API_KEY env var not set"

# Roboflow client
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

# Supervision ByteTrack
tracker = sv.ByteTrack(frame_rate=int(round(fps)))

def midpoint(p, q):
    return np.array([(p[0]+q[0])/2.0, (p[1]+q[1])/2.0], dtype=np.float32)

A_mid = midpoint(*GATE_A)
B_mid = midpoint(*GATE_B)
travel_vec = B_mid - A_mid
travel_unit = travel_vec / (np.linalg.norm(travel_vec) + 1e-9)

def rf_to_sv_detections(result, conf_th=0.4):
    """
    Convert Roboflow predictions (center x,y,w,h) -> Supervision Detections (xyxy).
    """
    xyxy, confs, class_ids = [], [], []
    for p in result.get("predictions", []):
        if p.get("confidence", 0.0) < conf_th:
            continue

        x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
        x1, y1 = x - w/2.0, y - h/2.0
        x2, y2 = x + w/2.0, y + h/2.0

        xyxy.append([x1, y1, x2, y2])
        confs.append(float(p["confidence"]))
        class_ids.append(0)  # or map classes if you need multi-class logic

    if not xyxy:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )

def bottom_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (0.5*(x1+x2), y2)

def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def segments_intersect(p1, p2, q1, q2):
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    # colinear fallback
    if (o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0):
        def in_box(a, b, c):
            return (min(a[0],b[0]) <= c[0] <= max(a[0],b[0]) and
                    min(a[1],b[1]) <= c[1] <= max(a[1],b[1]))
        return in_box(p1,p2,q1) or in_box(p1,p2,q2) or in_box(q1,q2,p1) or in_box(q1,q2,p2)

    return (o1 * o2 <= 0) and (o3 * o4 <= 0)

def moving_correct_direction(prev_pt, curr_pt, travel_unit, min_dot=0.2, min_move=2.0):
    v = np.array(curr_pt, dtype=np.float32) - np.array(prev_pt, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm < min_move:
        return False
    v_unit = v / (norm + 1e-9)
    return float(np.dot(v_unit, travel_unit)) >= min_dot

def inside_roi(pt, polygon):
    # pt is (x,y) float; OpenCV expects (x,y) and returns +1/0/-1
    return cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0

# Per-track state
track_state = {}
# {tid: {"last_pt":(x,y), "tA":int|None, "tB":int|None, "speed_kmh":float|None}}

frame_idx = 0
pbar_total = frames if frames > 0 else None
pbar = tqdm(total=pbar_total, desc="Detect+Track+Speed")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Roboflow inference on this frame (np.ndarray supported by SDK) ---
    result = client.infer(frame, model_id=MODEL_ID)

    detections = rf_to_sv_detections(result, conf_th=CONF_TH)
    detections = tracker.update_with_detections(detections)  # adds tracker_id

    # --- Draw ROI + Gates (visual sanity) ---
    cv2.polylines(frame, [POLYGON], True, (255, 0, 0), 2)
    cv2.line(frame, GATE_A[0], GATE_A[1], (0, 255, 255), 3)  # Gate A (yellow)
    cv2.line(frame, GATE_B[0], GATE_B[1], (0, 165, 255), 3)  # Gate B (orange)
    cv2.putText(frame, "Gate A", (GATE_A[0][0], max(GATE_A[0][1]-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, "Gate B", (GATE_B[0][0], max(GATE_B[0][1]-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

    if detections.tracker_id is not None and len(detections) > 0:
        for xyxy, tid in zip(detections.xyxy, detections.tracker_id):
            tid = int(tid)
            pt = bottom_center(xyxy)  # (x,y)

            # Optional: only consider tracks inside your ROI
            if not inside_roi(pt, POLYGON):
                # still draw bbox+id, but skip speed timing
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(y1-8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                continue

            st = track_state.get(tid, {"last_pt": None, "tA": None, "tB": None, "speed_kmh": None})

            if st["last_pt"] is not None:
                p_prev = st["last_pt"]
                p_curr = pt

                # Direction filter (your vehicles move ↘ : A -> B)
                if moving_correct_direction(p_prev, p_curr, travel_unit, min_dot=DIR_MIN_DOT, min_move=MIN_MOVE_PX):
                    # Gate A crossing
                    if st["tA"] is None:
                        if segments_intersect(p_prev, p_curr, GATE_A[0], GATE_A[1]):
                            st["tA"] = frame_idx

                    # Gate B crossing (only after A)
                    elif st["tB"] is None:
                        if segments_intersect(p_prev, p_curr, GATE_B[0], GATE_B[1]):
                            if frame_idx - st["tA"] > DEBOUNCE_FRAMES:
                                st["tB"] = frame_idx
                                dt = (st["tB"] - st["tA"]) / fps
                                if dt > 1e-6:
                                    st["speed_kmh"] = (DISTANCE_M / dt) * 3.6

            st["last_pt"] = pt
            track_state[tid] = st

            # Draw bbox + speed text
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            label = f"VHCL {tid}"
            if st["speed_kmh"] is not None:
                label += f" | {st['speed_kmh']:.1f} km/h"

            # Place label above bbox
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    writer.write(frame)
    frame_idx += 1
    pbar.update(1)

pbar.close()
cap.release()
writer.release()
print("Saved:", OUTPUT_VIDEO)
