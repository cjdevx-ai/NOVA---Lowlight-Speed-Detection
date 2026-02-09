import os
import numpy as np
import cv2
from tqdm import tqdm

video_4 = "snippets/4.mp4"
INPUT_VIDEO = video_4
OUTPUT_VIDEO = "snippets/detection_area_output.mp4"

assert os.path.exists(INPUT_VIDEO), "Input video not found"

# --- Gate definitions ---
GATE_A = ((140, 230), (661, 132))        # 8m start line (top)
GATE_B = ((1207, 367), (370, 715))       # 8m end line (bottom)
DISTANCE_M = 8.0

# Optional: ROI polygon (for visualization only)
#Video source
"""
POLYGON = np.array([
    [140, 230],    # left-top
    [661, 132],    # right-top
    [1207, 367],   # right-bottom
    [370, 715]     # left-bottom
], dtype=np.int32)
"""
#OBS Source
POLYGON = np.array([
    [72, 156],    # left-top
    [333, 80],    # right-top
    [608, 238],   # right-bottom
    [187, 478]     # left-bottom
], dtype=np.int32)

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError("Failed to open video")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

def midpt(p, q):
    return (int((p[0]+q[0])/2), int((p[1]+q[1])/2))

for _ in tqdm(range(frames), desc="Rendering ROI + gates"):
    ret, frame = cap.read()
    if not ret:
        break

    # ROI outline + faint fill (optional)
    cv2.polylines(frame, [POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [POLYGON], (255, 0, 0))
    frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)

    # Draw Gate A and Gate B
    cv2.line(frame, GATE_A[0], GATE_A[1], (0, 255, 255), 3)  # yellow
    cv2.line(frame, GATE_B[0], GATE_B[1], (0, 165, 255), 3)  # orange

    # Labels
    a_mid = midpt(*GATE_A)
    b_mid = midpt(*GATE_B)

    cv2.putText(frame, "Gate A (Start)", (a_mid[0] + 10, a_mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, "Gate B (End)", (b_mid[0] + 10, b_mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.putText(frame, f"Distance A->B = {DISTANCE_M:.1f} m", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    writer.write(frame)

cap.release()
writer.release()

print("Saved:", OUTPUT_VIDEO)
