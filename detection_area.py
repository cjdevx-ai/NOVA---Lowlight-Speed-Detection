import os
import numpy as np
import cv2
from tqdm import tqdm

video_1 = "snippets/01.mp4"
video_2 = "snippets/02.mp4"
video_3 = "snippets/03.mp4"

INPUT_VIDEO = video_1
OUTPUT_VIDEO = "snippets/detection_area_output.mp4"

assert os.path.exists(INPUT_VIDEO), "Input video not found"

# Polygon points (clockwise)
POLYGON = np.array([
    [606, 600],   # left-top
    [1110, 504],  # right-top
    [1730, 800],  # right-bottom
    [1026, 1070]    # left-bottom
], dtype=np.int32)

cap = cv2.VideoCapture(INPUT_VIDEO)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

for _ in tqdm(range(frames), desc="Rendering polygon"):
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Draw polygon outline ----
    cv2.polylines(
        frame,
        [POLYGON],
        isClosed=True,
        color=(255, 0, 0),  # blue
        thickness=2
    )

    # ---- Optional: semi-transparent fill ----
    overlay = frame.copy()
    cv2.fillPoly(overlay, [POLYGON], (255, 0, 0))
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

    writer.write(frame)

cap.release()
writer.release()

print("Saved:", OUTPUT_VIDEO)
