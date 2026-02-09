import cv2
import os
import time

SOURCE = 1  # image path OR video path OR 0 for webcam
DISPLAY_SCALE = 0.8

paused = False
mouse_x, mouse_y = -1, -1

# --- console print throttle ---
_last_print = 0.0
_last_xy = None
PRINT_HZ = 30  # max prints per second

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = int(x / DISPLAY_SCALE)
        mouse_y = int(y / DISPLAY_SCALE)

def show_frame(frame):
    """Render one frame with scaled display + mouse coordinate overlay."""
    global _last_print, _last_xy

    display_frame = cv2.resize(
        frame,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_LINEAR
    )

    if mouse_x >= 0 and mouse_y >= 0:
        disp_x = int(mouse_x * DISPLAY_SCALE)
        disp_y = int(mouse_y * DISPLAY_SCALE)

        cv2.circle(display_frame, (disp_x, disp_y), 4, (0, 255, 0), -1)
        cv2.putText(
            display_frame,
            f"({mouse_x}, {mouse_y})",
            (disp_x + 8, disp_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ---- continuous console print (throttled) ----
        now = time.time()
        xy = (mouse_x, mouse_y)
        if (xy != _last_xy) and (now - _last_print >= 1.0 / PRINT_HZ):
            print(f"x={mouse_x}, y={mouse_y}", flush=True)
            _last_print = now
            _last_xy = xy

    cv2.imshow("Viewer", display_frame)

# Window + mouse hook
cv2.namedWindow("Viewer")
cv2.setMouseCallback("Viewer", mouse_callback)

# ---- Case 1: webcam index ----
if isinstance(SOURCE, int):
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    frame = None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

        show_frame(frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()

# ---- Case 2: file path (image or video) ----
else:
    if not os.path.exists(SOURCE):
        raise RuntimeError(f"File not found: {SOURCE}")

    ext = os.path.splitext(SOURCE)[1].lower()
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    # ---- Image ----
    if ext in image_exts:
        frame = cv2.imread(SOURCE)
        if frame is None:
            raise RuntimeError("Could not read image (bad path or unsupported format)")

        while True:
            show_frame(frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break

    # ---- Video ----
    else:
        cap = cv2.VideoCapture(SOURCE)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        frame = None
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

            show_frame(frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

        cap.release()

cv2.destroyAllWindows()
