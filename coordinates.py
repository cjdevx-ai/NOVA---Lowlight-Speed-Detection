import cv2

VIDEO_PATH = "snippets/01.mp4"   # change to your video file
DISPLAY_SCALE = 0.8
# --------------------------------------

paused = False
mouse_x, mouse_y = -1, -1

orig_w, orig_h = None, None

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        # Map display coords → original coords
        mouse_x = int(x / DISPLAY_SCALE)
        mouse_y = int(y / DISPLAY_SCALE)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

    # Resize for display ONLY
    display_frame = cv2.resize(
        frame,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_LINEAR
    )

    # Draw indicator using display-space coords
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

    cv2.imshow("Video", display_frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):  # SPACE = pause
        paused = not paused

cap.release()
cv2.destroyAllWindows()
