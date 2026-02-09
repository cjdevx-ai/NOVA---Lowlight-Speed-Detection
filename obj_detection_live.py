import cv2
import time
import numpy as np
import onnxruntime as ort

ONNX_PATH = r"G:\cla_projects\NOVA\best_320.onnx"
CAMERA_INDEX = 1

IMG_SIZE = 320
CONF_TH = 0.55
IOU_TH  = 0.50

# --- DirectML session (RX580) ---
sess = ort.InferenceSession(
    ONNX_PATH,
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)
inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name
print("Available providers:", ort.get_available_providers())
print("Session providers:", sess.get_providers())

if "DmlExecutionProvider" not in sess.get_providers():
    raise RuntimeError("DirectML provider not active. This run is CPU.")

def letterbox(im, new_shape=640, color=(114, 114, 114)):
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
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
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

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

# reduce capture lag
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

fps_ctr = 0
fps = 0.0
t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    img, r, pad_x, pad_y = letterbox(frame, IMG_SIZE)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]

    t_inf = time.perf_counter()
    y = sess.run([out], {inp: x})[0]
    infer_ms = (time.perf_counter() - t_inf) * 1000

    y = np.squeeze(y)
    if y.shape[0] < y.shape[1]:
        y = y.T  # (N,C)

    # YOLO-style: [x,y,w,h, scores...]
    xywh = y[:, 0:4]
    scores = y[:, 4:]
    conf = scores.max(axis=1)
    cls  = scores.argmax(axis=1)

    m = conf >= CONF_TH
    if np.any(m):
        xywh_m = xywh[m]
        conf_m = conf[m]
        cls_m  = cls[m]

        cx, cy, w, h = xywh_m[:, 0], xywh_m[:, 1], xywh_m[:, 2], xywh_m[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # undo letterbox to original frame coords
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= r

        # clip
        H, W = frame.shape[:2]
        boxes[:, 0] = np.clip(boxes[:, 0], 0, W - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, W - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, H - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, H - 1)

        K = 200  # try 100/200
        if conf_m.size > K:
            idx = np.argsort(conf_m)[::-1][:K]
            boxes = boxes[idx]
            conf_m = conf_m[idx]
            cls_m = cls_m[idx]

        keep = nms(boxes, conf_m, IOU_TH)

        for i in keep:
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"vhcl:{int(cls_m[i])} {conf_m[i]:.2f}",
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    fps_ctr += 1
    if time.time() - t0 >= 1.0:
        fps = fps_ctr / (time.time() - t0)
        fps_ctr = 0
        t0 = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f} | infer: {infer_ms:.1f}ms",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("NOVA YOLO11 (RX580 DirectML) - Q to quit", frame)
    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
