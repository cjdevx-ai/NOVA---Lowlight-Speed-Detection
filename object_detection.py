import argparse
import cv2
from cv2.dnn import Model
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict, deque

"""
# LONG DIST
SOURCE = np.array([
    [38, 56],
    [345, 9],
    [1207, 367],
    [370, 715]
], dtype=np.int32)

"""
"""
#SHORT DIST
SOURCE = np.array([
    [140, 230],
    [661, 132],
    [1207, 367],
    [370, 715]
], dtype=np.int32)
"""


#MIDDLE DIST
SOURCE = np.array([
    [128, 151],
    [461, 90],
    [1137, 397],
    [370, 715]
], dtype=np.int32)



TARGET_WIDTH = 4
TARGET_HEIGHT = 20

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Inference and Supervision"
    )

    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )

    return parser.parse_args()

def in_zone_mask(dets: sv.Detections, polygon: np.ndarray) -> np.ndarray:
    if len(dets) == 0:
        return np.array([], dtype=bool)
    xyxy = dets.xyxy
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0  # use center; or use y2 for bottom
    pts = np.stack([cx, cy], axis=1).astype(np.float32)
    return np.array([cv2.pointPolygonTest(polygon, tuple(p), False) >= 0 for p in pts], dtype=bool)



if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO(r"G:\cla_projects\NOVA\onnx\yolo11l_1k_320\best.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh=video_info.resolution_wh
    )

    bounding_box_annotator = sv.BoxAnnotator(
        thickness=thickness,
        color_lookup=sv.ColorLookup.TRACK
    )

    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )

    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )
    

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE, video_info.resolution_wh)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        mask = in_zone_mask(detections, SOURCE)
        detections = detections[mask]

        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
        points = view_transformer.transform_points(points=points).astype(int)

        for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

        # labels for speed
        labels = []
        for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

        # labels for coordinates
        """labels = [
            f"x:{x}, y:{y}"
            for [x,y]
            in points
        ]"""

         # labels for tracker id
        """labels = [
            f"#{tracker_id}" 
            for tracker_id
            in detections.tracker_id
        ]"""

        annotated_frame = frame.copy()
        sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)
        annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections, labels=labels
        )

        speed = 5  # 1.5 = normal, 0.5 = half-speed (slower), 2.0 = double-speed (faster)
        delay_ms = max(1, int((1000 / video_info.fps) / speed))

        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
