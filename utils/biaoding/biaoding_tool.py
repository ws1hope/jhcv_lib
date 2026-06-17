#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


DEFAULT_IMAGE_DIR = "images/test_video/biaoding"
DEFAULT_MAP_WIDTH = 12500
DEFAULT_MAP_HEIGHT = 3000
DEFAULT_OUTPUT_JSON = "result/biaoding_calibration.json"
WINDOW_NAME = "biaoding_tool"
VISUAL_SCALE = 5
POINT_RADIUS = 7 * VISUAL_SCALE
PENDING_POINT_RADIUS = 10 * VISUAL_SCALE
CROSS_HALF_LENGTH = 8 * VISUAL_SCALE
MARK_LINE_THICKNESS = 2 * VISUAL_SCALE
LABEL_SCALE = 0.55 * VISUAL_SCALE
LABEL_THICKNESS = 2 * VISUAL_SCALE
LABEL_PADDING = 6 * VISUAL_SCALE
LABEL_OFFSET_X = 8 * VISUAL_SCALE
LABEL_OFFSET_Y = -8 * VISUAL_SCALE
PENDING_LABEL_OFFSET_X = 10 * VISUAL_SCALE
PENDING_LABEL_OFFSET_Y = 18 * VISUAL_SCALE
DELETE_POINT_RADIUS = 18 * VISUAL_SCALE


@dataclass
class CameraImage:
    camera_id: int
    path: Path
    image: np.ndarray
    width: int
    height: int
    x_offset: int
    color: tuple[int, int, int]


@dataclass
class CalibrationRecord:
    camera_id: int
    image_point: tuple[float, float]
    map_point: tuple[float, float]


@dataclass
class PendingImagePoint:
    camera_id: int
    image_point: tuple[float, float]


@dataclass
class AppState:
    cameras: list[CameraImage]
    map_width: int
    map_height: int
    output_path: Path
    records: list[CalibrationRecord] = field(default_factory=list)
    pending: Optional[PendingImagePoint] = None
    top_width: int = 0
    top_height: int = 0
    canvas_width: int = 0
    canvas_height: int = 0
    dirty: bool = True
    delete_mode: bool = False


def extract_camera_id(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return 10**9
    return int(matches[-1])


def color_for_index(index: int) -> tuple[int, int, int]:
    colors = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 180, 0),
        (0, 180, 255),
        (180, 0, 180),
        (255, 180, 0),
    ]
    return colors[index % len(colors)]


def load_camera_images(image_dir: Path) -> list[CameraImage]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    image_paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ]
    image_paths.sort(key=lambda path: (extract_camera_id(path), path.name))

    cameras: list[CameraImage] = []
    x_offset = 0
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skip unreadable image: {path}")
            continue

        camera_id = extract_camera_id(path)
        if camera_id == 10**9:
            camera_id = len(cameras) + 1
        height, width = image.shape[:2]
        cameras.append(
            CameraImage(
                camera_id=camera_id,
                path=path,
                image=image,
                width=width,
                height=height,
                x_offset=x_offset,
                color=color_for_index(len(cameras)),
            )
        )
        x_offset += width

    if not cameras:
        raise RuntimeError(f"No readable camera images found in: {image_dir}")
    return cameras


def find_camera_by_id(state: AppState, camera_id: int) -> Optional[CameraImage]:
    for camera in state.cameras:
        if camera.camera_id == camera_id:
            return camera
    return None


def find_camera_at(state: AppState, x: int, y: int) -> Optional[CameraImage]:
    for camera in state.cameras:
        if camera.x_offset <= x < camera.x_offset + camera.width and 0 <= y < camera.height:
            return camera
    return None


def camera_point_index(state: AppState, record_index: int) -> int:
    record = state.records[record_index]
    count = 0
    for index in range(record_index + 1):
        if state.records[index].camera_id == record.camera_id:
            count += 1
    return count


def find_record_near_image_point(state: AppState, x: int, y: int) -> Optional[int]:
    best_index = None
    best_distance_sq = DELETE_POINT_RADIUS * DELETE_POINT_RADIUS
    for record_index, record in enumerate(state.records):
        camera = find_camera_by_id(state, record.camera_id)
        if camera is None:
            continue
        image_x = camera.x_offset + record.image_point[0]
        image_y = record.image_point[1]
        distance_sq = (image_x - x) ** 2 + (image_y - y) ** 2
        if distance_sq <= best_distance_sq:
            best_index = record_index
            best_distance_sq = distance_sq
    return best_index


def draw_label(
    canvas: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICKNESS
    )
    x, y = pos
    x0 = max(0, x)
    y0 = max(0, y - text_height - LABEL_PADDING)
    x1 = min(canvas.shape[1], x + text_width + LABEL_PADDING)
    y1 = min(canvas.shape[0], y + LABEL_PADDING)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        text,
        (x + LABEL_PADDING // 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_SCALE,
        color,
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def render_canvas(state: AppState) -> np.ndarray:
    canvas = np.full(
        (state.canvas_height, state.canvas_width, 3),
        (30, 30, 30),
        dtype=np.uint8,
    )

    for camera in state.cameras:
        roi = canvas[
            0 : camera.height,
            camera.x_offset : camera.x_offset + camera.width,
        ]
        roi[:] = camera.image
        cv2.rectangle(
            canvas,
            (camera.x_offset, 0),
            (camera.x_offset + camera.width - 1, camera.height - 1),
            camera.color,
            2,
        )
        draw_label(canvas, f"C{camera.camera_id}", (camera.x_offset + 12, 30), camera.color)

    if state.delete_mode:
        draw_label(
            canvas,
            "delete mode: click a marked image point",
            (20, 62),
            (0, 0, 255),
        )

    map_y0 = state.top_height
    canvas[map_y0 : map_y0 + state.map_height, 0 : state.map_width] = (255, 255, 255)
    cv2.rectangle(
        canvas,
        (0, map_y0),
        (state.map_width - 1, map_y0 + state.map_height - 1),
        (80, 80, 80),
        2,
    )

    for record_index, record in enumerate(state.records):
        camera = find_camera_by_id(state, record.camera_id)
        if camera is None:
            continue
        label = f"C{record.camera_id}-{camera_point_index(state, record_index)}"
        image_point = (
            camera.x_offset + round(record.image_point[0]),
            round(record.image_point[1]),
        )
        map_point = (
            round(record.map_point[0]),
            state.top_height + round(record.map_point[1]),
        )
        cv2.circle(canvas, image_point, POINT_RADIUS, camera.color, -1, cv2.LINE_AA)
        cv2.circle(canvas, map_point, POINT_RADIUS, camera.color, -1, cv2.LINE_AA)
        cv2.line(
            canvas,
            (map_point[0] - CROSS_HALF_LENGTH, map_point[1]),
            (map_point[0] + CROSS_HALF_LENGTH, map_point[1]),
            camera.color,
            MARK_LINE_THICKNESS,
        )
        cv2.line(
            canvas,
            (map_point[0], map_point[1] - CROSS_HALF_LENGTH),
            (map_point[0], map_point[1] + CROSS_HALF_LENGTH),
            camera.color,
            MARK_LINE_THICKNESS,
        )
        draw_label(
            canvas,
            label,
            (image_point[0] + LABEL_OFFSET_X, image_point[1] + LABEL_OFFSET_Y),
            camera.color,
        )
        draw_label(
            canvas,
            label,
            (map_point[0] + LABEL_OFFSET_X, map_point[1] + LABEL_OFFSET_Y),
            camera.color,
        )

    if state.pending:
        camera = find_camera_by_id(state, state.pending.camera_id)
        if camera is not None:
            point = (
                camera.x_offset + round(state.pending.image_point[0]),
                round(state.pending.image_point[1]),
            )
            cv2.circle(
                canvas,
                point,
                PENDING_POINT_RADIUS,
                (0, 255, 255),
                MARK_LINE_THICKNESS,
                cv2.LINE_AA,
            )
            draw_label(
                canvas,
                "pending",
                (point[0] + PENDING_LABEL_OFFSET_X, point[1] + PENDING_LABEL_OFFSET_Y),
                (0, 160, 160),
            )

    return canvas


def compute_homography(records: list[CalibrationRecord]) -> Optional[list[float]]:
    if len(records) < 4:
        return None
    image_points = np.array([record.image_point for record in records], dtype=np.float32)
    map_points = np.array([record.map_point for record in records], dtype=np.float32)
    matrix, _ = cv2.findHomography(image_points, map_points)
    if matrix is None:
        return None
    return [float(value) for value in matrix.reshape(-1)]


def load_existing_calibration(output_path: Path) -> tuple[Optional[int], Optional[int], list[CalibrationRecord]]:
    if not output_path.exists():
        return None, None, []

    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Warning: failed to read existing calibration JSON: {output_path} | {error}")
        return None, None, []

    map_width = data.get("map_width")
    map_height = data.get("map_height")
    if not isinstance(map_width, int) or map_width <= 0:
        map_width = None
    if not isinstance(map_height, int) or map_height <= 0:
        map_height = None

    records: list[CalibrationRecord] = []
    for camera_data in data.get("cameras", []):
        camera_id = camera_data.get("camera_id")
        if not isinstance(camera_id, int):
            continue
        for point_data in camera_data.get("points", []):
            image_point = point_data.get("image_point")
            map_point = point_data.get("map_point")
            if not is_point_pair(image_point) or not is_point_pair(map_point):
                continue
            records.append(
                CalibrationRecord(
                    camera_id=camera_id,
                    image_point=(float(image_point[0]), float(image_point[1])),
                    map_point=(float(map_point[0]), float(map_point[1])),
                )
            )

    print(
        f"Loaded existing calibration JSON: {output_path} "
        f"map={map_width or 'default'}x{map_height or 'default'} "
        f"points={len(records)}"
    )
    return map_width, map_height, records


def is_point_pair(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    return all(isinstance(item, (int, float)) for item in value)


def make_json(state: AppState) -> dict:
    root = {
        "map_width": state.map_width,
        "map_height": state.map_height,
        "cameras": [],
    }

    for camera in state.cameras:
        camera_records = [
            record for record in state.records if record.camera_id == camera.camera_id
        ]
        camera_json = {
            "camera_id": camera.camera_id,
            "image_path": camera.path.as_posix(),
            "image_width": camera.width,
            "image_height": camera.height,
            "x_offset": camera.x_offset,
            "points": [
                {
                    "image_point": [record.image_point[0], record.image_point[1]],
                    "map_point": [record.map_point[0], record.map_point[1]],
                }
                for record in camera_records
            ],
        }
        homography = compute_homography(camera_records)
        if homography is not None:
            camera_json["homography"] = homography
        root["cameras"].append(camera_json)

    return root


def save_calibration(state: AppState) -> None:
    state.output_path.parent.mkdir(parents=True, exist_ok=True)
    state.output_path.write_text(
        json.dumps(make_json(state), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {state.output_path}")


def print_status(state: AppState) -> None:
    if state.delete_mode:
        print("Delete mode: click a marked point in the top camera strip, or press m to exit")
        return
    if state.pending:
        x, y = state.pending.image_point
        print(f"Click a map point for C{state.pending.camera_id} image point ({x}, {y})")
    else:
        print("Click an image point in the top camera strip")


def on_mouse(event: int, x: int, y: int, flags: int, userdata: AppState) -> None:
    del flags
    state = userdata
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state.delete_mode:
        record_index = find_record_near_image_point(state, x, y)
        if record_index is None:
            print("No marked image point near click")
            return
        record = state.records.pop(record_index)
        state.dirty = True
        print(
            f"Deleted C{record.camera_id} image={record.image_point} map={record.map_point}"
        )
        print_status(state)
        return

    if state.pending is None:
        camera = find_camera_at(state, x, y)
        if camera is None:
            print("Ignored click outside camera images")
            return
        state.pending = PendingImagePoint(
            camera_id=camera.camera_id,
            image_point=(float(x - camera.x_offset), float(y)),
        )
        state.dirty = True
        print_status(state)
        return

    in_map = (
        0 <= x < state.map_width
        and state.top_height <= y < state.top_height + state.map_height
    )
    if not in_map:
        print("Ignored click outside map. Click in the white map area.")
        return

    record = CalibrationRecord(
        camera_id=state.pending.camera_id,
        image_point=state.pending.image_point,
        map_point=(float(x), float(y - state.top_height)),
    )
    state.records.append(record)
    state.pending = None
    state.dirty = True
    print(
        f"Added C{record.camera_id} image={record.image_point} map={record.map_point}"
    )
    print_status(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect camera image to map-plane calibration point pairs."
    )
    parser.add_argument("image_dir", nargs="?", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("map_width", nargs="?", type=int, default=DEFAULT_MAP_WIDTH)
    parser.add_argument("map_height", nargs="?", type=int, default=DEFAULT_MAP_HEIGHT)
    parser.add_argument("output_json", nargs="?", default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_json)
    existing_map_width, existing_map_height, existing_records = load_existing_calibration(
        output_path
    )
    map_width = existing_map_width or args.map_width
    map_height = existing_map_height or args.map_height
    if map_width <= 0 or map_height <= 0:
        raise ValueError("map_width and map_height must be positive")

    cameras = load_camera_images(Path(args.image_dir))
    valid_camera_ids = {camera.camera_id for camera in cameras}
    records = [
        record for record in existing_records if record.camera_id in valid_camera_ids
    ]
    skipped_records = len(existing_records) - len(records)
    if skipped_records:
        print(f"Skipped {skipped_records} historical points for missing cameras")

    top_width = sum(camera.width for camera in cameras)
    top_height = max(camera.height for camera in cameras)

    state = AppState(
        cameras=cameras,
        map_width=map_width,
        map_height=map_height,
        output_path=output_path,
        records=records,
        top_width=top_width,
        top_height=top_height,
        canvas_width=max(top_width, map_width),
        canvas_height=top_height + map_height,
    )

    print("Loaded cameras:")
    for camera in state.cameras:
        print(
            f"  C{camera.camera_id} {camera.path} "
            f"{camera.width}x{camera.height} x_offset={camera.x_offset}"
        )
    print(f"Map: {state.map_width}x{state.map_height} | Output: {state.output_path}")
    print("Keys: s=save, u=undo, r=reset, m=delete mode, q/esc=quit")
    print_status(state)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(state.canvas_width, 1600), min(state.canvas_height, 1000))
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    while True:
        if state.dirty:
            cv2.imshow(WINDOW_NAME, render_canvas(state))
            state.dirty = False

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            save_calibration(state)
        elif key == ord("m"):
            state.delete_mode = not state.delete_mode
            if state.delete_mode and state.pending is not None:
                state.pending = None
                print("Cleared pending image point")
            state.dirty = True
            print_status(state)
        elif key == ord("u"):
            if state.pending is not None:
                state.pending = None
                print("Cleared pending image point")
            elif state.records:
                last = state.records.pop()
                print(f"Removed last record for C{last.camera_id}")
            else:
                print("Nothing to undo")
            state.dirty = True
            print_status(state)
        elif key == ord("r"):
            state.records.clear()
            state.pending = None
            state.dirty = True
            print("Reset all calibration points")
            print_status(state)

    cv2.destroyWindow(WINDOW_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
