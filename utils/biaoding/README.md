# biaoding calibration tool

`biaoding_tool.py` is a Python + OpenCV utility for collecting calibration pairs
between stitched camera images and a white map plane.

## Requirements

```bash
python3 -m pip install opencv-python numpy
```

If your Python environment already has `cv2` and `numpy`, no extra install is
needed.

## Run

From the project root:

```bash
python3 utils/biaoding/biaoding_tool.py [image_dir] [map_width] [map_height] [output_json]
```

Defaults:

```text
image_dir   = images/test_video/biaoding
map_width   = 12500
map_height  = 3000
output_json = result/biaoding_calibration.json
```

Example:

```bash
python3 utils/biaoding/biaoding_tool.py images/test_video/biaoding 12500 3000 result/biaoding_calibration.json
```

The tool scans the image directory, sorts files by the number in the filename,
and keeps that number as `camera_id`. For example, `camera6.png` is saved as
camera `6`.

If `output_json` already exists, the tool reads it before opening the window:

- `map_width` and `map_height` from the JSON override the command-line width and
  height.
- Historical calibration points under `cameras[].points` are restored and drawn
  on the canvas.
- Points for camera IDs that are not present in the current image directory are
  skipped.

## Interaction

Click points in pairs:

1. Click an image point in the top stitched camera strip.
2. Click the matching map point in the bottom white map area.

Keys:

```text
s       save JSON
u       undo pending image point or last completed point pair
r       reset all points
m       enter or exit delete mode; then click a marked image point to delete its pair
q / Esc quit
```

Each camera needs at least four point pairs before a homography matrix is saved
for that camera.

Point markers and labels are drawn large for high-resolution images. Delete
mode also uses a larger click radius around each marked image point.

## Output

The JSON contains map size, camera image metadata, collected points, and a
row-major 3x3 homography when enough points exist:

```json
{
  "map_width": 12500,
  "map_height": 3000,
  "cameras": [
    {
      "camera_id": 1,
      "image_path": "images/test_video/biaoding/camera1.png",
      "image_width": 2560,
      "image_height": 1440,
      "x_offset": 0,
      "points": [
        {
          "image_point": [1692.0, 153.0],
          "map_point": [750.0, 836.0]
        }
      ],
      "homography": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }
  ]
}
```
