# AGENTS.md

## Build

Configure (VS 2022 / MSVC v143, x64):

```
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -T v143
```

Build:

```
cmake --build build --config Release
cmake --build build --config Debug
```

Default task in `.vscode/tasks.json` runs configure then Release build.

## CMake options

- `ENABLE_CUDA` (default `ON`) — links CUDA 11.8 + TensorRT + ONNX GPU provider libs and defines `USE_CUDA`. Pass `-DENABLE_CUDA=OFF` for CPU-only builds.
- `BUILD_TEST` (default `ON`) — builds all test executables under `test/`.

## Dependencies (hardcoded paths)

All third-party libs live under `G:/3thirdparty/`:

| Dependency | Path |
|---|---|
| OpenCV 4.90 | `G:/3thirdparty/opencv/build` |
| ONNX Runtime | `G:/3thirdparty/onnxruntime` |
| TensorRT | `G:/3thirdparty/tensorrt` |
| Paddle Inference | `G:/3thirdparty/paddle_inference` |
| CUDA | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8` |

Runtime DLLs from these dirs must be on `PATH` to run any executable. See `.vscode/launch.json` for the required PATH entries.

## Architecture

```
include/JHDeepCore.h      ← public API header (JHDeepCore namespace)
src/
  ai_platform/             ← primary API layer (PIMPL pattern)
    JHDeepCore.cpp         ← pImpl wrappers for all task classes
    jhdeepcore_inference/  ← ONNX session management (base + factory)
    jhdeepcore_utils/      ← config_loader, device_utils
    jhdeepcore_classify/   ← ClassifierImpl
    jhdeepcore_detect/     ← DetectorImpl
    jhdeepcore_segment/    ← SegmenterImpl
    jhdeepcore_instance_segment/ ← InstanceSegmenterImpl
    jhdeepcore_ocr/        ← OCRRecognizerImpl
  cv_infer/                ← legacy/low-level inference modules
    yolo_inference.*       ← YOLO detection
    cls_inference.*        ← classification
    ocr_inference.*        ← OCR
    penma_rec_inference.*  ← plate/label recognition
    penma_rec_dll.*        ← shared DLL export (uses .def file)
  nisco_project/           ← business-specific service modules
    dabang_jiguang_service ← OCR service (label det → char det → OCR pipeline)
  third_party/             ← cpp-httplib, json.hpp
```

Two build targets:
- `jhcv_lib` — static library containing all source
- `penma_rec_dll` — shared library wrapping penma recognition, exports via `penma_rec_dll.def`

## Conventions

- C++17, MSVC runtime (`MultiThreaded` / `MultiThreadedDebug`).
- `/utf-8` compile flag is applied to `ai_platform` and `nisco_project` sources (they contain Chinese characters). New source files with non-ASCII literals must be added to the `set_source_files_properties` block in `src/CMakeLists.txt`.
- `cv_infer` sources do **not** get `/utf-8` — keep them ASCII.
- All task classes use PIMPL (pImpl idiom) to hide ONNX/Paddle details from the public header.
- yaml-cpp is linked statically (`YAML_CPP_STATIC_DEFINE`).

## Test executables

No test framework. Each subdirectory under `test/` has a standalone `main.cpp` that runs a specific inference task. They all link against `jhcv_lib` and require model files + images to run (not committed; see `.gitignore`).

## Known gotchas

- `build/`, `build_cpu/`, `build_projects/` are in `.gitignore` but may exist locally with stale VS project files — don't trust their contents.
- `images/` and `models/` are gitignored. Tests and debug configs reference files under these dirs; they must be supplied manually.
- CMakeLists.txt hardcodes `opencv_world490.lib` — if OpenCV version changes, update the lib name in both root `CMakeLists.txt:17` and anywhere else it's referenced.
