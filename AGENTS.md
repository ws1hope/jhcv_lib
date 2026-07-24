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

## Required post-change workflow

This workflow is mandatory whenever an agent actually changes project code in this repository. Do not mark a task complete until every applicable step below has succeeded.

### Scope and safety

- Record only changes that were actually completed. Do not add planned, speculative, or attempted-only work to the resolved summary.
- Preserve all existing history in `/Users/wangsen/Documents/GitHub/issue-tracking/jhcv_lib/issues.md` and `/Users/wangsen/Documents/GitHub/issue-tracking/jhcv_lib/resolved.md`; append new records and never replace or rewrite older entries.
- Do not stage unrelated user changes. Stage explicit paths only; never use `git add .` or `git add -A` without first proving every included change belongs to the current task.
- Before each commit, inspect `git status --short` and the relevant staged and unstaged diffs. Use `git diff` and `git diff --cached` to verify the exact contents.
- Never create an empty or meaningless commit. If no project code was changed, do not add a resolved entry merely to manufacture a commit.

### Required sequence

1. Finish the requested project-code change and run the relevant formatter, build, tests, or other verification available in the current environment.
2. In `/Users/wangsen/Documents/GitHub/jhcv_lib`, inspect the diff, stage only files belonging to the task, inspect the staged diff, and commit the project change first.
3. Capture the full project commit hash with `git rev-parse HEAD`. Never guess it or use a pre-commit hash.
4. Add one truthful resolved record to the two tracking files without altering historical records:
   - Append exactly one index row to the Markdown table immediately below `## 已解决汇总`. The table columns must remain `日期 | 问题 | 解决办法`.
   - Keep the index row concise. Its `解决办法` cell must be a clickable cross-file link in the form `[查看详情](resolved.md#resolved-YYYYMMDD-short-english-id)`. Never write plain text such as `详见下条`, and never put detailed records in `issues.md`.
   - Append the corresponding detailed record to the sibling file `/Users/wangsen/Documents/GitHub/issue-tracking/jhcv_lib/resolved.md`. Before it, add an explicit HTML anchor such as `<a id="resolved-YYYYMMDD-short-english-id"></a>`, followed by a stable, unique Markdown level-two heading such as `## YYYY-MM-DD · 问题简述`.
   - Anchor IDs may contain only lowercase ASCII letters, digits, and hyphens. Use the form `resolved-YYYYMMDD-short-english-id`, ensure it is unique within `resolved.md`, and make the link target in `issues.md` match it exactly.
   - The detailed record must contain `问题`, `修改文件`, `解决方案`, `验证结果`, and the corresponding full `jhcv_lib` project `commit hash`. List only the key project files and record actual verification results; do not invent successful checks.
   - Before saving, verify that the new row is inside the table, `issues.md` contains no expanded detailed record, the anchor is unique, and the cross-file link and anchor text are identical.
5. In `/Users/wangsen/Documents/GitHub/issue-tracking`, inspect the diff, stage only `jhcv_lib/issues.md` and `jhcv_lib/resolved.md`, inspect the staged diff, and commit the tracking update.
6. Confirm both commits exist and both repositories have no task-related uncommitted changes. Report both commit hashes in the final response.

The two commits are intentionally separate: use the project commit hash in the tracking entry, then commit that entry in the issue-tracking repository. If the issue-tracking file cannot be updated or its repository cannot be committed, state the blocker and do **not** claim that the workflow or task is complete.

### Commit messages

- Project repository: use a concise Conventional Commit-style subject such as `fix(scope): ...`, `feat(scope): ...`, `refactor(scope): ...`, `test(scope): ...`, or `docs(scope): ...`.
- Issue-tracking repository: use `docs(jhcv_lib): 记录<本次问题的简短说明>`.
- Subjects must describe the actual change; avoid generic messages such as `update`, `changes`, `AI update`, or `fix issue`.
