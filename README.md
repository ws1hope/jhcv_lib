# jhcv_lib

基于 C++17 + ONNX Runtime 的工业视觉推理库，面向钢铁/冶金场景的字符识别（炉号 OCR）、标签识别与目标跟踪。封装检测、分类、语义/实例分割、OCR、单/跨相机跟踪、单应矩阵等通用推理任务，并在其上构建了多个开箱即用的 HTTP 业务服务。

> 公共 API 全部位于 `JHDeepCore` 命名空间，头文件 `include/JHDeepCore.h`，采用 PIMPL 隐藏 ONNX Runtime / OpenCV 实现细节。

## 功能特性

- **通用推理任务**：`Classifier` / `Detector` / `Segmenter` / `InstanceSegmenter` / `OCRRecognizer`，统一 `process(images, results)` 接口。
- **业务服务**：6 个面向场景的 HTTP 服务（详见下表），每个都提供「服务器模式」和「本地单图测试模式」。
- **目标跟踪**：单相机 `Tracker`（卡尔曼 + 匈牙利/LAPJV 或 ByteTrack）与 `CrossCameraTracker`（跨相机关联，基于单应矩阵映射到公共平面）。
- **单应矩阵**：`Homography` 类，雷达/相机坐标平面映射。
- **跨平台构建**：Windows (MSVC + CUDA/TensorRT) / macOS / Linux，CMake 统一管理。

## 目录结构

```
jhcv_lib/
├── include/JHDeepCore.h        # 公共 API（任务类、服务类、跟踪、单应矩阵）
├── src/
│   ├── ai_platform/            # 主 API 层（PIMPL）：classify/detect/segment/instance_segment/ocr/inference/utils
│   ├── cv_infer/               # 底层推理：yolo / cls / ocr / penma_rec + penma_rec_dll 导出
│   ├── nisco_project/          # 业务服务模块
│   │   ├── dabang_jiguang/     # 大棒激光 OCR 服务
│   │   ├── tiebiao/            # 贴标炉号识别服务（也被 gx_jingzheng gangbiao 复用）
│   │   ├── dispatch/           # 调度服务（分类器路由到 dabang / tiebiao）
│   │   ├── zbhc/               # 钢坯号字符识别服务
│   │   ├── zbsltj/             # 钢坯数量统计 / 炉号匹配服务
│   │   └── gx_jingzheng/       # 钢芯精整服务（zifu 字符 / gangbiao 钢标 双分支）
│   ├── mtracker/               # 跟踪算法（卡尔曼、匈牙利、LAPJV、ByteTrack、跨相机）
│   ├── cv_homography/          # 单应矩阵实现
│   ├── utils/                  # file_utils / infer_utils / image_utils
│   └── third_party/            # cpp-httplib、json.hpp
├── test/                       # 每个子目录一个独立 main.cpp 可执行（链接 jhcv_lib）
├── deploy/                     # Docker 构建相关
├── models/                     # 模型文件（onnx/yaml，gitignore，需自行准备）
├── labels/                     # 标注/标签文件
├── CMakeLists.txt              # 顶层跨平台配置
└── AGENTS.md                   # 构建约定（Windows/MSVC 视角）
```

## 依赖

| 依赖 | 说明 |
|---|---|
| OpenCV ≥ 4.x | 图像处理（Windows 固定 `opencv_world490`） |
| ONNX Runtime | 推理后端（编译期定义 `USE_ONNXRUNTIME`） |
| yaml-cpp | 配置解析（静态链接，`YAML_CPP_STATIC_DEFINE`） |
| CUDA 12.8 + TensorRT（可选） | 仅 Windows GPU 构建，开启 `ENABLE_CUDA` |

**macOS / Linux**：依赖安装到 `/usr/local`（OpenCV、ONNX Runtime、yaml-cpp），可用 brew / apt 安装。

**Windows**：第三方库统一放在 `G:/3thirdparty/`（opencv / onnxruntime / tensorrt / paddle_inference），CUDA 在 `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8`。运行时需把上述 DLL 加入 `PATH`（见 `.vscode/launch.json`）。

## 构建

CMake ≥ 3.15，C++17。

**macOS / Linux（CPU）**

```bash
cmake -B build -S .
cmake --build build -j8        # 或 make -C build -j8
```

**Windows（MSVC v143 + 可选 CUDA）**

```bash
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -T v143
cmake --build build --config Release
```

**CMake 选项**

| 选项 | 默认 | 说明 |
|---|---|---|
| `ENABLE_CUDA` | Win=ON，mac/linux=OFF | 链接 CUDA/TensorRT，定义 `USE_CUDA`；关掉则纯 CPU |
| `BUILD_TEST` | ON | 构建 `test/` 下所有可执行 |

**构建产物**（位于 `build/`）

- `lib/libjhcv_lib.a` — 包含全部源码的静态库
- `lib/libpenma_rec_dll.{so,dylib}` / `penma_rec_dll.dll` — 喷码识别动态库（按 `penma_rec_dll.def` 导出）
- `bin/<service>` — 各业务服务可执行（见下表）

## 业务服务

| 服务 | 可执行 | API 类 | 说明 |
|---|---|---|---|
| 高线精整 | `gx_jingzheng` | `GxJingzhengService` | 高线精整场景，`zifu`(字符) / `gangbiao`(钢标) 双分支；gangbiao 复用 tiebiao 子 pipeline |
| 贴标识别 | `tiebiao` | `TiebiaoService` | 标签分割→字符分割→方向分类→OCR→炉号匹配 |
| 大棒激光 OCR | `dabang_jiguang` | `OCRService` | 大棒激光刻号 OCR（标签检测→字符检测→OCR） |
| 钢坯号识别 | `zbhc` | `ZbhcService` | 钢坯字符识别与方向矫正 |
| 钢坯数量统计 | `zbsltj` | `ZbsltjService` | 坯料数量统计、炉号匹配、PDI 支数、跟踪序号 |
| 调度 | `dispatch` | `DispatchService` | 用分类器把图像路由到 dabang / tiebiao 分支 |

每个服务统一提供：`handleRequest(json_body)`（HTTP 用）和 `runLocalTest(image, heat, station)`（本地测试用）。

## 配置

每个服务一份 YAML 配置（示例见 `test/<service>/config.yaml`）。以 `gx_jingzheng` 为例：

```yaml
server:
  service_name: "character_recognition"   # HTTP 路径前缀
  host: "127.0.0.1"
  port: 8080

models:
  dingwei_model: "models/gx_xm/piliao_dingwei.onnx"   # 定位检测
  dingwei_label: "models/gx_xm/piliao_dingwei.yaml"
  seg_model: "models/gx_xm/pm_yb_dw_instance.onnx"    # 实例分割
  seg_label: "models/gx_xm/pm_yb_dw_instance.yaml"
  direction_cls_model: "models/gx_xm/pm_fx_cls.onnx"  # 方向分类
  ocr_model: "models/gx_xm/penma_rec.onnx"            # OCR
  ocr_label: "models/gx_xm/penma_rec.yaml"
  tiebiao_config: "test/tiebiao/config.yaml"          # gangbiao 分支复用 tiebiao

classes:
  zifu_class_name: "zifu"
  gangbiao_class_name: "gangbiao"

inference:
  device: "gpu"        # gpu/cuda 走 GPU，其余 CPU

output:
  result_dir: "D:\\GxJingzhengResult"
  split_dir: "D:\\GxJingzhengSplit"
  log_dir: "D:\\GxJingzhengLog"
  char_crop_dir: "D:\\GxJingzhengCharCrop"
```

## 运行

**服务器模式**（默认，启动 HTTP 服务）

```bash
./build/bin/gx_jingzheng -c test/gx_jingzheng/config.yaml
```

**本地单图测试模式**

```bash
./build/bin/gx_jingzheng --test -c test/gx_jingzheng/config.yaml \
    -i image.jpg -H <炉号> -s <工位ID>
# -i 支持用 # 分隔多张图
```

通用参数：`-c/--config`、`--test`、`-i/--image`、`-H/--heat`、`-s/--station`、`-h/--help`。

**HTTP 接口**（路径前缀 = 配置中的 `service_name`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/{service_name}` | 推理，请求体为 JSON |
| `GET` | `/{service_name}/hello` | 健康检查 |
| `GET` | `/{service_name}/stop` | 停止服务 |

请求示例（`character_recognition` 服务）：

```json
{
  "station_id": 0,
  "heat_number": "GH123456",
  "picture_path": "/data/img.jpg"
}
```

> `picture_path` 为图片路径，多张图用 `#` 分隔（服务端按 C# 风格切分）。服务读取本地图片做推理，**不接收 base64**。

响应示例：

```json
{
  "station_id": 0,
  "items": [
    { "state_flag": "OK", "result": "90199 9E819" }
  ]
}
```

## 公共 API 速览

```cpp
#include "JHDeepCore.h"

using namespace JHDeepCore;

// 通用推理任务（PIMPL，构造即加载模型）
Classifier         cls("model.onnx", "label.yaml", device_id);
Detector          det("model.onnx", "label.yaml", device_id, "", 0.25f, 0.45f);
InstanceSegmenter  iseg("model.onnx");
OCRRecognizer      ocr("model.onnx", "label.yaml");

std::vector<cv::Mat> imgs = {img};
std::vector<OCRResult> res;  ocr.process(imgs, res);

// 业务服务（HTTP / 本地测试）
GxJingzhengService svc("config.yaml");
std::string resp = svc.handleRequest(json_body);
svc.runLocalTest("img.jpg", "GH123456", 0);

// 跟踪
Tracker tracker(TrackerConfig{});          // 默认 ByteTrack
std::vector<TrackedObject> trk;  tracker.update(dets, frame, trk);

// 单应矩阵
Homography H;
cv::Mat M = H.compute(pairs);             // ≥4 对点
cv::Point2f dst = H.project_point(src);
```

## Docker 部署

`deploy/` 下提供基于 `nvidia/cuda:11.8.0-devel-ubuntu22.04` 的镜像，配置为 SSH 开发容器：

- `Dockerfile` — 主镜像（清华源 + SSH，`root:123`）
- `Dockerfile.deps` — 依赖镜像
- `cmake.sh` / `docker_compose.sh` / `docker_run_deps.sh` — 构建与运行脚本

```bash
cd deploy && bash docker_compose.sh
```

## 开发约定

- **C++17**；MSVC 运行库统一 `MultiThreaded` / `MultiThreadedDebug`。
- **PIMPL**：所有任务类用 pImpl 隐藏 ONNX 细节，公共头不暴露第三方类型。
- **`/utf-8`**：含中文的源文件（`ai_platform`、`nisco_project`、`utils`、`cv_homography`）在 MSVC 下加 `/utf-8`；`cv_infer` 保持纯 ASCII。新增含非 ASCII 字面量的源文件需加入 `src/CMakeLists.txt` 的 `set_source_files_properties` 块。
- **yaml-cpp** 静态链接（`YAML_CPP_STATIC_DEFINE`）。
- 无单元测试框架；`test/<name>/main.cpp` 各自独立运行某项推理任务，依赖 `models/` 与 `images/`（均 gitignore，需自行准备）。

## 备注

- `models/`、`images/`、`docs/`、`result/`、`output/` 已在 `.gitignore`，不入库；测试与配置中引用的模型/图片需手动放置。
- OpenCV 版本若变更，需同步更新 `CMakeLists.txt` 中 `opencv_world490.lib` 的版本号。
