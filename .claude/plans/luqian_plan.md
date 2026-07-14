# Plan: 创建 luqian 字符识别项目

## 目标
在 `src/nisco_project/luqian/` 下新建一个字符识别项目，流程为：
**目标检测(det) → 裁剪 → 实例分割(seg) → 识别(参考 zbhc)**

即 zbhc 去掉第二级检测器(det2)：zbhc 是 `det1 → det2 → seg → 识别`，luqian 是 `det → seg → 识别`。
每个 det 检出的框 = 一个"目标"(对应 zbhc 的坯料 billet)，裁剪后直接送 seg。

## 核心假设(请确认)
1. **单级目标检测**：只有一个 `det_model`，对应 zbhc 的 det1+det2 合并为一级。每个检出框裁剪后直接送实例分割。
2. **识别逻辑忠实参考 zbhc**：保留 zbhc 的逐字符识别机制(mask→轮廓→最小外接矩算倾角→加边旋转水平化→紧裁回→方向判定→OCR→片段排序拼接)。**包括 zbhc 可选的 heat_number(炉号)路径**：传入 ≥8 位 heat_number 时用炉号匹配定朝向+覆盖纠错+重排序+5/6位排序与末三 位前插'#'；不传 heat_number 时走低置信度翻转的通用路径。该路径全部由 `use_heat` 门控，不传则自动退化为通用识别。
   - 若 luqian 不涉及炉号、不需要 '#' 拼接与炉号匹配，请告知，我会去掉这部分只保留通用逐字符 OCR。
3. **端口** 8085(gx_jingzheng 用 8084，顺延)。
4. 模型：det_model / seg_model / ocr_model / ocr_label / direction_cls_model。

## 文件清单

### 新建
- `src/nisco_project/luqian/luqian_pipeline.h`
  结构体 `LuqianCharInfo`、`LuqianTargetResult`(对应 BilletResult)、`LuqianPipelineResult`；类 `LuqianPipeline`，持有 `det_ / seg_ / ocr_ / direction_cls_`(无 det2_)。
- `src/nisco_project/luqian/luqian_pipeline.cpp`
  `process()`：det → 遍历检出框裁剪 → seg → zbhc 式逐字符识别与拼装 → 可视化 `createAnnotatedImage`；`warmup()`。
- `src/nisco_project/luqian/luqian_service.cpp`
  `LuqianServicePrivate`(handleRequest + runLocalTest + saveCharCrops) 与 `LuqianService` 外壳，照搬 zbhc_service 结构。
- `test/luqian/main.cpp` —— 入口(HTTP 服务 + --test 本地测试)，照搬 zbhc main。
- `test/luqian/config.yaml` —— det_model / seg_model / ocr_model / ocr_label / direction_cls_model / device:cpu。

### 修改
- `src/utils/file_utils.h` —— 新增 `struct LuqianServerConfig`(service_name="luqian", port=8085, det_model/seg_model/ocr_model/ocr_label/direction_cls_model/device)。
- `src/utils/file_utils.cpp` —— 新增 `FileHelper::loadLuqianConfig()`。
- `include/JHDeepCore.h` —— 前置声明 `LuqianServicePrivate` + `class LuqianService`(config/handleRequest/runLocalTest)。
- `src/CMakeLists.txt` —— `NISCO_PROJECT_SOURCES` 增加 luqian 三个 cpp；`target_include_directories` 增加 luqian 目录。
- `test/CMakeLists.txt` —— 增加 `add_executable(luqian ...)` 与 MSVC /utf-8 设置，加入 TEST_TARGETS。

## 验证
1. `cmake -B build && cmake --build build --target luqian` 编译通过(本地 macOS CPU 模式)。
2. `./build/bin/luqian --test -c test/luqian/config.yaml -i <图> -H <炉号>` 跑通本地测试，输出 JSON 与 `luqian_result_*.jpg`。
3. 结果字段与 zbhc 对齐：read_picture_flag / rec_state_flag / rec_results / picture_path。
