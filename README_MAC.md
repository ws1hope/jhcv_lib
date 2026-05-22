# jhcv_lib - Mac 环境搭建指南

## 项目简介

jhcv_lib 是一个深度学习推理库，支持多种计算机视觉任务：
- 🎯 目标检测 (YOLO)
- 🏷️ 图像分类 (ResNet)
- 🔤 OCR 文字识别
- 🎨 语义分割
- 🖼️ 实例分割

## 系统要求

- **操作系统**: macOS 10.15+ 
- **编译器**: Xcode Command Line Tools
- **CMake**: 3.15+
- **内存**: 建议 8GB+

## 快速开始

### 1. 安装 Homebrew（如果未安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. 安装依赖库

```bash
# 安装核心依赖
brew install cmake
brew install opencv
brew install onnxruntime
brew install yaml-cpp

# 验证安装
cmake --version    # 应该显示 4.3.2
opencv --version    # 应该显示 4.13.0
```

### 3. 克隆项目

```bash
git clone https://github.com/ws1hope/jhcv_lib.git
cd jhcv_lib
```

### 4. 编译项目

```bash
# 创建构建目录
mkdir build && cd build

# 配置 CMake（CPU 模式）
cmake .. -DENABLE_CUDA=OFF

# 编译（使用 4 核并行编译）
make -j4
```

## 运行测试

### 准备测试数据

确保项目中有以下文件结构：

```
jhcv_lib/
├── models/           # 模型文件目录
│   ├── cls/         # 分类模型
│   ├── det/         # 检测模型
│   ├── ocr/         # OCR 模型
│   ├── seg/         # 分割模型
│   └── instance_segment/  # 实例分割模型
└── images/          # 测试图片目录
    ├── test_cls.jpg
    ├── test_det.png
    ├── test_ocr.png
    ├── test_seg.png
    └── test_isg.jpg
```

### 运行测试程序

```bash
cd build/bin

# 图像分类测试
./jhdeepcore_cls_test ../../models/cls/resnet18_cls3.onnx ../../images/test_cls.jpg cpu

# 目标检测测试
./jhdeepcore_det_test ../../models/det/best.onnx ../../images/test_det.png

# OCR 识别测试
./jhdeepcore_ocr_test ../../models/ocr/rec_model.onnx ../../models/ocr/best.yaml ../../images/test_ocr.png cpu

# 语义分割测试
./jhdeepcore_seg_test ../../models/seg/best.onnx ../../images/test_seg.png cpu

# 实例分割测试
./jhdeepcore_iseg_test ../../models/instance_segment/best.onnx ../../images/test_isg.jpg cpu
```

## 结果输出

测试结果会自动保存到 `result/` 文件夹：

```
result/
├── det.png   # 目标检测结果
├── seg.png   # 语义分割结果
├── iseg.png  # 实例分割结果
└── ocr.png   # OCR 识别结果
```

## 常见问题

### Q: 编译时出现 "library not found" 错误
**A**: 确保所有依赖都已通过 Homebrew 安装：
```bash
brew list opencv onnxruntime yaml-cpp
```

### Q: 链接错误 "undefined symbol"
**A**: 清理构建目录并重新编译：
```bash
cd build
rm -rf *
cmake .. -DENABLE_CUDA=OFF
make -j4
```

### Q: 程序运行时找不到模型文件
**A**: 检查模型文件路径是否正确，使用绝对路径或确保在正确的目录下运行。

### Q: 内存不足错误
**A**: 尝试减少并行编译线程数：
```bash
make -j2  # 使用 2 个线程而不是 4 个
```

## 目录结构

```
build/
├── bin/              # 可执行文件
│   ├── jhdeepcore_cls_test
│   ├── jhdeepcore_det_test
│   ├── jhdeepcore_ocr_test
│   ├── jhdeepcore_seg_test
│   └── jhdeepcore_iseg_test
├── lib/              # 静态库文件
│   ├── libjhcv_lib.a
│   └── libthird_party_lib.a
└── result/           # 测试结果输出
```

## 开发说明

### VS Code 调试

项目包含 Mac 的 VS Code 配置：
- `.vscode/tasks.json` - 编译任务
- `.vscode/launch.json` - 调试配置

使用 `F5` 键启动调试，或使用 `Cmd+Shift+P` 打开命令面板选择任务。

### 添加新的测试程序

1. 在 `test/` 目录下创建新的测试文件
2. 修改 `test/CMakeLists.txt` 添加新的可执行文件
3. 重新编译项目

## 性能建议

- **CPU 模式**: 适合推理和开发，兼容性好
- **并行编译**: 使用 `-j4` 可以显著加快编译速度
- **内存管理**: 处理大图片时注意内存使用

## 更新日志

- **2025-05-22**: 添加 macOS 平台支持
- 改进测试结果可视化
- 添加自动目录创建功能

## 技术支持

如遇到问题，请检查：
1. Xcode Command Line Tools 是否安装
2. Homebrew 依赖是否完整
3. 模型文件路径是否正确

## 许可证

请查看项目根目录的 LICENSE 文件。

---

**注意**: macOS 不支持 CUDA/TensorRT，所有推理在 CPU 上运行。推理速度取决于硬件配置。
