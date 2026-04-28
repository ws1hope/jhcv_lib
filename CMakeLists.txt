cmake_minimum_required(VERSION 3.15)
project(inference_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-DUSE_CUDA)
add_definitions(-DUSE_ONNXRUNTIME)

set(3RDPARTY_ROOT     "G:/3thirdparty")
set(OPENCV_ROOT       "${3RDPARTY_ROOT}/opencv/build")
set(ONNXRUNTIME_ROOT  "${3RDPARTY_ROOT}/onnxruntime")
set(CUDA_ROOT         "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
set(TENSORRT_ROOT     "${3RDPARTY_ROOT}/tensorrt")
set(PADDLE_ROOT       "${3RDPARTY_ROOT}/paddle_inference")

set(OpenCV_LIBS ${OPENCV_ROOT}/x64/vc16/lib/opencv_world490.lib)

set(ONNX_LIBS
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_tensorrt.lib
    ${TENSORRT_ROOT}/lib/nvinfer.lib
    ${TENSORRT_ROOT}/lib/nvinfer_plugin.lib
    ${CUDA_ROOT}/lib/x64/cudart.lib
)

set(PADDLE_LIBS
    ${PADDLE_ROOT}/paddle/lib/paddle_inference.lib
    ${PADDLE_ROOT}/paddle/lib/common.lib
    ${PADDLE_ROOT}/third_party/install/protobuf/lib/libprotobuf.lib
    ${PADDLE_ROOT}/third_party/install/gflags/lib/gflags_static.lib
    ${PADDLE_ROOT}/third_party/install/glog/lib/glog.lib
    ${PADDLE_ROOT}/third_party/install/xxhash/lib/xxhash.lib
    ${PADDLE_ROOT}/third_party/install/yaml-cpp/lib/yaml-cpp.lib
    ${PADDLE_ROOT}/third_party/install/mklml/lib/mklml.lib
    ${PADDLE_ROOT}/third_party/install/mklml/lib/libiomp5md.lib
    ${PADDLE_ROOT}/third_party/install/onednn/lib/mkldnn.lib
)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${OPENCV_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include
    ${CUDA_ROOT}/include
    ${TENSORRT_ROOT}/include
    ${PADDLE_ROOT}/paddle/include
    ${PADDLE_ROOT}/third_party/install/protobuf/include
    ${PADDLE_ROOT}/third_party/install/gflags/include
    ${PADDLE_ROOT}/third_party/install/glog/include
    ${PADDLE_ROOT}/third_party/install/xxhash/include
    ${PADDLE_ROOT}/third_party/install/yaml-cpp/include
    ${PADDLE_ROOT}/third_party/install/mklml/include
    ${PADDLE_ROOT}/third_party/install/onednn/include
)

link_directories(
    ${OPENCV_ROOT}/x64/vc16/lib
    ${ONNXRUNTIME_ROOT}/lib
    ${CUDA_ROOT}/lib/x64
    ${TENSORRT_ROOT}/lib
    ${PADDLE_ROOT}/paddle/lib
    ${PADDLE_ROOT}/third_party/install/protobuf/lib
    ${PADDLE_ROOT}/third_party/install/gflags/lib
    ${PADDLE_ROOT}/third_party/install/glog/lib
    ${PADDLE_ROOT}/third_party/install/xxhash/lib
    ${PADDLE_ROOT}/third_party/install/yaml-cpp/lib
    ${PADDLE_ROOT}/third_party/install/mklml/lib
    ${PADDLE_ROOT}/third_party/install/onednn/lib
)

# set(PADDLE_OCR_BUILD_LIB ON CACHE BOOL "Build as library" FORCE)
# add_subdirectory(Paddle_VX)

add_subdirectory(src)
add_subdirectory(test/yolo)
add_subdirectory(test/cls)
# add_subdirectory(test/ocr)
