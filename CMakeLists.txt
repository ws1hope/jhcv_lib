cmake_minimum_required(VERSION 3.15)
project(inference_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-DUSE_ONNXRUNTIME)

set(3RDPARTY_ROOT     "G:/3thirdparty")
set(OPENCV_ROOT       "${3RDPARTY_ROOT}/opencv/build")
set(ONNXRUNTIME_ROOT  "${3RDPARTY_ROOT}/onnxruntime")
set(CUDA_ROOT         "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
set(TENSORRT_ROOT     "${3RDPARTY_ROOT}/tensorrt")
set(PADDLE_ROOT       "${3RDPARTY_ROOT}/paddle_inference")

set(OpenCV_LIBS ${OPENCV_ROOT}/x64/vc16/lib/opencv_world490.lib)

option(ENABLE_CUDA "Enable CUDA/TensorRT GPU support" ON)

if(ENABLE_CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA support: ON")
else()
    message(STATUS "CUDA support: OFF (CPU-only mode)")
endif()

set(ONNX_CPU_LIBS
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib
)

if(ENABLE_CUDA)
    set(ONNX_GPU_LIBS
        ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.lib
        ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib
        ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_tensorrt.lib
        ${TENSORRT_ROOT}/lib/nvinfer.lib
        ${TENSORRT_ROOT}/lib/nvinfer_plugin.lib
        ${CUDA_ROOT}/lib/x64/cudart.lib
    )
else()
    set(ONNX_GPU_LIBS)
endif()

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
    ${CMAKE_SOURCE_DIR}/src/cv_infer
    ${OPENCV_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include
    ${PADDLE_ROOT}/paddle/include
    ${PADDLE_ROOT}/third_party/install/protobuf/include
    ${PADDLE_ROOT}/third_party/install/gflags/include
    ${PADDLE_ROOT}/third_party/install/glog/include
    ${PADDLE_ROOT}/third_party/install/xxhash/include
    ${PADDLE_ROOT}/third_party/install/yaml-cpp/include
    ${PADDLE_ROOT}/third_party/install/mklml/include
    ${PADDLE_ROOT}/third_party/install/onednn/include
)

if(ENABLE_CUDA)
    include_directories(
        ${CUDA_ROOT}/include
        ${TENSORRT_ROOT}/include
    )
endif()

link_directories(
    ${OPENCV_ROOT}/x64/vc16/lib
    ${ONNXRUNTIME_ROOT}/lib
    ${PADDLE_ROOT}/paddle/lib
    ${PADDLE_ROOT}/third_party/install/protobuf/lib
    ${PADDLE_ROOT}/third_party/install/gflags/lib
    ${PADDLE_ROOT}/third_party/install/glog/lib
    ${PADDLE_ROOT}/third_party/install/xxhash/lib
    ${PADDLE_ROOT}/third_party/install/yaml-cpp/lib
    ${PADDLE_ROOT}/third_party/install/mklml/lib
    ${PADDLE_ROOT}/third_party/install/onednn/lib
)

if(ENABLE_CUDA)
    link_directories(
        ${CUDA_ROOT}/lib/x64
        ${TENSORRT_ROOT}/lib
    )
endif()

add_subdirectory(src)

option(BUILD_TEST "Build test programs" ON)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

message(STATUS "${PROJECT_NAME} | C++${CMAKE_CXX_STANDARD} | BUILD_TEST: ${BUILD_TEST} | ENABLE_CUDA: ${ENABLE_CUDA}")
