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
set(YAML_CPP_ROOT    "${3RDPARTY_ROOT}/paddle_inference/third_party/install/yaml-cpp")

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
        ${CUDA_ROOT}/lib/x64/cudart.lib
    )
else()
    set(ONNX_GPU_LIBS)
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cv_infer
    ${OPENCV_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include
    ${YAML_CPP_ROOT}/include
)

if(ENABLE_CUDA)
    include_directories(
        ${CUDA_ROOT}/include
    )
endif()

link_directories(
    ${OPENCV_ROOT}/x64/vc16/lib
    ${ONNXRUNTIME_ROOT}/lib
    ${YAML_CPP_ROOT}/lib
)

if(ENABLE_CUDA)
    link_directories(
        ${CUDA_ROOT}/lib/x64
    )
endif()

add_subdirectory(src)

option(BUILD_TEST "Build test programs" ON)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

message(STATUS "${PROJECT_NAME} | C++${CMAKE_CXX_STANDARD} | BUILD_TEST: ${BUILD_TEST} | ENABLE_CUDA: ${ENABLE_CUDA}")
