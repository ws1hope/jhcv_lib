cmake_minimum_required(VERSION 3.15)
project(inference_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-DUSE_ONNXRUNTIME)

# ========== 跨平台配置（路径 + 库 + 头文件 + 库目录） ==========
if(WIN32)
    set(3RDPARTY_ROOT     "G:/3thirdparty")
    set(OPENCV_ROOT       "${3RDPARTY_ROOT}/opencv/build")
    set(ONNXRUNTIME_ROOT  "${3RDPARTY_ROOT}/onnxruntime")
    set(CUDA_ROOT         "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    set(YAML_CPP_ROOT     "${3RDPARTY_ROOT}/paddle_inference/third_party/install/yaml-cpp")

    set(OpenCV_LIBS       ${OPENCV_ROOT}/x64/vc16/lib/opencv_world490.lib)
    set(ONNX_CPU_LIBS     ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)

    set(INCLUDE_DIRS
        ${OPENCV_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include
        ${YAML_CPP_ROOT}/include
    )
    set(LIB_DIRS
        ${OPENCV_ROOT}/x64/vc16/lib
        ${ONNXRUNTIME_ROOT}/lib
        ${YAML_CPP_ROOT}/lib
    )
    option(ENABLE_CUDA "Enable CUDA/TensorRT GPU support" ON)

elseif(APPLE)
    set(3RDPARTY_ROOT     "/usr/local")
    set(OPENCV_ROOT       "${3RDPARTY_ROOT}")
    set(ONNXRUNTIME_ROOT  "${3RDPARTY_ROOT}")
    set(YAML_CPP_ROOT     "${3RDPARTY_ROOT}")

    file(GLOB OpenCV_LIBS "${OPENCV_ROOT}/lib/libopencv_*.dylib")
    set(ONNX_CPU_LIBS     ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)

    set(INCLUDE_DIRS
        ${OPENCV_ROOT}/include
        ${OPENCV_ROOT}/include/opencv4
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include/onnxruntime
        ${YAML_CPP_ROOT}/include
    )
    set(LIB_DIRS
        ${OPENCV_ROOT}/lib
        ${YAML_CPP_ROOT}/lib
    )
    option(ENABLE_CUDA "Enable CUDA/TensorRT GPU support" OFF)

else()
    set(3RDPARTY_ROOT     "/usr/local")
    set(OPENCV_ROOT       "${3RDPARTY_ROOT}")
    set(ONNXRUNTIME_ROOT  "${3RDPARTY_ROOT}")
    set(YAML_CPP_ROOT     "${3RDPARTY_ROOT}")

    file(GLOB OpenCV_LIBS "${OPENCV_ROOT}/lib/libopencv_*.so")
    set(ONNX_CPU_LIBS     ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)

    set(INCLUDE_DIRS
        ${OPENCV_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include
        ${YAML_CPP_ROOT}/include
    )
    set(LIB_DIRS
        ${OPENCV_ROOT}/lib
        ${YAML_CPP_ROOT}/lib
    )
    option(ENABLE_CUDA "Enable CUDA/TensorRT GPU support" OFF)
endif()

# ========== CUDA ==========
if(ENABLE_CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA support: ON")
    if(WIN32)
        set(ONNX_GPU_LIBS
            ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.lib
            ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib
            ${CUDA_ROOT}/lib/x64/cudart.lib
        )
        list(APPEND INCLUDE_DIRS ${CUDA_ROOT}/include)
        list(APPEND LIB_DIRS ${CUDA_ROOT}/lib/x64)
    endif()
else()
    message(STATUS "CUDA support: OFF (CPU-only mode)")
    set(ONNX_GPU_LIBS)
endif()

# ========== 应用头文件和库目录 ==========
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cv_infer
    ${INCLUDE_DIRS}
)
link_directories(${LIB_DIRS})

# ========== 构建目标 ==========
add_subdirectory(src)

option(BUILD_TEST "Build test programs" ON)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

message(STATUS "=========================================")
message(STATUS "${PROJECT_NAME} | ${CMAKE_SYSTEM_NAME}")
message(STATUS "C++${CMAKE_CXX_STANDARD} | BUILD_TEST: ${BUILD_TEST} | ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "OpenCV: ${OPENCV_ROOT}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "YAML-CPP: ${YAML_CPP_ROOT}")
message(STATUS "=========================================")
