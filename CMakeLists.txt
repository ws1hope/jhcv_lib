cmake_minimum_required(VERSION 3.15)
project(inference_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-DUSE_ONNXRUNTIME)

# ========== 跨平台库路径配置 ==========
if(WIN32)
    # Windows 平台配置
    set(3RDPARTY_ROOT     "G:/3thirdparty")
    set(OPENCV_ROOT       "${3RDPARTY_ROOT}/opencv/build")
    set(ONNXRUNTIME_ROOT  "${3RDPARTY_ROOT}/onnxruntime")
    set(CUDA_ROOT         "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    set(YAML_CPP_ROOT     "${3RDPARTY_ROOT}/paddle_inference/third_party/install/yaml-cpp")

    # Windows 静态库
    set(OpenCV_LIBS       ${OPENCV_ROOT}/x64/vc16/lib/opencv_world490.lib)
    set(ONNX_CPU_LIBS     ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)

    # Windows 库目录
    set(OPENCV_LIB_DIR    ${OPENCV_ROOT}/x64/vc16/lib)
    set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
    set(YAML_CPP_LIB_DIR  ${YAML_CPP_ROOT}/lib)

elseif(APPLE)
    # macOS 平台配置 (Homebrew 安装路径)
    set(3RDPARTY_ROOT     "/usr/local")
    set(OPENCV_ROOT       "${3RDPARTY_ROOT}")
    set(ONNXRUNTIME_ROOT  "${3RDPARTY_ROOT}")
    set(YAML_CPP_ROOT     "${3RDPARTY_ROOT}")

    # macOS 动态库
    set(OpenCV_LIBS
        ${OPENCV_ROOT}/lib/libopencv_gapi.dylib
        ${OPENCV_ROOT}/lib/libopencv_stitching.dylib
        ${OPENCV_ROOT}/lib/libopencv_alphamat.dylib
        ${OPENCV_ROOT}/lib/libopencv_aruco.dylib
        ${OPENCV_ROOT}/lib/libopencv_bgsegm.dylib
        ${OPENCV_ROOT}/lib/libopencv_bioinspired.dylib
        ${OPENCV_ROOT}/lib/libopencv_ccalib.dylib
        ${OPENCV_ROOT}/lib/libopencv_dnn_objdetect.dylib
        ${OPENCV_ROOT}/lib/libopencv_dnn_superres.dylib
        ${OPENCV_ROOT}/lib/libopencv_dpm.dylib
        ${OPENCV_ROOT}/lib/libopencv_face.dylib
        ${OPENCV_ROOT}/lib/libopencv_freetype.dylib
        ${OPENCV_ROOT}/lib/libopencv_fuzzy.dylib
        ${OPENCV_ROOT}/lib/libopencv_hfs.dylib
        ${OPENCV_ROOT}/lib/libopencv_img_hash.dylib
        ${OPENCV_ROOT}/lib/libopencv_intensity_transform.dylib
        ${OPENCV_ROOT}/lib/libopencv_line_descriptor.dylib
        ${OPENCV_ROOT}/lib/libopencv_mcc.dylib
        ${OPENCV_ROOT}/lib/libopencv_quality.dylib
        ${OPENCV_ROOT}/lib/libopencv_rapid.dylib
        ${OPENCV_ROOT}/lib/libopencv_reg.dylib
        ${OPENCV_ROOT}/lib/libopencv_rgbd.dylib
        ${OPENCV_ROOT}/lib/libopencv_saliency.dylib
        ${OPENCV_ROOT}/lib/libopencv_sfm.dylib
        ${OPENCV_ROOT}/lib/libopencv_signal.dylib
        ${OPENCV_ROOT}/lib/libopencv_stereo.dylib
        ${OPENCV_ROOT}/lib/libopencv_structured_light.dylib
        ${OPENCV_ROOT}/lib/libopencv_phase_unwrapping.dylib
        ${OPENCV_ROOT}/lib/libopencv_superres.dylib
        ${OPENCV_ROOT}/lib/libopencv_optflow.dylib
        ${OPENCV_ROOT}/lib/libopencv_surface_matching.dylib
        ${OPENCV_ROOT}/lib/libopencv_tracking.dylib
        ${OPENCV_ROOT}/lib/libopencv_highgui.dylib
        ${OPENCV_ROOT}/lib/libopencv_datasets.dylib
        ${OPENCV_ROOT}/lib/libopencv_text.dylib
        ${OPENCV_ROOT}/lib/libopencv_plot.dylib
        ${OPENCV_ROOT}/lib/libopencv_videostab.dylib
        ${OPENCV_ROOT}/lib/libopencv_videoio.dylib
        ${OPENCV_ROOT}/lib/libopencv_viz.dylib
        ${OPENCV_ROOT}/lib/libopencv_wechat_qrcode.dylib
        ${OPENCV_ROOT}/lib/libopencv_xfeatures2d.dylib
        ${OPENCV_ROOT}/lib/libopencv_shape.dylib
        ${OPENCV_ROOT}/lib/libopencv_ml.dylib
        ${OPENCV_ROOT}/lib/libopencv_ximgproc.dylib
        ${OPENCV_ROOT}/lib/libopencv_video.dylib
        ${OPENCV_ROOT}/lib/libopencv_xobjdetect.dylib
        ${OPENCV_ROOT}/lib/libopencv_objdetect.dylib
        ${OPENCV_ROOT}/lib/libopencv_calib3d.dylib
        ${OPENCV_ROOT}/lib/libopencv_imgcodecs.dylib
        ${OPENCV_ROOT}/lib/libopencv_features2d.dylib
        ${OPENCV_ROOT}/lib/libopencv_dnn.dylib
        ${OPENCV_ROOT}/lib/libopencv_flann.dylib
        ${OPENCV_ROOT}/lib/libopencv_xphoto.dylib
        ${OPENCV_ROOT}/lib/libopencv_photo.dylib
        ${OPENCV_ROOT}/lib/libopencv_imgproc.dylib
        ${OPENCV_ROOT}/lib/libopencv_core.dylib
    )

    set(ONNX_CPU_LIBS     ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)

    # macOS 库目录
    set(OPENCV_LIB_DIR    ${OPENCV_ROOT}/lib)
    set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
    set(YAML_CPP_LIB_DIR  ${YAML_CPP_ROOT}/lib)

else()
    # Linux 平台配置
    set(3RDPARTY_ROOT     "/usr/local")
    set(OPENCV_ROOT       "${3RDPARTY_ROOT}")
    set(ONNXRUNTIME_ROOT  "${3RDPARTY_ROOT}")
    set(YAML_CPP_ROOT     "${3RDPARTY_ROOT}")

    # Linux 动态库
    set(OpenCV_LIBS       opencv_gapi opencv_stitching opencv_alphamat opencv_aruco
                          opencv_bgsegm opencv_bioinspired opencv_ccalib opencv_dnn_objdetect
                          opencv_dnn_superres opencv_dpm opencv_face opencv_freetype
                          opencv_fuzzy opencv_hfs opencv_img_hash opencv_intensity_transform
                          opencv_line_descriptor opencv_mcc opencv_quality opencv_rapid
                          opencv_reg opencv_rgbd opencv_saliency opencv_sfm opencv_signal
                          opencv_stereo opencv_structured_light opencv_phase_unwrapping
                          opencv_superres opencv_optflow opencv_surface_matching opencv_tracking
                          opencv_highgui opencv_datasets opencv_text opencv_plot
                          opencv_videostab opencv_videoio opencv_viz opencv_wechat_qrcode
                          opencv_xfeatures2d opencv_shape opencv_ml opencv_ximgproc
                          opencv_video opencv_xobjdetect opencv_objdetect opencv_calib3d
                          opencv_imgcodecs opencv_features2d opencv_dnn opencv_flann
                          opencv_xphoto opencv_photo opencv_imgproc opencv_core)

    set(ONNX_CPU_LIBS     ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)

    # Linux 库目录
    set(OPENCV_LIB_DIR    ${OPENCV_ROOT}/lib)
    set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
    set(YAML_CPP_LIB_DIR  ${YAML_CPP_ROOT}/lib)
endif()

# ========== CUDA 配置 ==========
option(ENABLE_CUDA "Enable CUDA/TensorRT GPU support" OFF)

if(ENABLE_CUDA)
    if(WIN32)
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA support: ON (Windows)")
        set(ONNX_GPU_LIBS
            ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.lib
            ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib
            ${CUDA_ROOT}/lib/x64/cudart.lib
        )
        set(CUDA_LIB_DIR ${CUDA_ROOT}/lib/x64)
        set(CUDA_INCLUDE_DIR ${CUDA_ROOT}/include)
    else()
        message(WARNING "CUDA/TensorRT is not supported on macOS/Linux in this configuration, forcing CPU-only mode")
        set(ENABLE_CUDA OFF)
        set(ONNX_GPU_LIBS)
    endif()
else()
    message(STATUS "CUDA support: OFF (CPU-only mode)")
    set(ONNX_GPU_LIBS)
endif()

# ========== 头文件目录 ==========
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cv_infer
    ${OPENCV_ROOT}/include
    ${OPENCV_ROOT}/include/opencv4
    ${ONNXRUNTIME_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include/onnxruntime
    ${YAML_CPP_ROOT}/include
)

if(ENABLE_CUDA AND WIN32)
    include_directories(${CUDA_INCLUDE_DIR})
endif()

# ========== 库文件目录 ==========
link_directories(
    ${OPENCV_LIB_DIR}
    ${ONNXRUNTIME_LIB_DIR}
    ${YAML_CPP_LIB_DIR}
)

if(ENABLE_CUDA AND WIN32)
    link_directories(${CUDA_LIB_DIR})
endif()

# ========== 添加子目录 ==========
add_subdirectory(src)

option(BUILD_TEST "Build test programs" ON)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

message(STATUS "=========================================")
message(STATUS "${PROJECT_NAME} Configuration Summary")
message(STATUS "=========================================")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Test: ${BUILD_TEST}")
message(STATUS "Enable CUDA: ${ENABLE_CUDA}")
message(STATUS "OpenCV Root: ${OPENCV_ROOT}")
message(STATUS "ONNX Runtime Root: ${ONNXRUNTIME_ROOT}")
message(STATUS "YAML-CPP Root: ${YAML_CPP_ROOT}")
message(STATUS "=========================================")
