# ========== 第三方库 ==========
set(THIRD_PARTY_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/third_party/httplib.cc
)

add_library(third_party_lib STATIC ${THIRD_PARTY_SOURCES})

target_include_directories(third_party_lib PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/third_party
)

target_compile_definitions(third_party_lib PUBLIC)

if(MSVC)
    set_property(TARGET third_party_lib PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(WIN32)
    target_link_libraries(third_party_lib PUBLIC ws2_32)
endif()

set_target_properties(third_party_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ========== 源文件列表 ==========
# cv_infer 模块
set(CV_INFER_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/cv_infer/yolo_inference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cv_infer/cls_inference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cv_infer/ocr_inference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cv_infer/cvHelp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cv_infer/penma_rec_inference.cpp
)

# ai_platform 模块 - utils
set(AI_PLATFORM_UTILS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_utils/config_loader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_utils/device_utils.cpp
)

# ai_platform 模块 - inference
set(AI_PLATFORM_INFERENCE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_inference/base_inference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_inference/onnx_inference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_inference/inference_factory.cpp
)

# ai_platform 模块 - 业务层
set(AI_PLATFORM_CLASSIFY_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_classify/classifier.cpp
)

set(AI_PLATFORM_DETECT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_detect/detector.cpp
)

set(AI_PLATFORM_SEGMENT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_segment/segmenter.cpp
)

set(AI_PLATFORM_INSTANCE_SEGMENT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_instance_segment/instance_segmenter.cpp
)

set(AI_PLATFORM_OCR_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/jhdeepcore_ocr/ocr_recognizer.cpp
)

# 对外接口
set(AI_PLATFORM_MAIN_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/ai_platform/JHDeepCore.cpp
)

# nisco_project 模块
set(NISCO_PROJECT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/nisco_project/dabang_jiguang_service.cpp
)

# 所有源文件
set(ALL_SOURCES
    ${CV_INFER_SOURCES}
    ${AI_PLATFORM_UTILS_SOURCES}
    ${AI_PLATFORM_INFERENCE_SOURCES}
    ${AI_PLATFORM_CLASSIFY_SOURCES}
    ${AI_PLATFORM_DETECT_SOURCES}
    ${AI_PLATFORM_SEGMENT_SOURCES}
    ${AI_PLATFORM_INSTANCE_SEGMENT_SOURCES}
    ${AI_PLATFORM_OCR_SOURCES}
    ${AI_PLATFORM_MAIN_SOURCES}
    ${NISCO_PROJECT_SOURCES}
)

# ========== 构建统一静态库 jhcv_lib ==========
add_library(jhcv_lib STATIC ${ALL_SOURCES})

target_include_directories(jhcv_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cv_infer
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_platform
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_platform/jhdeepcore_utils
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_platform/jhdeepcore_inference
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_platform/jhdeepcore_classify
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_platform/jhdeepcore_detect
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_platform/jhdeepcore_segment
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_platform/jhdeepcore_instance_segment
    ${CMAKE_CURRENT_SOURCE_DIR}/ai_platform/jhdeepcore_ocr
    ${CMAKE_CURRENT_SOURCE_DIR}/nisco_project
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_definitions(jhcv_lib PRIVATE
    USE_ONNXRUNTIME
    YAMLCPP_FOUND
    ONNXRUNTIME_FOUND
    YAML_CPP_STATIC_DEFINE
)

if(ENABLE_CUDA)
    target_compile_definitions(jhcv_lib PRIVATE USE_CUDA)
endif()

if(MSVC)
    set_property(TARGET jhcv_lib PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set_source_files_properties(
        ${AI_PLATFORM_UTILS_SOURCES}
        ${AI_PLATFORM_INFERENCE_SOURCES}
        ${AI_PLATFORM_CLASSIFY_SOURCES}
        ${AI_PLATFORM_DETECT_SOURCES}
        ${AI_PLATFORM_SEGMENT_SOURCES}
        ${AI_PLATFORM_INSTANCE_SEGMENT_SOURCES}
        ${AI_PLATFORM_OCR_SOURCES}
        ${AI_PLATFORM_MAIN_SOURCES}
        ${NISCO_PROJECT_SOURCES}
        PROPERTIES COMPILE_FLAGS "/utf-8"
    )
endif()

target_link_libraries(jhcv_lib PUBLIC
    ${OpenCV_LIBS}
    ${ONNX_CPU_LIBS}
    third_party_lib
    PRIVATE
    ${PADDLE_ROOT}/third_party/install/yaml-cpp/lib/yaml-cpp.lib
)

if(ENABLE_CUDA)
    target_link_libraries(jhcv_lib PUBLIC ${ONNX_GPU_LIBS})
endif()

target_link_directories(jhcv_lib PRIVATE
    ${PADDLE_ROOT}/third_party/install/yaml-cpp/lib
)

set_target_properties(jhcv_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========== 安装规则 ==========
install(TARGETS jhcv_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# ========== penma_rec_dll 动态库 ==========
file(GLOB PENMA_DLL_SRC ${CMAKE_CURRENT_LIST_DIR}/cv_infer/penma_rec_dll.cpp)
file(GLOB PENMA_HEADER_FILES ${CMAKE_CURRENT_LIST_DIR}/cv_infer/*.h)

add_library(penma_rec_dll SHARED ${PENMA_DLL_SRC} ${PENMA_HEADER_FILES})

target_compile_definitions(penma_rec_dll PRIVATE PENMA_REC_DLL_EXPORTS)

set_target_properties(penma_rec_dll PROPERTIES
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(WIN32)
    set_target_properties(penma_rec_dll PROPERTIES
        LINK_FLAGS "/DEF:\"${CMAKE_CURRENT_LIST_DIR}/cv_infer/penma_rec_dll.def\""
    )
endif()

if(MSVC)
    set_property(TARGET penma_rec_dll PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_include_directories(penma_rec_dll PRIVATE ${CMAKE_CURRENT_LIST_DIR}/cv_infer)

target_link_libraries(penma_rec_dll PRIVATE jhcv_lib ${OpenCV_LIBS} ${ONNX_CPU_LIBS})

if(ENABLE_CUDA)
    target_link_libraries(penma_rec_dll PRIVATE ${ONNX_GPU_LIBS})
endif()

install(TARGETS penma_rec_dll DESTINATION bin)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/cv_infer/penma_rec_dll.h DESTINATION include)
