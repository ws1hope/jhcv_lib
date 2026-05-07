set(JHDEEPCORE_UTILS_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_utils/config_loader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_utils/device_utils.cpp
)

set(JHDEEPCORE_INFERENCE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_inference/base_inference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_inference/onnx_inference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_inference/inference_factory.cpp
)

set(JHDEEPCORE_CLASSIFY_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_classify/classifier.cpp
)

set(JHDEEPCORE_DETECT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_detect/detector.cpp
)

set(JHDEEPCORE_SEGMENT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_segment/segmenter.cpp
)

set(JHDEEPCORE_INSTANCE_SEGMENT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_instance_segment/instance_segmenter.cpp
)

set(JHDEEPCORE_OCR_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/jhdeepcore_ocr/ocr_recognizer.cpp
)

set(JHDEEPCORE_MAIN_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/JHDeepCore.cpp
)

set(JHDEEPCORE_ALL_SOURCES
    ${JHDEEPCORE_UTILS_SOURCES}
    ${JHDEEPCORE_INFERENCE_SOURCES}
    ${JHDEEPCORE_CLASSIFY_SOURCES}
    ${JHDEEPCORE_DETECT_SOURCES}
    ${JHDEEPCORE_SEGMENT_SOURCES}
    ${JHDEEPCORE_INSTANCE_SEGMENT_SOURCES}
    ${JHDEEPCORE_OCR_SOURCES}
    ${JHDEEPCORE_MAIN_SOURCES}
)

add_library(jhdeepcore STATIC ${JHDEEPCORE_ALL_SOURCES})

target_include_directories(jhdeepcore
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PADDLE_ROOT}/third_party/install/yaml-cpp/include
)

target_compile_definitions(jhdeepcore PRIVATE
    YAMLCPP_FOUND
    ONNXRUNTIME_FOUND
    USE_CUDA
    YAML_CPP_STATIC_DEFINE
)

if(MSVC)
    target_compile_options(jhdeepcore PRIVATE /utf-8)
    set_property(TARGET jhdeepcore PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

target_link_libraries(jhdeepcore
    PUBLIC ${OpenCV_LIBS} ${ONNX_LIBS}
    PRIVATE ${PADDLE_ROOT}/third_party/install/yaml-cpp/lib/yaml-cpp.lib
)

target_link_directories(jhdeepcore PRIVATE
    ${PADDLE_ROOT}/third_party/install/yaml-cpp/lib
)

install(TARGETS jhdeepcore DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
