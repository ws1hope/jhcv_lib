# ========== Existing inference library (unchanged) ==========
file(GLOB YOLO_SRC ${CMAKE_CURRENT_LIST_DIR}/yolo_inference.cpp)
file(GLOB CLS_SRC ${CMAKE_CURRENT_LIST_DIR}/cls_inference.cpp)
file(GLOB OCR_SRC ${CMAKE_CURRENT_LIST_DIR}/ocr_inference.cpp)
file(GLOB CVHELP_SRC ${CMAKE_CURRENT_LIST_DIR}/cvHelp.cpp)
file(GLOB PENMA_SRC ${CMAKE_CURRENT_LIST_DIR}/penma_rec_inference.cpp)
file(GLOB PENMA_DLL_SRC ${CMAKE_CURRENT_LIST_DIR}/penma_rec_dll.cpp)
file(GLOB HEADER_FILES ${CMAKE_CURRENT_LIST_DIR}/*.h)

add_library(inference STATIC ${YOLO_SRC} ${CLS_SRC} ${OCR_SRC} ${CVHELP_SRC} ${PENMA_SRC} ${HEADER_FILES})

target_include_directories(inference PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(inference PRIVATE ${OpenCV_LIBS} ${ONNX_LIBS})

install(TARGETS inference DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION include)

# ========== penma_rec_dll (unchanged) ==========
add_library(penma_rec_dll SHARED ${PENMA_DLL_SRC} ${HEADER_FILES})

target_compile_definitions(penma_rec_dll PRIVATE PENMA_REC_DLL_EXPORTS)

set_target_properties(penma_rec_dll PROPERTIES
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(WIN32)
    set_target_properties(penma_rec_dll PROPERTIES
        LINK_FLAGS "/DEF:\"${CMAKE_CURRENT_LIST_DIR}/penma_rec_dll.def\""
    )
endif()

target_include_directories(penma_rec_dll PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(penma_rec_dll PRIVATE inference ${OpenCV_LIBS} ${ONNX_LIBS})

install(TARGETS penma_rec_dll DESTINATION bin)
install(FILES penma_rec_dll.h DESTINATION include)

# ========== JHDeepCore library (new) ==========
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
