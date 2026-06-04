# ========== cv_infer 测试 ==========
add_executable(yolo_inference ${CMAKE_CURRENT_SOURCE_DIR}/yolo/main.cpp)
add_executable(cls_inference ${CMAKE_CURRENT_SOURCE_DIR}/cls/main.cpp)
add_executable(ocr_inference ${CMAKE_CURRENT_SOURCE_DIR}/ocr/main.cpp)
add_executable(penma_rec ${CMAKE_CURRENT_SOURCE_DIR}/penma_rec/main.cpp)

add_executable(penma_rec_dll_test ${CMAKE_CURRENT_SOURCE_DIR}/penma_rec_dll/main.cpp)

# ========== ai_platform 测试 ==========
add_executable(jhdeepcore_cls_test ${CMAKE_CURRENT_SOURCE_DIR}/jhdeepcore_classify/main.cpp)
add_executable(jhdeepcore_det_test ${CMAKE_CURRENT_SOURCE_DIR}/jhdeepcore_detect/main.cpp)
add_executable(jhdeepcore_seg_test ${CMAKE_CURRENT_SOURCE_DIR}/jhdeepcore_segment/main.cpp)
add_executable(jhdeepcore_iseg_test ${CMAKE_CURRENT_SOURCE_DIR}/jhdeepcore_instance_segment/main.cpp)
add_executable(jhdeepcore_ocr_test ${CMAKE_CURRENT_SOURCE_DIR}/jhdeepcore_ocr/main.cpp)
add_executable(jhdeepcore_tracker_test ${CMAKE_CURRENT_SOURCE_DIR}/jhdeepcore_tracker/main.cpp)

# ========== 大棒激光 HTTP 服务 ==========
add_executable(dabang_jiguang ${CMAKE_CURRENT_SOURCE_DIR}/dabang_jiguang/main.cpp)

# ========== 贴标 OCR HTTP 服务 ==========
add_executable(tiebiao ${CMAKE_CURRENT_SOURCE_DIR}/tiebiao/main.cpp)

# ========== 调度服务 ==========
add_executable(dispatch ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/main.cpp)

# ========== 统一设置 ==========
set(TEST_TARGETS
    yolo_inference
    cls_inference
    ocr_inference
    penma_rec
    jhdeepcore_cls_test
    jhdeepcore_det_test
    jhdeepcore_seg_test
    jhdeepcore_iseg_test
    jhdeepcore_ocr_test
    jhdeepcore_tracker_test
    dabang_jiguang
    tiebiao
    dispatch
)

foreach(TARGET ${TEST_TARGETS})
    target_link_libraries(${TARGET} PRIVATE jhcv_lib)
    if(MSVC)
        set_property(TARGET ${TARGET} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endforeach()

set_target_properties(${TEST_TARGETS} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========== 大棒激光 HTTP 服务 ==========
if(MSVC)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/dabang_jiguang/main.cpp
        PROPERTIES COMPILE_FLAGS "/utf-8"
    )
endif()

# ========== 贴标 OCR HTTP 服务 ==========
if(MSVC)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/tiebiao/main.cpp
        PROPERTIES COMPILE_FLAGS "/utf-8"
    )
endif()

# ========== 调度服务 ==========
if(MSVC)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/dispatch/main.cpp
        PROPERTIES COMPILE_FLAGS "/utf-8"
    )
endif()

# ========== penma_rec_dll 测试（链接动态库） ==========
target_include_directories(penma_rec_dll_test PRIVATE ${CMAKE_SOURCE_DIR}/src/cv_infer)
target_link_libraries(penma_rec_dll_test PRIVATE penma_rec_dll)
if(MSVC)
    set_property(TARGET penma_rec_dll_test PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
set_target_properties(penma_rec_dll_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
