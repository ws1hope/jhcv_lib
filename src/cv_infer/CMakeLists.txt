file(GLOB YOLO_SRC ${CMAKE_CURRENT_LIST_DIR}/yolo_inference.cpp)
file(GLOB CLS_SRC ${CMAKE_CURRENT_LIST_DIR}/cls_inference.cpp)
file(GLOB OCR_SRC ${CMAKE_CURRENT_LIST_DIR}/ocr_inference.cpp)
file(GLOB CVHELP_SRC ${CMAKE_CURRENT_LIST_DIR}/cvHelp.cpp)
file(GLOB PENMA_SRC ${CMAKE_CURRENT_LIST_DIR}/penma_rec_inference.cpp)
file(GLOB PENMA_DLL_SRC ${CMAKE_CURRENT_LIST_DIR}/penma_rec_dll.cpp)
file(GLOB HEADER_FILES ${CMAKE_CURRENT_LIST_DIR}/*.h)

add_library(inference STATIC ${YOLO_SRC} ${CLS_SRC} ${OCR_SRC} ${CVHELP_SRC} ${PENMA_SRC} ${HEADER_FILES})

target_include_directories(inference PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(inference PRIVATE ${OpenCV_LIBS} ${ONNX_LIBS})

install(TARGETS inference DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION include)

# ========== penma_rec_dll ==========
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
