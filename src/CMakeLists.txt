file(GLOB YOLO_SRC ${CMAKE_CURRENT_LIST_DIR}/yolo_inference.cpp)
file(GLOB CLS_SRC ${CMAKE_CURRENT_LIST_DIR}/cls_inference.cpp)
file(GLOB OCR_SRC ${CMAKE_CURRENT_LIST_DIR}/ocr_inference.cpp)
file(GLOB HEADER_FILES ${CMAKE_CURRENT_LIST_DIR}/*.h)

add_library(inference STATIC ${YOLO_SRC} ${CLS_SRC} ${OCR_SRC} ${HEADER_FILES})

target_include_directories(inference PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(inference PRIVATE ${OpenCV_LIBS} ${ONNX_LIBS})

install(TARGETS inference DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION include)