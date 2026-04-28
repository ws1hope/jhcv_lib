add_executable(ocr_inference ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(ocr_inference PRIVATE inference ${PADDLE_LIBS} paddle_ocr_lib)
