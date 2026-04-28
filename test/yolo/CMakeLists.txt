add_executable(yolo_inference ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(yolo_inference PRIVATE inference)
