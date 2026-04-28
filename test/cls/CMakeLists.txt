add_executable(cls_inference ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(cls_inference PRIVATE inference)
