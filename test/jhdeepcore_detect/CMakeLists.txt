add_executable(jhdeepcore_det_test ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(jhdeepcore_det_test PRIVATE jhdeepcore)
if(MSVC)
    set_property(TARGET jhdeepcore_det_test PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
