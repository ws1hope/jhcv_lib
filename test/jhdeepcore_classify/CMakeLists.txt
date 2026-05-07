add_executable(jhdeepcore_cls_test ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(jhdeepcore_cls_test PRIVATE jhdeepcore)
if(MSVC)
    set_property(TARGET jhdeepcore_cls_test PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
