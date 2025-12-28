add_executable(forward_elementwise_exe forward_elementwise/test.cc)
target_link_libraries(forward_elementwise_exe Tensor)
target_include_directories(forward_elementwise_exe PRIVATE ..)

add_executable(backward_elementwise_exe backward_elementwise/test.cc)
target_link_libraries(backward_elementwise_exe Tensor)
target_include_directories(backward_elementwise_exe PRIVATE ..)

set(Python3_ROOT_DIR "${CMAKE_SOURCE_DIR}/../venv")
set(Python3_FIND_VIRTUALENV FIRST)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Register the tests with ctest
add_test(NAME forward_elementwise_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_and_compare.py 
            $<TARGET_FILE:forward_elementwise_exe> 
            ${CMAKE_CURRENT_SOURCE_DIR}/forward_elementwise/test.py
)

add_test(NAME backward_elementwise_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_and_compare.py 
            $<TARGET_FILE:backward_elementwise_exe> 
            ${CMAKE_CURRENT_SOURCE_DIR}/backward_elementwise/test.py
)
