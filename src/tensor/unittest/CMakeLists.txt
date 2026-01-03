add_executable(forward_elementwise_exe forward_elementwise/test.cc)
target_link_libraries(forward_elementwise_exe Tensor)
target_include_directories(forward_elementwise_exe PRIVATE ..)

add_executable(backward_elementwise_exe backward_elementwise/test.cc)
target_link_libraries(backward_elementwise_exe Tensor)
target_include_directories(backward_elementwise_exe PRIVATE ..)

add_executable(forward_matmul_exe forward_matmul/test.cc)
target_link_libraries(forward_matmul_exe Tensor)
target_include_directories(forward_matmul_exe PRIVATE ..)

add_executable(backward_matmul_exe backward_matmul/test.cc)
target_link_libraries(backward_matmul_exe Tensor)
target_include_directories(backward_matmul_exe PRIVATE ..)

add_executable(reduction_expansion_exe reduction_expansion/test.cc)
target_link_libraries(reduction_expansion_exe Tensor)
target_include_directories(reduction_expansion_exe PRIVATE ..)

# Force CMake to look for the virtual environment first
set(Python3_FIND_VIRTUALENV ONLY)
set(Python3_ROOT_DIR "${CMAKE_SOURCE_DIR}/../venv")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Could not find Python 3 in the virtual environment at ${Python3_ROOT_DIR}")
endif()

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

add_test(NAME forward_matmul_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_and_compare.py 
            $<TARGET_FILE:forward_matmul_exe> 
            ${CMAKE_CURRENT_SOURCE_DIR}/forward_matmul/test.py
)

add_test(NAME backward_matmul_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_and_compare.py 
            $<TARGET_FILE:backward_matmul_exe> 
            ${CMAKE_CURRENT_SOURCE_DIR}/backward_matmul/test.py
)

add_test(NAME reduction_expansion_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_and_compare.py 
            $<TARGET_FILE:reduction_expansion_exe> 
            ${CMAKE_CURRENT_SOURCE_DIR}/reduction_expansion/test.py
)
