add_executable(loss_function_exe loss_function/test.cc)
target_link_libraries(loss_function_exe Module Tensor util)
target_include_directories(loss_function_exe PRIVATE ..)

# Force CMake to look for the virtual environment first
set(Python3_FIND_VIRTUALENV ONLY)
set(Python3_ROOT_DIR "${CMAKE_SOURCE_DIR}/../venv")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(NOT Python3_FOUND)
    message(FATAL_ERROR "Could not find Python 3 in the virtual environment at ${Python3_ROOT_DIR}")
endif()

add_test(NAME loss_function_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_and_compare.py 
            $<TARGET_FILE:loss_function_exe> 
            ${CMAKE_CURRENT_SOURCE_DIR}/loss_function/test.py
)
