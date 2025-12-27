add_executable(test_forward forward_test/test_forward.cc)
target_link_libraries(test_forward Tensor)
target_include_directories(test_forward
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)

set(Python3_ROOT_DIR "${CMAKE_SOURCE_DIR}/../venv")
set(Python3_FIND_VIRTUALENV FIRST)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME test_forward_broadcasting
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_and_compare.py 
            $<TARGET_FILE:test_forward> 
            ${CMAKE_CURRENT_SOURCE_DIR}/forward_test/test_forward.py
)