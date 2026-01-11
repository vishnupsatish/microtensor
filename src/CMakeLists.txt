cmake_minimum_required(VERSION 3.10)
project(MicroTensor)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Main main.cc)
target_link_libraries(Main Tensor Module util)

add_subdirectory(tensor)
add_subdirectory(module)
add_subdirectory(util)
add_subdirectory(models)
