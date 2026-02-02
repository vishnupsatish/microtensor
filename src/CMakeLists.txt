cmake_minimum_required(VERSION 3.10)
project(MicroTensor)
enable_testing()

# Respect _ROOT env vars (allows us to find OpenMP)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -funroll-loops -flto")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
endif()

add_executable(Main main.cc)
target_link_libraries(Main Tensor Module util)

add_subdirectory(tensor)
add_subdirectory(module)
add_subdirectory(util)
add_subdirectory(models)

# Copy to the root build directory (hack...)
configure_file(models/gpt2/text.txt ${CMAKE_BINARY_DIR}/text.txt COPYONLY)
configure_file(models/gpt2/smallerWikipedia.txt ${CMAKE_BINARY_DIR}/smallerWikipedia.txt COPYONLY)
