# Module abstraction and some common layers/modules. Also includes optimizers.
add_library(Module 
    module.h
    module.cc
    optimizer.h
    optimizer.cc
    layers.h
    layers.cc
    loss_function.h
    loss_function.cc
    transformer.cc
    transformer.h
)

target_link_libraries(Module PUBLIC Tensor util)

target_include_directories(Module
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(unittest)
