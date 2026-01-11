add_library(Tensor 
    tensor.cc
    tensor.h
    grad_mode.cc
    grad_mode.h
    private/tensor_impl.cc
    private/tensor_impl.h
    private/operation.h
    private/operation.cc
    private/shape.cc
    private/shape.h
    private/kernels.cc
    private/kernels.h
    private/kernels.ipp
)

target_include_directories(Tensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)


add_subdirectory(unittest)
