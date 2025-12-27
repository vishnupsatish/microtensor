add_library(Tensor 
    tensor.cc
    tensor.h
    private/tensor_impl.cc
    private/tensor_impl.h
    private/operation.h
    private/operation.cc
    private/shape.cc
    private/shape.h
)

add_subdirectory(unittest)
