add_library(Models 
    xor.h
    xor.cc
)

target_link_libraries(Models PUBLIC Module Tensor util)
