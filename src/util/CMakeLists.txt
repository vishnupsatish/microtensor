add_library(util 
    random_gen.cc
    random_gen.h
    pair_hash.h
    tokenization.cc
    tokenization.h
)

target_include_directories(util
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
