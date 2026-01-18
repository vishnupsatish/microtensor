add_executable(xor xor.cc)
target_link_libraries(xor Tensor Module util)

add_library(GPT2Lib gpt2/gpt2.cc)
target_link_libraries(GPT2Lib Tensor Module)
target_include_directories(GPT2Lib PUBLIC gpt2)

add_executable(gpt2_train gpt2/train.cc)
target_link_libraries(gpt2_train GPT2Lib Tensor Module util)

add_executable(gpt2_inference gpt2/inference.cc)
target_link_libraries(gpt2_inference GPT2Lib Tensor Module util)

configure_file(gpt2/text.txt ${CMAKE_CURRENT_BINARY_DIR}/text.txt COPYONLY)
