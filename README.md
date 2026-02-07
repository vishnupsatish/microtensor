# MicroTensor – DL Library in C++

Supports training and inference. No external dependencies.

### Features

- Fully-featured tensor library with support for data sharing
  - Supports strided views, memory offset, etc.
  - Some operations are parallelized using OpenMP, such as certain elementwise operations and batched matmul
- Automatic differentiation engine
- Differential testing support (against PyTorch) for tensor library and gradient calculation
- Several built-in modules, such as linear, MLP, layernorm
  - [src/module/layers.cc](src/module/layers.cc)
- Stochastic gradient descent and AdamW optimizers
  - [src/module/optimizer.cc](src/module/optimizer.cc)
- Multi-headed self-attention, transformer, and GPT-2 implementation
  - [src/module/transformer.cc](src/module/transformer.cc)
  - [src/models/gpt2/train.cc](src/models/gpt2/train.cc)
- Byte-pair encoding implementation, supporting both training and tokenization
  - [src/util/tokenization.cc](src/util/tokenization.cc)
  - Several efficient data structures are used to ensure that token merges are performed in O(k log n) time
    - k: number of occurrences of the pair
    - n: total number of unique token pairs

### Future Goals

- Weight saving and resumable training, so inference can be separate from training
- Better way to specify hyperparameters and module initialization
- Weight tying
- Allow gradients requiring gradients (is not supported at the moment due to a circular reference issue)
- CPU acceleration by manually managing a thread pool (currently done using OpenMP)
- Python bindings
- GPU acceleration

### Build

Builds in C++ 20 using Apple Clang and G++. Mainly tested on MacOS.

1. Create a directory `build` in the root
2. Run `cmake ../src` to generate build files, `cmake ../src -DCMAKE_BUILD_TYPE=Release` ensures it will be built with optimizations enabled (highly recommended)
3. Run `make` to build the project
4. To train GPT-2, run `./models/gpt2_train`. Send the signal `SIGINT` (usually ctrl-c) once to perform inference.

### Testing

The tensor library is differentially tested against PyTorch. Look at [src/tensor/unittest](src/tensor/unittest) for examples. To run tests, run `ctest -V`. You may need to create a virtual environment and install dependencies from `requirements.txt`. If, after building within the virtual environment, there are still dependency issues, clear the CMake cache by running `rm CMakeCache.txt` and rebuilding.

### Notes

Important optimizations (with approximate speedups per GPT-2 training step, in order):

- Parallelizing batch dimensions of matmul (7x speedup)
- Making `a` contiguous and `b` transpose-contiguous (in memory) when performing `ab` in `matmulBatched` (2x speedup)
- Parallelizing the `makeContinguous` operation (~18% speedup)
- Moved making contiguous to `matmul` operation, rather than in `matmulBatched` kernel (so it is done before broadcasting, so we avoid allocating memory when we have something like a strided view) (~5-8% speedup)

### References

- PyTorch documentation (for tensor operations and several abstractions)
- https://jalammar.github.io/illustrated-gpt2/
- https://en.wikipedia.org/wiki/Byte-pair_encoding
- Gemini, for explaining several deep learning concepts to me and for writing a lot of the testing infrastructure and tests themselves
- https://karpathy.ai/zero-to-hero.html: Karpathy's deep learning series

### Where did I use AI?

I limited my usage of AI to write the code. It wrote some of the differential tests and the testing infrastructure scripts, as well as some helpers (such as `TensorImpl::print`). I also got it to help me when I faced build issues with CMake. In general, all of the core logic is written by me.
