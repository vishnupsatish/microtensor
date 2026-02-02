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
- Multi-headed self-attention, transformer, and GPT2 implementation
  - [src/module/transformer.cc](src/module/transformer.cc)
  - [src/models/gpt2/train.cc](src/models/gpt2/train.cc)
- Byte-pair encoding implementation optimized for pure algorithmic efficiency (as opposed to practical efficiency)
  - [src/util/tokenization.cc](src/util/tokenization.cc)
  - Did not consult any references (other than the definition of BPE) when developing this

### Future Goals

- Weight saving and resumable training
- Better way to specify hyperparameters and module initialization
- Weight tying
- Allow gradients requiring gradients (is not supported at the moment due to a circular reference issue)
- CPU acceleration by manually managing a thread pool (currently done using OpenMP)
- Python bindings
- GPU acceleration

### Build

Probably requires C++ 20 (need to write more).

### Testing

The tensor library (and eventually, much of this deep learning engine) is differentially tested against PyTorch. Look at `src/tensor/unittest` for examples.

### Notes

Important optimizations (with approximate speedups per GPT2 training step, in order):

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
