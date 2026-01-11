# MicroTensor – Small DL Library in C++

Inspired by some of the things I worked on and saw colleagues work on at Huawei.

No external dependencies.

### Goals

1. Deep learning engine in C++, which should be able to implement a basic GPT and generate text
2. Differential testing using PyTorch
3. Python bindings
4. CPU acceleration
5. GPU acceleration

### Testing

The tensor library (and eventually, much of this deep learning engine) is differentially tested against PyTorch. Look at `src/tensor/unittest` for examples.

Notes:

- Tensors are immutable if the goal is to perform backpropagation
- Not very efficient at the moment; virtual function calls, pointer chasing, no parallelism, etc.
  - Even element-wise operations are not efficient, iterating the elements of a tensor is not clean at the moment
- Most abstractions are modelled off of PyTorch
