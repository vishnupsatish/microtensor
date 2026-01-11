/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#pragma once

#include <memory>
#include <vector>

#include "tensor.h"

class Module {
  // Invariant: every module is responsible for managing its own submodules (the
  // simplest way are members that are unique_ptrs).
  std::vector<Module*> m_subModules;
  std::vector<Tensor> m_parameters;

 protected:
  void insertSubModule(Module* subModule);
  void insertParameter(Tensor param);

 public:
  std::vector<Tensor> getParameters();

  // Note the lack of a `forward` method. This is due to the variadic nature of
  // inputs/outputs. This means that polymorphic usage of Module (i.e., using
  // Module* or Module&) will not work.
};
