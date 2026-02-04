/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#pragma once

#include <memory>
#include <vector>

#include "tensor.h"

enum class Mode { Train, Eval };
class Module {
  Mode m_mode = Mode::Train;
  // Invariant: every module is responsible for managing its own submodules (the
  // simplest way are members that are unique_ptrs).
  // Note: this can go wrong easily. Since m_subModules stores direct pointers
  // and does not manage the actual submodules, if the submodule's memory
  // address is ever changed, this can lead to dangling pointers existing in
  // m_subModules. As an example, consider a concrete module that stores
  // submodules as direct class members, and calls insertSubModule(pointer to
  // direct class member modules). In case the concrete module were to get
  // moved, the locations of the submodules would change in memory. One solution
  // is to force each concrete module to store unique_ptrs to submodules so
  // their memory locations cannot change, but this is a poor solution. In the
  // future, m_subModules will manage the submodules (i.e., unique_ptr).
  std::vector<Module*> m_subModules;
  std::vector<Tensor> m_parameters;

 protected:
  void insertSubModule(Module* subModule);
  void insertParameter(Tensor param);

 public:
  std::vector<Tensor> getParameters();

  // No virtual destructor. Users should not use in polymorphic setting. See
  // comment below for another reason why.

  // Note the lack of a `forward` method. This is due to the variadic nature of
  // inputs/outputs. This means that polymorphic usage of Module (i.e., using
  // Module* or Module&) will not work.

  void setMode(Mode mode);
  Mode getMode();
};
