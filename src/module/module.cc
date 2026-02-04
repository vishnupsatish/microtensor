/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "module.h"

void Module::insertSubModule(Module* subModule) {
  m_subModules.push_back(subModule);
}

void Module::insertParameter(Tensor param) {
  m_parameters.push_back(std::move(param));
}

std::vector<Tensor> Module::getParameters() {
  std::vector<Tensor> ret = m_parameters;
  for (auto& mod : m_subModules) {
    auto modParams = mod->getParameters();
    ret.insert(ret.end(), std::make_move_iterator(modParams.begin()),
               std::make_move_iterator(modParams.end()));
  }
  return ret;
}

void Module::setMode(Mode mode) {
  m_mode = mode;
  for (auto& mod : m_subModules) {
    mod->setMode(mode);
  }
}

Mode Module::getMode() { return m_mode; }
