/**
 *    Author:  Vishnu Satish
 *    Created: Jan 31, 2026
 **/

#pragma once

#include <string>
#include <vector>

#include "tensor.h"

std::string readFileToString(const std::string& path);

std::pair<Tensor, Tensor> getBatch(const std::vector<int>& rawData,
                                   int batchSize, int seqLen);

std::vector<std::string> preTokenizeSimple(const std::string& text);
