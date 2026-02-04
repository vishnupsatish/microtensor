/**
 *    Author:  Vishnu Satish
 *    Created: Jan 31, 2026
 **/

#pragma once

#include <random>
#include <string>
#include <vector>

#include "gpt2.h"
#include "tensor.h"
#include "tokenization.h"

std::string readFileToString(const std::string& path);

std::pair<Tensor, Tensor> getBatch(const std::vector<int>& rawData,
                                   int batchSize, int seqLen);

std::vector<std::string> preTokenizeSimple(const std::string& text);

Text getPretokenizedBytes(const std::string& text);

std::vector<float> getNextTokenLogits(std::vector<float> context, GPT2& model);

int sampleTopK(int k, const std::vector<float>& logits, std::mt19937& random);

int sampleTopP(float p, const std::vector<float>& logits, std::mt19937& random);
