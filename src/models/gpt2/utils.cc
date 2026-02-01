/**
 *    Author:  Vishnu Satish
 *    Created: Jan 31, 2026
 **/

#include "utils.h"

#include <fstream>
#include <random>

#include "random_gen.h"

std::string readFileToString(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file");
  }

  // Seek to end to get file size
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (size < 0) {
    throw std::runtime_error("Failed to determine file size");
  }
  if (size == 0) {
    return {};
  }

  std::string buffer(size, '\0');
  file.read(buffer.data(), size);

  if (!file) {
    throw std::runtime_error("Failed to read entire file");
  }

  return buffer;
}

// Returns a pair: {InputTensor, TargetTensor}
std::pair<Tensor, Tensor> getBatch(const std::vector<int>& rawData,
                                   int batchSize, int seqLen) {
  std::vector<float> inputBatch;
  std::vector<float> targetBatch;
  inputBatch.reserve(batchSize * seqLen);
  targetBatch.reserve(batchSize * seqLen);

  int maxIdx = static_cast<int>(rawData.size()) - seqLen - 1;
  auto dist = std::uniform_int_distribution<int>{0, maxIdx - 1};

  for (int b = 0; b < batchSize; ++b) {
    int randomIdx = dist(RNG::get());
    for (int t = 0; t < seqLen; ++t) {
      inputBatch.push_back(static_cast<float>(rawData[randomIdx + t]));
      targetBatch.push_back(static_cast<float>(rawData[randomIdx + t + 1]));
    }
  }

  Tensor inputs{Shape{(size_t)batchSize, (size_t)seqLen}, inputBatch};
  Tensor targets{Shape{(size_t)batchSize, (size_t)seqLen}, targetBatch};

  return {inputs, targets};
}

std::vector<std::string> preTokenizeSimple(const std::string& text) {
  std::vector<std::string> segments;
  std::string current;

  for (char c : text) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (!current.empty()) {
        segments.push_back(std::move(current));
      }
      current = c;
    } else if (std::ispunct(static_cast<unsigned char>(c))) {
      if (!current.empty()) {
        segments.push_back(std::move(current));
      }
      segments.push_back(std::string{c});
      current.clear();
    } else {
      current += c;
    }
  }

  if (!current.empty()) {
    segments.push_back(current);
  }

  return segments;
}
