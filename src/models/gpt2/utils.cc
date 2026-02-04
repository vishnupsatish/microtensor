/**
 *    Author:  Vishnu Satish
 *    Created: Jan 31, 2026
 **/

#include "utils.h"

#include <fstream>
#include <random>

#include "random_gen.h"
#include "tokenization.h"

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

Text getPretokenizedBytes(const std::string& text) {
  std::vector<std::string> preTokenized = preTokenizeSimple(text);
  Text byteSeq;
  std::transform(
      preTokenized.begin(), preTokenized.end(), std::back_inserter(byteSeq),
      [](const std::string& st) { return createByteSequenceFromString(st); });
  return byteSeq;
}

std::vector<float> getNextTokenLogits(std::vector<float> context, GPT2& model) {
  Tensor input{Shape{1, context.size()}, context};
  // Output: (1, Seq, vocabSize)
  Tensor logits = model.forward(input);
  // We only care about the prediction for the very last position
  int seqLen = context.size();
  // TODO: really need to do something about these `static_cast`s.
  size_t vocabSize = static_cast<size_t>(model.getVocabSize());
  Tensor lastLogits = logits.slice({0, seqLen - 1, 0}, {1, 1, vocabSize});
  return lastLogits.data();
}

std::vector<float> softmaxFromLogits(const std::vector<float>& logits) {
  std::vector<float> prob = logits;
  float maxLogit = *std::max_element(logits.begin(), logits.end());
  float sumExp = 0;
  for (float& l : prob) {
    l = std::exp(l - maxLogit);
    sumExp += l;
  }
  for (float& l : prob) {
    l /= sumExp;
  }
  return prob;
}

int sampleTopK(int k, const std::vector<float>& logits, std::mt19937& random) {
  if (logits.empty()) {
    throw std::runtime_error{"No logits provided"};
  }

  // Get index of the highest score using top-k sampling.
  std::vector<int> indices(logits.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Sort indices based on logits
  std::partial_sort(
      indices.begin(), indices.begin() + k, indices.end(),
      [&logits](int i1, int i2) { return logits[i1] > logits[i2]; });

  // Get top-k logits and apply softmax
  std::vector<float> topKLogits;
  topKLogits.reserve(k);
  for (int j = 0; j < k; ++j) {
    topKLogits.push_back(logits[indices[j]]);
  }

  auto topKProb = softmaxFromLogits(topKLogits);

  // Sample from top-k
  std::discrete_distribution<> d(topKProb.begin(), topKProb.end());
  return indices[d(random)];
}

int sampleTopP(float p, const std::vector<float>& logits,
               std::mt19937& random) {
  if (logits.empty()) {
    throw std::runtime_error{"No logits provided"};
  }

  std::vector<float> prob = softmaxFromLogits(logits);

  std::vector<int> indices(logits.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Since we sort based on the same condition in both, after sorting, `indices`
  // lines up with `prob`.
  std::sort(indices.begin(), indices.end(),
            [&prob](int i1, int i2) { return prob[i1] > prob[i2]; });
  std::sort(prob.begin(), prob.end(), std::greater<float>());

  float curProb = 0;
  // The first index that we ignore.
  int ignorePoint = prob.size();
  for (size_t i = 0; i < prob.size(); ++i) {
    curProb += prob[i];
    if (curProb >= p) {
      ignorePoint = i + 1;
      break;
    }
  }

  // Discrete distribution automatically normalizes the probabilities that are
  // passed in.
  std::discrete_distribution<> d(prob.begin(), prob.begin() + ignorePoint);
  return indices[d(random)];
}
