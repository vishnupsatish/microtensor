/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#include "gpt2.h"
#include "grad_mode.h"
#include "loss_function.h"
#include "optimizer.h"

std::atomic<bool> stopTraining(false);

void signalHandler(int signum) {
  if (signum == SIGINT) {
    if (stopTraining) {
      std::cout << "\nSecond interrupt received. Force exiting...\n";
      std::exit(signum);
    }
    std::cout << "\nInterrupt received. Stopping training and jumping to "
                 "inference...\n";
    stopTraining = true;
  }
}

std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file");
  }

  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  file.read(reinterpret_cast<char*>(buffer.data()), size);

  if (!file) {
    throw std::runtime_error("Failed to read entire file");
  }

  return buffer;
}

// Returns a pair: {InputTensor, TargetTensor}
std::pair<Tensor, Tensor> getBatch(const std::vector<uint8_t>& rawData,
                                   int batchSize, int seqLen) {
  std::vector<float> inputBatch;
  std::vector<float> targetBatch;
  inputBatch.reserve(batchSize * seqLen);
  targetBatch.reserve(batchSize * seqLen);

  for (int b = 0; b < batchSize; ++b) {
    int maxIdx = rawData.size() - seqLen - 1;
    int randomIdx = rand() % maxIdx;
    for (int t = 0; t < seqLen; ++t) {
      inputBatch.push_back(static_cast<float>(rawData[randomIdx + t]));
      targetBatch.push_back(static_cast<float>(rawData[randomIdx + t + 1]));
    }
  }

  Tensor inputs{Shape{(size_t)batchSize, (size_t)seqLen}, inputBatch};
  Tensor targets{Shape{(size_t)batchSize, (size_t)seqLen}, targetBatch};

  return {inputs, targets};
}

const int maxSequenceLength = 128;
const int embeddingSize = 384;

int main() {
  std::signal(SIGINT, signalHandler);
  auto trainData = readFile("text.txt");

  GPT2 model{256, maxSequenceLength, embeddingSize};
  auto opt = AdamW{model.getParameters(), 3e-4, 0.9, 0.999, 1e-8, 0.01};
  std::cout << "Training...\n";
  int accumulationSteps = 4;
  for (int epoch = 1; epoch <= 2000; ++epoch) {
    if (stopTraining) break;
    std::cout << "Starting Global Step " << epoch << " (accumulating "
              << accumulationSteps << " batches)\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    opt.zeroGrad();
    float totalLoss = 0;

    for (int i = 0; i < accumulationSteps; ++i) {
      auto [inp, target] = getBatch(trainData, 12, maxSequenceLength);
      auto gptRes = model.forward(inp);
      auto loss = crossEntropy(gptRes, target);
      totalLoss += loss.item();

      // Scale loss for gradient accumulation so the step is averaged
      auto scaledLoss = loss / (float)accumulationSteps;
      scaledLoss.backward();
      std::cout << "  Accumulation batch " << i + 1 << "/" << accumulationSteps
                << " | Batch Loss: " << loss.item() << std::endl;
    }

    std::cout << "  Average Step Loss: " << (totalLoss / accumulationSteps)
              << '\n';
    std::cout << "  Optimizing parameters...\n";
    opt.step();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "  Global step took " << elapsed_time_ms
              << " milliseconds\n\n";
  }

  std::cout << "Inference...\n";
  {
    NoGrad guard;
    std::string prompt = "What is the world definition please tell me ";
    std::vector<float> context;
    for (uint8_t c : prompt) {
      context.push_back(static_cast<float>(c));
    }

    int tokensToGenerate = 100;
    for (int i = 0; i < tokensToGenerate; ++i) {
      if (context.size() > maxSequenceLength) {
        size_t overflow = context.size() - maxSequenceLength;
        context.erase(context.begin(), context.begin() + overflow);
      }
      Tensor input{Shape{1, context.size()}, context};

      // Output: (1, Seq, 256)
      Tensor logits = model.forward(input);

      // We only care about the prediction for the very last position
      int seqLen = context.size();
      Tensor lastLogits =
          logits.slice({0, seqLen - 1, 0}, {1, 1, embeddingSize});

      // Get index of the highest score using top-k sampling (k=10).
      std::vector<float> logitsVec = lastLogits.data();
      std::vector<int> indices(logitsVec.size());
      std::iota(indices.begin(), indices.end(), 0);

      // Sort indices based on logits
      int k = 10;
      std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                        [&logitsVec](int i1, int i2) {
                          return logitsVec[i1] > logitsVec[i2];
                        });

      // Get top-k logits and apply softmax
      std::vector<float> topKLogits;
      for (int j = 0; j < k; ++j) {
        topKLogits.push_back(logitsVec[indices[j]]);
      }

      // Softmax on top-k
      float maxLogit = *std::max_element(topKLogits.begin(), topKLogits.end());
      float sumExp = 0;
      for (float& l : topKLogits) {
        l = std::exp(l - maxLogit);
        sumExp += l;
      }
      for (float& l : topKLogits) {
        l /= sumExp;
      }

      // Sample from top-k
      static std::mt19937 gen(42);
      std::discrete_distribution<> d(topKLogits.begin(), topKLogits.end());
      int nextTokenId = indices[d(gen)];

      char nextChar = static_cast<char>(nextTokenId);
      std::cout << nextChar << std::flush;
      context.push_back(static_cast<float>(nextTokenId));
    }
  }
}
