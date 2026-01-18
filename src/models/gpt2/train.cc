/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#include <atomic>
#include <csignal>
#include <fstream>
#include <iostream>

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

int main() {
  std::signal(SIGINT, signalHandler);
  auto trainData = readFile("text.txt");

  GPT2 model{256, 256, 128};
  auto opt = AdamW{model.getParameters(), 5e-4, 0.9, 0.999, 1e-8, 0.01};
  std::cout << "Training...\n";
  for (int epoch = 1; epoch <= 500; ++epoch) {
    if (stopTraining) break;

    auto [inp, target] = getBatch(trainData, 4, 256);
    auto gptRes = model.forward(inp);
    auto loss = crossEntropy(gptRes, target);
    std::cout << "Current epoch " << epoch << '\n';
    std::cout << "Current loss: ";
    loss.dumpTensor(std::cout);
    loss.backward();
    opt.step();
  }

  std::cout << "Inference...\n";
  {
    NoGrad guard;
    std::string prompt = "The ";
    std::vector<float> context;
    for (uint8_t c : prompt) {
      context.push_back(static_cast<float>(c));
    }

    int tokensToGenerate = 100;
    for (int i = 0; i < tokensToGenerate; ++i) {
      if (context.size() > 256) {
        size_t overflow = context.size() - 256;
        context.erase(context.begin(), context.begin() + overflow);
      }
      Tensor input{Shape{1, context.size()}, context};

      // Output: (1, Seq, 256)
      Tensor logits = model.forward(input);

      // We only care about the prediction for the very last position
      int seqLen = context.size();
      Tensor lastLogits = logits.slice({0, seqLen - 1, 0}, {1, 1, 256});

      // Get index of the highest score.
      int nextTokenId = lastLogits.argmax().item();
      char nextChar = static_cast<char>(nextTokenId);
      std::cout << nextChar << std::flush;  // Flush to see output in real-time
      context.push_back(static_cast<float>(nextTokenId));
    }
  }
}
