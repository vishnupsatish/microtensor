/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>

#include "gpt2.h"
#include "grad_mode.h"
#include "loss_function.h"
#include "optimizer.h"
#include "random_gen.h"
#include "tokenization.h"
#include "utils.h"

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

const int maxSequenceLength = 128;
const int embeddingSize = 256;
const int vocabSize = 4096;

const int batchesForOptimizerUpdate = 48;
const int parallelBatches = 24;

int main() {
  std::signal(SIGINT, signalHandler);

  std::cout << "Pre-tokenizing...\n";
  // Thanks to
  // https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish
  // for the Wikipedia dataset (MIT License).
  std::string trainData = readFileToString("smallerWikipedia.txt");
  std::vector<std::string> preTokenized = preTokenizeSimple(trainData);
  Text byteSeq;
  std::transform(
      preTokenized.begin(), preTokenized.end(), std::back_inserter(byteSeq),
      [](const std::string& st) { return createByteSequenceFromString(st); });
  Tokenization bpe = trainBPE(byteSeq, vocabSize);
  std::cout << "Training tokenizer...\n";
  std::vector<int> tokenizedTrain = tokenizeBPE(bpe, byteSeq);

  GPT2 model{vocabSize, maxSequenceLength, embeddingSize};
  auto opt = AdamW{model.getParameters(), 3e-4f, 0.9, 0.999, 1e-8f, 0.01};
  std::cout << "Training...\n";
  assert(batchesForOptimizerUpdate % parallelBatches == 0);
  int accumulationSteps = batchesForOptimizerUpdate / parallelBatches;
  for (int epoch = 1; epoch <= 10000; ++epoch) {
    if (stopTraining) break;
    std::cout << "Starting Global Step " << epoch << " (accumulating "
              << accumulationSteps << " batches)\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    opt.zeroGrad();
    float totalLoss = 0;

    for (int i = 0; i < accumulationSteps; ++i) {
      auto [inp, target] =
          getBatch(tokenizedTrain, parallelBatches, maxSequenceLength);
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
    std::string prompt = "Cricket is a sport ";
    std::vector<std::string> preTokenizedPrompt = preTokenizeSimple(prompt);
    Text byteSeqPrompt;
    std::transform(
        preTokenizedPrompt.begin(), preTokenizedPrompt.end(),
        std::back_inserter(byteSeqPrompt),
        [](const std::string& st) { return createByteSequenceFromString(st); });
    std::vector<int> tokenizedPrompt = tokenizeBPE(bpe, byteSeqPrompt);

    std::vector<float> context;
    for (int t : tokenizedPrompt) {
      context.push_back(static_cast<float>(t));
    }

    int tokensToGenerate = 100;
    for (int i = 0; i < tokensToGenerate; ++i) {
      if (context.size() > maxSequenceLength) {
        size_t overflow = context.size() - maxSequenceLength;
        context.erase(context.begin(), context.begin() + overflow);
      }
      Tensor input{Shape{1, context.size()}, context};

      // Output: (1, Seq, vocabSize)
      Tensor logits = model.forward(input);

      // We only care about the prediction for the very last position
      int seqLen = context.size();
      Tensor lastLogits = logits.slice({0, seqLen - 1, 0}, {1, 1, vocabSize});

      // Get index of the highest score using top-k sampling.
      std::vector<float> logitsVec = lastLogits.data();
      std::vector<int> indices(logitsVec.size());
      std::iota(indices.begin(), indices.end(), 0);

      // Sort indices based on logits
      int k = 5;
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
      std::discrete_distribution<> d(topKLogits.begin(), topKLogits.end());
      int nextTokenId = indices[d(RNG::get())];

      ByteSequence byteSeq = bpe.vocab[nextTokenId];
      std::string outputStr = createStringFromByteSequence(byteSeq);
      std::cout << outputStr << std::flush;

      context.push_back(static_cast<float>(nextTokenId));
    }
  }
}
