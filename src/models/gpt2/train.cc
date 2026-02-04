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

const int maxSequenceLength = 256;
const int embeddingSize = 192;
const int vocabSize = 4096;

const int batchesForOptimizerUpdate = 32;
const int parallelBatches = 16;

int main() {
  std::signal(SIGINT, signalHandler);

  std::cout << "Pre-tokenizing...\n";
  std::string trainData = readFileToString("train.txt");
  Text byteSeq = getPretokenizedBytes(trainData);
  Tokenization bpe = trainBPE(byteSeq, vocabSize);
  std::cout << "Training tokenizer...\n";
  std::vector<int> tokenizedTrain = tokenizeBPE(bpe, byteSeq);

  std::string valData = readFileToString("validation.txt");
  Text valByteSeq = getPretokenizedBytes(valData);
  std::vector<int> tokenizedVal = tokenizeBPE(bpe, valByteSeq);

  GPT2 model{vocabSize, maxSequenceLength, embeddingSize};
  model.setMode(Mode::Train);
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

    if (epoch % 250 == 0) {
      NoGrad guard;
      model.setMode(Mode::Eval);
      std::cout << "Running model on validation data set...\n";
      auto [inp, target] =
          getBatch(tokenizedVal, parallelBatches, maxSequenceLength);
      auto valRes = model.forward(inp);
      auto loss = crossEntropy(valRes, target);
      std::cout << "Loss: " << loss.item() << '\n';
      model.setMode(Mode::Train);
    }
  }

  std::cout << "Inference...\n";
  {
    NoGrad guard;
    model.setMode(Mode::Eval);
    std::string prompt =
        "In the Bennet household, gossip spread like wildfire ";
    Text byteSeqPrompt = getPretokenizedBytes(prompt);
    std::vector<int> tokenizedPrompt = tokenizeBPE(bpe, byteSeqPrompt);

    std::vector<float> context;
    for (int t : tokenizedPrompt) {
      context.push_back(static_cast<float>(t));
    }

    int tokensToGenerate = 300;
    for (int i = 0; i < tokensToGenerate; ++i) {
      if (context.size() > maxSequenceLength) {
        size_t overflow = context.size() - maxSequenceLength;
        context.erase(context.begin(), context.begin() + overflow);
      }
      auto logitsVec = getNextTokenLogits(context, model);

      int nextTokenId = sampleTopP(40, logitsVec, RNG::get());

      ByteSequence byteSeq = bpe.vocab[nextTokenId];
      std::string outputStr = createStringFromByteSequence(byteSeq);
      std::cout << outputStr << std::flush;

      context.push_back(static_cast<float>(nextTokenId));
    }
  }
}
