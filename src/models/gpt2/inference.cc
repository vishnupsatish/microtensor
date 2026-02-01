/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#include <iostream>
#include <iterator>
#include <string>

#include "tokenization.h"

int main() {
  // Testing tokenization.
  std::vector<std::string> data = {"aaabdaaabac", "ababababa"};
  Text trainingText;
  std::transform(data.begin(), data.end(), std::back_inserter(trainingText),
                 [&](const std::string& str) {
                   return createByteSequenceFromString(str);
                 });

  auto tokenization = trainBPE(trainingText, 261);

  std::vector<std::string> tokenizeData = {"aaabdaaabac", "ababababa"};
  Text tokenizeText;
  std::transform(data.begin(), data.end(), std::back_inserter(tokenizeText),
                 [&](const std::string& str) {
                   return createByteSequenceFromString(str);
                 });

  auto tokens = tokenizeBPE(tokenization, tokenizeText);
}
