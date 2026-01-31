/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#include <iostream>
#include <string>

#include "tokenization.h"

int main() {
  // Testing tokenization.
  std::string tok = "aaabdaaabac";
  std::string tok2 = "ababababa";

  auto byteSeq = createByteSequenceFromString(tok);
  auto bs2 = createByteSequenceFromString(tok2);
  TrainingText txt = {byteSeq, bs2};
  train(txt, 261);
}
