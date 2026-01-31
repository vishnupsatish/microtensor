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

  auto byteSeq = createByteSequenceFromString(tok);
  TrainingText txt = {byteSeq};
  train(txt, 260);
}
