/**
 *    Author:  Vishnu Satish
 *    Created: Jan 17, 2026
 **/

#pragma once

#include <map>
#include <vector>

// Want to work with unsigned, since tokens should ideally start from 0 and
// not -128. Char can be signed in certain architectures/compilers, so we cast
// char -> uint8_t.
using ByteSequence = std::vector<uint8_t>;

// Indicates merge boundaries.
using Text = std::vector<ByteSequence>;

// Maps token -> sequence of bytes.
using Vocabulary = std::vector<ByteSequence>;

// Maps list of bytes -> token.
using TokenMap = std::map<ByteSequence, int>;

// Ordered merge rules (tokens).
using Merges = std::vector<std::pair<int, int>>;

struct Tokenization {
  Vocabulary vocab;
  TokenMap tokenMap;
  Merges merges;
};

// Helpers
ByteSequence createByteSequenceFromString(const std::string& input);
std::string createStringFromByteSequence(const ByteSequence& seq);

void emitTokenization(const Tokenization& bpe, std::ostream& os);
Tokenization parseTokenization(std::istream& is);

// Core BPE functions. For `train` and `tokenize`, the text must be
// pre-tokenized, i.e., split up based on merge boundaries.
Tokenization trainBPE(Text text, int vocabSize);
std::vector<int> tokenizeBPE(const Tokenization& bpe, Text input);

ByteSequence convertTokenizationToBytes(const Tokenization& bpe,
                                        std::vector<int> tokenization);
