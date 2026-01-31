/**
 *    Author:  Vishnu Satish
 *    Created: Jan 17, 2026
 **/

#include "tokenization.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <unordered_set>

#include "pair_hash.h"

ByteSequence createByteSequenceFromString(const std::string& input) {
  ByteSequence ret;
  for (char c : input) {
    ret.push_back(static_cast<uint8_t>(c));
  }
  return ret;
}

std::string createStringFromByteSequence(const ByteSequence& seq) {
  std::string s;
  for (uint8_t byte : seq) {
    s.push_back(static_cast<char>(byte));
  }
  return s;
}

////////////////////////////////////////////////////////////////////////////////

namespace {

// Note: this is optimized for pure algorithmic efficiency, not practical
// efficiency.
class BPETokenizationBuilder {
  TrainingText m_text;

  int m_vocabSize;
  Vocabulary m_vocab;
  TokenMap m_tokenMap;
  Merges m_merges;

  using TokenPair = std::pair<int, int>;

  // Represents starting index of token within that word, and the token.
  struct TokenLocation {
    // Which word are we in? From the perspective of `words`.
    int wordIndex;
    // Index within the word. This is actually the character-based index,
    // meaning if characters get merged, it is NOT reflected in this value.
    int index;
    int token;
  };

  // Linked list of tokens that make up a single word.
  // Note: we are using "word" as a way of representing sequences of tokens that
  // can be merged within. It does not actually have to be a "word" by the
  // English definition.
  using WordList = std::list<TokenLocation>;

  std::vector<WordList> m_words;

  // Represents a unique identifier to a specific token.
  using TokenID = WordList::iterator;

  // When we map token pairs to a sequence of their locations, we want to order
  // the sequence by index of their locations (within the same word). This is
  // because of potentially overlapping occurrences, and merges must occur
  // left-to-right within the same word.
  struct TokenIDCmp {
    bool operator()(TokenID a, TokenID b) const {
      if (a->wordIndex == b->wordIndex) {
        return a->index < b->index;
      }
      return a->wordIndex < b->wordIndex;
    }
  };

  // Maps pairs of the tokens to the token identifiers (i.e., list iterators) at
  // which the pairs start.
  std::unordered_map<TokenPair, std::set<TokenID, TokenIDCmp>, PairHash>
      m_tokenPairIndex;

  // Maps integer to the list of token pairs that occur that many times.
  std::map<int, std::unordered_set<TokenPair, PairHash>, std::greater<>>
      m_numTokenPair;

  ////////////////////////////////////////////////////
  // Debugging helpers
  void printByteSeq(const ByteSequence& seq) {
    std::cout << createStringFromByteSequence(seq);
  }

  void printDebug() {
    // First, print each list of tokens
    std::cout << "Printing words:\n";
    for (int i = 0; i < m_words.size(); ++i) {
      auto& word = m_words[i];
      for (const TokenLocation& loc : word) {
        std::cout << loc.token << ':';
        printByteSeq(m_vocab[loc.token]);
        std::cout << ' ';
      }
      std::cout << '\n';
    }

    // Print `m_tokenPairIndex`.
    std::cout << "Printing token pairs and their occurrences:\n";
    for (auto& [pair, tokenIds] : m_tokenPairIndex) {
      std::cout << "Pair " << pair.first << ' ' << pair.second << '\n';
      std::cout << "Occurrences\n";
      for (auto id : tokenIds) {
        std::cout << id->wordIndex << ' ' << id->index << '\n';
      }
    }
  }

  ////////////////////////////////////////////////////

  size_t tokenSize(int token) { return m_vocab[token].size(); }

  WordList& getListAssociatedToNode(TokenID tokenId) {
    return m_words[tokenId->wordIndex];
  }

  std::optional<TokenID> getPreviousToken(TokenID tokenId) {
    auto& lst = getListAssociatedToNode(tokenId);
    assert(tokenId != lst.end());
    if (tokenId == lst.begin()) {
      return {};
    }
    return std::prev(tokenId);
  }

  std::optional<TokenID> getNextToken(TokenID tokenId) {
    auto& lst = getListAssociatedToNode(tokenId);
    assert(tokenId != lst.end());
    if (std::next(tokenId) == lst.end()) {
      return {};
    }
    return std::next(tokenId);
  }

  // Given a pair and a starting token, remove it from `m_tokenPairIndex` and
  // also update the respective entry in `m_numTokenPair`.
  void eraseStartingTokenOfPair(TokenPair tokenPair, TokenID tokenId) {
    int occurrencesOfPair = m_tokenPairIndex[tokenPair].size();
    assert(m_numTokenPair[occurrencesOfPair].contains(tokenPair));
    m_numTokenPair[occurrencesOfPair].erase(tokenPair);
    if (m_numTokenPair[occurrencesOfPair].size() == 0) {
      m_numTokenPair.erase(occurrencesOfPair);
    }

    assert(m_tokenPairIndex[tokenPair].contains(tokenId));
    m_tokenPairIndex[tokenPair].erase(tokenId);
    if (occurrencesOfPair > 1) {
      m_numTokenPair[occurrencesOfPair - 1].insert(tokenPair);
    } else {
      m_tokenPairIndex.erase(tokenPair);
    }
  }

  void insertStartingTokenOfPair(TokenPair tokenPair, TokenID tokenId) {
    int occurrencesOfPair = m_tokenPairIndex[tokenPair].size();
    if (occurrencesOfPair > 0) {
      assert(m_numTokenPair[occurrencesOfPair].contains(tokenPair));
      m_numTokenPair[occurrencesOfPair].erase(tokenPair);
      if (m_numTokenPair[occurrencesOfPair].size() == 0) {
        m_numTokenPair.erase(occurrencesOfPair);
      }
    }
    m_numTokenPair[occurrencesOfPair + 1].insert(tokenPair);
    m_tokenPairIndex[tokenPair].insert(tokenId);
  }

  void initializeVocab() {
    assert(std::numeric_limits<uint8_t>::min() == 0);
    assert(std::numeric_limits<uint8_t>::max() == 255);
    for (int i = 0; i < 256; ++i) {
      uint8_t bytes = static_cast<uint8_t>(i);
      m_vocab[i] = {bytes};
      m_tokenMap[{bytes}] = i;
    }
  }

  void createInitialPairs() {
    // Create linked lists asociated with each word and populate the pairs of
    // tokens -> token identifier map.
    for (int i = 0; i < m_text.size(); ++i) {
      // Create empty list for this word.
      m_words.push_back({});
      auto& byteSeq = m_text[i];
      for (int j = 0; j < byteSeq.size(); ++j) {
        TokenLocation loc{i, j, m_tokenMap[{byteSeq[j]}]};
        auto currentTokenId =
            m_words[i].insert(m_words[i].end(), std::move(loc));
        if (j + 1 != byteSeq.size()) {
          m_tokenPairIndex[{byteSeq[j], byteSeq[j + 1]}].insert(currentTokenId);
        }
      }
    }
    for (const auto& [tokenPair, indices] : m_tokenPairIndex) {
      m_numTokenPair[indices.size()].insert(tokenPair);
    }
  }

  void mergeTokens(TokenPair pair, int newToken) {
    auto& identifiers = m_tokenPairIndex[pair];
    int t1 = pair.first;
    int t2 = pair.second;
    for (auto it = identifiers.begin(); it != identifiers.end(); ++it) {
      TokenID firstToken = *it;
      WordList& lst = getListAssociatedToNode(firstToken);

      // tokenId is the start of a pair. Therefore, there is always a next token
      // in that word.
      assert(std::next(firstToken) != m_words[firstToken->wordIndex].end());
      TokenID secondToken = std::next(firstToken);

      // TODO: think about edge cases; single element linked list, maybe zero(?)
      // element linked lists, etc.

      // Update `m_tokenPairIndex` and `m_numTokenPair`. Note: in the case of
      // overlapping pairs (such as aaa where we are merging a,a) we will be
      // deleting from the current std::set that we are iterating. But, std::set
      // guarantees pointer validity even after deletion (as long as we don't
      // delete the current element), so this is OK.
      auto prevTokenOpt = getPreviousToken(firstToken);
      if (prevTokenOpt) {
        TokenID prevToken = *prevTokenOpt;
        TokenPair prevPair{prevToken->token, firstToken->token};
        // Remove this occurrence of `prevPair` from `m_tokenPairIndex`. It
        // should hold that if t1 == t2, prevToken's token number is NOT equal
        // to t1, because otherwise we would have replaced that pair instead
        // (since we are guaranteed to go left-to-right within the same word).
        assert(!(t1 == t2 && prevToken->token == t1));
        eraseStartingTokenOfPair(prevPair, prevToken);
      }

      auto nextTokenOpt = getNextToken(secondToken);
      if (nextTokenOpt) {
        TokenID nextToken = *nextTokenOpt;
        TokenPair nextPair{secondToken->token, nextToken->token};
        // `secondToken` is the startng token of `nextPair`.
        eraseStartingTokenOfPair(nextPair, secondToken);
      }

      // Make the first token represent the new, merged token, delete the second
      // token.
      firstToken->token = newToken;
      lst.erase(secondToken);

      // Insert the new previous and next pairs into m_numTokenPair.
      if (prevTokenOpt) {
        TokenID prevToken = *prevTokenOpt;
        TokenPair newPrevPair{prevToken->token, newToken};
        insertStartingTokenOfPair(newPrevPair, prevToken);
      }

      if (nextTokenOpt) {
        TokenID nextToken = *nextTokenOpt;
        TokenPair newNextPair{newToken, nextToken->token};
        // `firstToken` LL node represents the new token that was just created.
        insertStartingTokenOfPair(newNextPair, firstToken);
      }
    }

    int numOccurrences = m_tokenPairIndex[pair].size();
    // Could be zero in the case of overlapping pairs potentially? I need to
    // think about this.
    if (numOccurrences > 0) {
      m_numTokenPair[numOccurrences].erase(pair);
      if (m_numTokenPair[numOccurrences].empty()) {
        m_numTokenPair.erase(numOccurrences);
      }
    }
    m_tokenPairIndex.erase(pair);
  }

  void mergeMostCommonPair() {
    assert(!m_numTokenPair.empty());
    assert(!m_numTokenPair.begin()->second.empty());
    // Out of the most common pairs, pick the first that we see.
    auto mergePair = *m_numTokenPair.begin()->second.begin();
    // Update m_tokenPairIndex and m_numTokenPair correctly by performing
    // merges.
    const ByteSequence& t1Bytes = m_vocab[mergePair.first];
    const ByteSequence& t2Bytes = m_vocab[mergePair.second];

    m_merges.push_back(mergePair);

    // Create new token which is a concatenation of the pair.
    // The token -> vocab mapping could actually be a trie for better
    // algorithmic efficiency.
    ByteSequence newTokenBytes;
    newTokenBytes.reserve(t1Bytes.size() + t2Bytes.size());
    newTokenBytes.insert(newTokenBytes.end(), t1Bytes.begin(), t1Bytes.end());
    newTokenBytes.insert(newTokenBytes.end(), t2Bytes.begin(), t2Bytes.end());
    int newToken = m_vocab.size();
    m_vocab[newToken] = newTokenBytes;
    m_tokenMap[std::move(newTokenBytes)] = newToken;

    mergeTokens(mergePair, newToken);
  }

 public:
  BPETokenizationBuilder(TrainingText text, int vocabSize)
      : m_text{std::move(text)}, m_vocabSize{vocabSize} {}

  Tokenization train() {
    if (m_vocabSize < 256) {
      throw std::runtime_error{"Vocabulary size must be at least 256"};
    }
    initializeVocab();
    createInitialPairs();
    printDebug();
    while (m_vocab.size() < m_vocabSize && m_numTokenPair.size() >= 1) {
      mergeMostCommonPair();
      printDebug();
    }
    return {m_vocab, m_tokenMap, m_merges};
  }
};

}  // namespace

Tokenization train(TrainingText text, int vocabSize) {
  return BPETokenizationBuilder{text, vocabSize}.train();
}
