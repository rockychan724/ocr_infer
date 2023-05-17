#ifndef OCR_INFER_CORE_MATCH_MATCHER_ENGINE_H_
#define OCR_INFER_CORE_MATCH_MATCHER_ENGINE_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "ocr_infer/core/match/ac/ac_matcher.h"
#include "ocr_infer/core/match/rule_tree.h"

typedef int KeywordId;

class MatcherEngine {
 public:
  MatcherEngine(const std::string &keyword_dir);

  KeywordId Match(const std::vector<std::string> &texts);

  int AddKeyword(KeywordId id, const std::string &keyword, int flag);

  int DeleteKeyword(KeywordId id, int flag);

 private:
  std::unique_ptr<AcMatcher> matcher_;
  std::unique_ptr<RuleTree<int, std::vector<int>>> rule_tree_;

  std::unordered_map<KeywordId, std::vector<std::wstring>> ruleid_to_keyword_;
  std::unordered_map<std::wstring, int> keyword_to_wordid_;
  std::vector<int> invalid_rule_;

  void ReadKeyword(std::string keyword_dir);

  std::vector<int> CombBack(std::vector<int> &hit_word_vector, int m, int r);
};

#endif  // OCR_INFER_CORE_MATCH_MATCHER_ENGINE_H_
