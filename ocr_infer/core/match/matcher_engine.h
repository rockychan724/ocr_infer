#ifndef OCR_INFER_CORE_MATCH_MATCHER_ENGINE_H_
#define OCR_INFER_CORE_MATCH_MATCHER_ENGINE_H_

#include <memory>
#include <unordered_map>

#include "ocr_infer/core/match/matcher_base.h"
#include "ocr_infer/core/match/rule_tree.h"

class MatcherEngine {
 public:
  MatcherEngine(const std::string &keyword_dir);

  KeywordId Match(const std::vector<std::string> &texts);

  int AddKeyword(KeywordId id, const std::string &keyword, int flag);

  int DeleteKeyword(KeywordId id, int flag);

 private:
  std::unique_ptr<MatcherBase> matcher_;
  std::unique_ptr<RuleTree> rule_tree_;

  std::unordered_map<KeywordId, std::vector<std::wstring>> id_to_keyword_;
  std::unordered_map<std::wstring, KeywordId> keyword_to_id_;

  void ReadKeyword(const std::string &keyword_dir);
};

#endif  // OCR_INFER_CORE_MATCH_MATCHER_ENGINE_H_
