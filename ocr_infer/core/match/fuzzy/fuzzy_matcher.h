#ifndef OCR_INFER_CORE_MATCH_FUZZY_FUZZY_MATCHER_H_
#define OCR_INFER_CORE_MATCH_FUZZY_FUZZY_MATCHER_H_

#include <string>
#include <vector>

class FuzzyMatcher {
 public:
  std::vector<std::wstring> Parse(const std::wstring &text);
};

#endif  // OCR_INFER_CORE_MATCH_FUZZY_FUZZY_MATCHER_H_
