#ifndef OCR_INFER_CORE_MATCH_MATCHER_BASE_H_
#define OCR_INFER_CORE_MATCH_MATCHER_BASE_H_

#include <string>
#include <vector>

typedef int KeywordId;

class MatcherBase {
 public:
  virtual KeywordId Parse(const std::vector<std::string> &text) = 0;

 private:
  
};

#endif  // OCR_INFER_CORE_MATCH_MATCHER_BASE_H_
