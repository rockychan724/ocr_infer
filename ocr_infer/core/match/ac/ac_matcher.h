#ifndef OCR_INFER_CORE_MATCH_AC_AC_MATCHER_H_
#define OCR_INFER_CORE_MATCH_AC_AC_MATCHER_H_

#include "ocr_infer/core/match/matcher_base.h"

class AcMatcher : public MatcherBase {
 public:
  KeywordId Parse(const std::vector<std::string> &text);
};

#endif  // OCR_INFER_CORE_MATCH_AC_AC_MATCHER_H_
