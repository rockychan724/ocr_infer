#ifndef OCR_INFER_CORE_NODE_CORE_MATCH_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_MATCH_CORE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/match/matcher_engine.h"
#include "ocr_infer/core/node_core/core_base.h"

class MatchCore : public NodeCoreBase<OcrOutput, MatchOutput> {
 public:
  MatchCore(const std::unordered_map<std::string, std::string> &config);

  std::shared_ptr<MatchOutput> Process(
      const std::shared_ptr<OcrOutput> &in) override;

 private:
  std::unique_ptr<MatcherEngine> matcher_engine_;
};

#endif  // OCR_INFER_CORE_NODE_CORE_MATCH_CORE_H_
