#ifndef OCR_INFER_CORE_NODE_CORE_MATCH_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_MATCH_CORE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/node_core/core_base.h"

class MatchCore : public CoreBase<RecOutput, MatchOutput> {
 public:
  MatchCore(const std::unordered_map<std::string, std::string> &config);

  std::shared_ptr<MatchOutput> Process(const std::shared_ptr<RecOutput> &in) override;
};

#endif  // OCR_INFER_CORE_NODE_CORE_MATCH_CORE_H_
