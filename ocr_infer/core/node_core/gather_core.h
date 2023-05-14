#ifndef OCR_INFER_CORE_NODE_CORE_GATHER_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_GATHER_CORE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/node_core/core_base.h"

class GatherCore : public NodeCoreBase<RecOutput, OcrOutput> {
 public:
  GatherCore(const std::unordered_map<std::string, std::string> &config);

  std::shared_ptr<OcrOutput> Process(
      const std::shared_ptr<RecOutput> &in) override;

 private:
  std::vector<std::string> buffer_text_;
  std::vector<cv::RotatedRect> buffer_boxes_;
};

#endif  // OCR_INFER_CORE_NODE_CORE_GATHER_CORE_H_
