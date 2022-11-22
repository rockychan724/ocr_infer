#ifndef OCR_INFER_CORE_NODE_CORE_DETECT_POSTPROCESSING_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_DETECT_POSTPROCESSING_CORE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/model/detect/db_pp.h"
#include "ocr_infer/core/node_core/core_base.h"

class DetectPostProcessingCore : public NodeCoreBase<DetOutput, DetBox> {
 public:
  DetectPostProcessingCore(const std::unordered_map<std::string, std::string> &config);

  std::shared_ptr<DetBox> Process(const std::shared_ptr<DetOutput> &in) override;

 private:
  std::unique_ptr<DbPostprocessing> det_pp_;
};

#endif  // OCR_INFER_CORE_NODE_CORE_DETECT_POSTPROCESSING_CORE_H_
