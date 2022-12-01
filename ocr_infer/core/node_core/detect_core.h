#ifndef OCR_INFER_CORE_NODE_CORE_DETECT_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_DETECT_CORE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/model/detect/db.h"
#include "ocr_infer/core/node_core/core_base.h"

class DetectCore : public NodeCoreBase<DetInput, DetOutput> {
 public:
  DetectCore(const std::unordered_map<std::string, std::string> &config);

  std::shared_ptr<DetOutput> Process(
      const std::shared_ptr<DetInput> &in) override;

 private:
  int detector_num_;
  std::vector<std::unique_ptr<Db>> detector_;
  cv::Size det_input_size_;
};

#endif  // OCR_INFER_CORE_NODE_CORE_DETECT_CORE_H_
