#ifndef OCR_INFER_CORE_NODE_CORE_CLIP_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_CLIP_CORE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/node_core/core_base.h"

class ClipCore : public NodeCoreBase<DetBox, RecInput> {
 public:
  ClipCore(const std::unordered_map<std::string, std::string> &config);

  std::shared_ptr<RecInput> Process(const std::shared_ptr<DetBox> &in) override;
};

#endif  // OCR_INFER_CORE_NODE_CORE_CLIP_CORE_H_
