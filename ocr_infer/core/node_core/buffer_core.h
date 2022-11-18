#ifndef OCR_INFER_CORE_NODE_CORE_BUFFER_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_BUFFER_CORE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/node_core/core_base.h"

class BufferCore : public BufferCoreBase<RecInput, RecInput> {
 public:
  BufferCore(const std::unordered_map<std::string, std::string> &config);

  void Process(const std::shared_ptr<RecInput> &in,
               std::vector<std::shared_ptr<RecInput>> *out) override;
};

#endif  // OCR_INFER_CORE_NODE_CORE_BUFFER_CORE_H_
