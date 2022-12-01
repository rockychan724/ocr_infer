#ifndef OCR_INFER_CORE_NODE_CORE_RECOGNIZE_CORE_H_
#define OCR_INFER_CORE_NODE_CORE_RECOGNIZE_CORE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/model/recognize/crnn.h"
#include "ocr_infer/core/node_core/core_base.h"

class RecognizeCore : public NodeCoreBase<RecInput, RecOutput> {
 public:
  RecognizeCore(const std::unordered_map<std::string, std::string> &config);

  std::shared_ptr<RecOutput> Process(
      const std::shared_ptr<RecInput> &in) override;

 private:
  int recognizer_num_;
  std::vector<std::unique_ptr<Crnn>> recognizer_;
};

#endif  // OCR_INFER_CORE_NODE_CORE_RECOGNIZE_CORE_H_
