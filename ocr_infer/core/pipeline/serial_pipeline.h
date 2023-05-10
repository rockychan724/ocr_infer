#ifndef OCR_INFER_CORE_PIPELINE_SERIAL_PIPELINE_H_
#define OCR_INFER_CORE_PIPELINE_SERIAL_PIPELINE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/common/transmission.h"
#include "ocr_infer/core/node_core/buffer_core.h"
#include "ocr_infer/core/node_core/clip_core.h"
#include "ocr_infer/core/node_core/detect_core.h"
#include "ocr_infer/core/node_core/detect_postprocessing_core.h"
#include "ocr_infer/core/node_core/match_core.h"
#include "ocr_infer/core/node_core/node.h"
#include "ocr_infer/core/node_core/recognize_core.h"

typedef std::unordered_map<std::string, std::string> Config;

// serial pipeline
class SerialE2ePipeline {
 public:
  SerialE2ePipeline(const Config &config);

  std::shared_ptr<MatchOutput> Run(const std::shared_ptr<DetInput> &in);

 private:
  std::unique_ptr<DetectCore> det_node_;
  std::unique_ptr<DetectPostProcessingCore> det_pp_node_;
  std::unique_ptr<ClipCore> clip_node_;
  std::unique_ptr<BufferCore> buffer_node_;
  std::unique_ptr<RecognizeCore> rec_node_;
  std::unique_ptr<MatchCore> match_node_;

  int rec_batch_size_;
};

#endif  // OCR_INFER_CORE_PIPELINE_SERIAL_PIPELINE_H_
