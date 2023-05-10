#ifndef OCR_INFER_CORE_PIPELINE_PIPELINE_H_
#define OCR_INFER_CORE_PIPELINE_PIPELINE_H_

#include "ocr_infer/core/common/data_structure.h"
#include "ocr_infer/core/common/transmission.h"

typedef std::unordered_map<std::string, std::string> Config;
typedef std::pair<std::shared_ptr<QueueSender<DetInput>>,
                  std::shared_ptr<QueueReceiver<MatchOutput>>>
    E2eInOutPair;
typedef std::pair<std::shared_ptr<QueueSender<DetInput>>,
                  std::shared_ptr<QueueReceiver<RecInput>>>
    DetInOutPair;
typedef std::pair<std::shared_ptr<QueueSender<RecInput>>,
                  std::shared_ptr<QueueReceiver<MatchOutput>>>
    RecInOutPair;

// parallel pipeline
class PipelineFactory {
 public:
  static E2eInOutPair BuildE2e(const Config &config);
  static DetInOutPair BuildDet(const Config &config);
  static RecInOutPair BuildRec(const Config &config);
};

#endif  // OCR_INFER_CORE_PIPELINE_PIPELINE_H_
