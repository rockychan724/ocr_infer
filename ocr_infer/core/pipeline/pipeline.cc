#include "ocr_infer/core/pipeline/pipeline.h"

#include "ocr_infer/core/node_core/buffer_core.h"
#include "ocr_infer/core/node_core/clip_core.h"
#include "ocr_infer/core/node_core/detect_core.h"
#include "ocr_infer/core/node_core/detect_postprocessing_core.h"
#include "ocr_infer/core/node_core/match_core.h"
#include "ocr_infer/core/node_core/node.h"
#include "ocr_infer/core/node_core/recognize_core.h"

class PipelineE2e {
 public:
  std::shared_ptr<QueueSender<DetInput>> e2e_sender_;
  std::shared_ptr<QueueReceiver<MatchOutput>> e2e_receiver_;

  PipelineE2e(const Config &config) {
    auto det_input_pair = QueueFactory<DetInput>::BuildQueue();
    auto det_output_pair = QueueFactory<DetOutput>::BuildQueue();
    auto det_box_pair = QueueFactory<DetBox>::BuildQueue();
    auto rec_input_pair1 = QueueFactory<RecInput>::BuildQueue();
    auto rec_input_pair2 = QueueFactory<RecInput>::BuildQueue();
    auto rec_output_pair = QueueFactory<RecOutput>::BuildQueue();
    auto match_output_pair = QueueFactory<MatchOutput>::BuildQueue();

    e2e_sender_ = det_input_pair.first;
    det_node_.SetUp(config, det_input_pair.second, det_output_pair.first, "");
    det_pp_node_.SetUp(config, det_output_pair.second, det_box_pair.first, "");
    clip_node_.SetUp(config, det_box_pair.second, rec_input_pair1.first, "");
    buffer_node_.SetUp(config, rec_input_pair1.second, rec_input_pair2.first, "");
    rec_node_.SetUp(config, rec_input_pair2.second, rec_output_pair.first, "");
    match_node_.SetUp(config, rec_output_pair.second, match_output_pair.first, "");
    e2e_receiver_ = match_output_pair.second;
  }

 private:
  Node<DetInput, DetOutput, DetectCore> det_node_;
  Node<DetOutput, DetBox, DetectPostProcessingCore> det_pp_node_;
  Node<DetBox, RecInput, ClipCore> clip_node_;
  Buffer<RecInput, RecInput, BufferCore> buffer_node_;
  Node<RecInput, RecOutput, RecognizeCore> rec_node_;
  Node<RecOutput, MatchOutput, MatchCore> match_node_;
};

E2eInOutPair PipelineFactory::BuildE2e(const Config &config) {
  // std::shared_ptr<PipelineE2e> pipeline = std::make_shared<PipelineE2e>(config);
  auto pipeline = new PipelineE2e(config);
  return std::make_pair(pipeline->e2e_sender_, pipeline->e2e_receiver_);
}

DetInOutPair PipelineFactory::BuildDet(const Config &config) { return {}; }

RecInOutPair PipelineFactory::BuildRec(const Config &config) { return {}; }
