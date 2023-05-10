#include "ocr_infer/core/pipeline/serial_pipeline.h"

#include <cmath>

SerialE2ePipeline::SerialE2ePipeline(const Config &config) {
  det_node_ = std::make_unique<DetectCore>(config);
  det_pp_node_ = std::make_unique<DetectPostProcessingCore>(config);
  clip_node_ = std::make_unique<ClipCore>(config);
  buffer_node_ = std::make_unique<BufferCore>(config);
  rec_node_ = std::make_unique<RecognizeCore>(config);
  match_node_ = std::make_unique<MatchCore>(config);

  rec_batch_size_ = std::stoi(Query(config, "reco_batch_size"));
}

std::shared_ptr<MatchOutput> SerialE2ePipeline::Run(
    const std::shared_ptr<DetInput> &in) {
  std::shared_ptr<MatchOutput> output = std::make_shared<MatchOutput>();

  std::shared_ptr<DetOutput> det_out = det_node_->Process(in);
  std::shared_ptr<DetBox> det_pp_out = det_pp_node_->Process(det_out);
  std::shared_ptr<RecInput> clip_out = clip_node_->Process(det_pp_out);

  int clip_num = clip_out->clips.size();
  int rec_batch_num = std::ceil(double(clip_num) / rec_batch_size_);
  int begin_index = 0;
  for (int i = 0; i < rec_batch_num; i++) {
    std::shared_ptr<RecInput> rec_in = std::make_shared<RecInput>();
    int end_index = begin_index + rec_batch_size_ >= clip_num
                        ? clip_num
                        : begin_index + rec_batch_size_;
    rec_in->names.assign(clip_out->names.begin() + begin_index,
                         clip_out->names.begin() + end_index);
    rec_in->clips.assign(clip_out->clips.begin() + begin_index,
                         clip_out->clips.begin() + end_index);
    rec_in->boxnum.assign(clip_out->boxnum.begin() + begin_index,
                          clip_out->boxnum.begin() + end_index);
    rec_in->boxes.assign(clip_out->boxes.begin() + begin_index,
                         clip_out->boxes.begin() + end_index);
    begin_index = end_index;

    std::shared_ptr<RecOutput> rec_out = rec_node_->Process(rec_in);
    std::shared_ptr<MatchOutput> match_out = match_node_->Process(rec_out);

    // gather match result
    output->name2boxnum.insert(match_out->name2boxnum.begin(),
                               match_out->name2boxnum.end());
    output->name2text.insert(match_out->name2text.begin(),
                             match_out->name2text.end());
    output->name2boxes.insert(match_out->name2boxes.begin(),
                              match_out->name2boxes.end());
    output->name2hitid.insert(match_out->name2hitid.begin(),
                              match_out->name2hitid.end());
  }

  return output;
}
