#include "ocr_infer/core/node_core/detect_postprocessing_core.h"

#include "glog/logging.h"

DetectPostProcessingCore::DetectPostProcessingCore(
    const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Detect postprocessing node init...";
  det_pp_ = std::make_unique<DbPostprocessing>();
  LOG(INFO) << "Detect postprocessing node init over!";
}

std::shared_ptr<DetBox> DetectPostProcessingCore::Process(const std::shared_ptr<DetOutput> &in) {
  VLOG(1) << "*** Detect postprocessing node, in size = " << in->images.size();
  if (in->images.size() == 0) {
    return {};
  }

  auto out = std::make_shared<DetBox>();

  for (int i = 0; i < in->images.size(); i++) {
    std::vector<cv::RotatedRect> results;
    det_pp_->Parse(in->results[i], in->scales[i], &results);
    out->boxes.emplace_back(results);
    VLOG(1) << "box size = " << results.size();
  }
  out->names.assign(in->names.begin(), in->names.end());
  out->images.assign(in->images.begin(), in->images.end());

  VLOG(1) << "*** Detect postprocessing node, out size = " << in->images.size();
  return out;
}
