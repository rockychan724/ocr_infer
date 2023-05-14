#include "ocr_infer/core/node_core/match_core.h"

#include "glog/logging.h"

MatchCore::MatchCore(
    const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Match node init...";

  // const std::string keyword_dir = Inquire(config, "keyword_dir");
  std::string root_path = Inquire(config, "root_path");
  if (root_path.back() != '/') {
    root_path += "/";
  }
  std::string keyword_dir = Inquire(config, "keyword_dir");
  keyword_dir = root_path + keyword_dir;
  matcher_engine_ = std::make_unique<MatcherEngine>(keyword_dir);

  LOG(INFO) << "Match node init over!";
}

std::shared_ptr<MatchOutput> MatchCore::Process(
    const std::shared_ptr<OcrOutput> &in) {
  auto out = std::make_shared<MatchOutput>(*in.get());

  for (int i = 0; i < in->names.size(); i++) {
    out->hitid.emplace_back(matcher_engine_->Match(in->multitext[i]));
  }

  return out;
}
