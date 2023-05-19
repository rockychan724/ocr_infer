#include "ocr_infer/core/node_core/match_core.h"

#include <filesystem>

#include "glog/logging.h"

MatchCore::MatchCore(
    const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Match node init...";

  std::string keyword_dir = Inquire(config, "keyword_dir");
  CHECK(std::filesystem::exists(keyword_dir))
      << "Can't find keyword directory " << keyword_dir;

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
