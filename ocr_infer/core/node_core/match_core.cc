#include "ocr_infer/core/node_core/match_core.h"

#include "glog/logging.h"

MatchCore::MatchCore(const std::unordered_map<std::string, std::string> &config) {
  LOG(INFO) << "Match node init over!";
}

std::shared_ptr<MatchOutput> MatchCore::Process(const std::shared_ptr<RecOutput> &in) {
  return {};
}
