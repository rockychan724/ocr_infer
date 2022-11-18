#include "ocr_infer/core/node_core/match_core.h"

MatchCore::MatchCore(const std::unordered_map<std::string, std::string> &config) {
  std::cout << "*** Match core ***\n";
}

std::shared_ptr<MatchOutput> MatchCore::Process(const std::shared_ptr<RecOutput> &in) {
  std::cout << "=== match core process ===\n";
  return {};
}
