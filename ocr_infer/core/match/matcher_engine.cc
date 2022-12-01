#include "ocr_infer/core/match/matcher_engine.h"

#include "glog/logging.h"
#include "ocr_infer/core/match/ac/ac_matcher.h"
#include "ocr_infer/core/match/fuzzy/fuzzy_matcher.h"
#include "ocr_infer/core/util/file_handle.h"

MatcherEngine::MatcherEngine(const std::string &keyword_dir) {
  LOG(INFO) << "Initializing matcher...";
  // TODO: use AcMatcher or FuzzyMatcher
  matcher_ = std::make_unique<AcMatcher>();
  rule_tree_ = std::make_unique<RuleTree>();

  ReadKeyword(keyword_dir);

  LOG(INFO) << "Matcher init over!";
}

KeywordId MatcherEngine::Match(const std::vector<std::string> &texts) {}

int MatcherEngine::AddKeyword(KeywordId id, const std::string &keyword,
                              int flag) {}

int MatcherEngine::DeleteKeyword(KeywordId id, int flag) {}

void MatcherEngine::ReadKeyword(const std::string &keyword_dir) {
  // const std::wstring ANDPRE = L"#AND";
  // const std::wstring ORPRE = L"#OR";
  // const std::wstring NOTPRE = L"#NOT";
  // std::vector<std::string> filename = GetFiles(keyword_dir, "txt", false,
  // false); int word_cnt = 0;
  // // wcout.imbue(locale("chs"));
  // int minganci_num;

  // std::unordered_map<long long, std::vector<std::std::wstring>> sensitives;

  // for (int i = 0; i < filename.size(); i++) {
  //   std::string tempfile = dirname + filename[i];
  //   std::vector<std::wstring> tl = readUnicodeFile_pwy(tempfile.c_str());
  //   long long id = atoi(filename[i].substr(0, filename[i].length() -
  //   4).c_str());
  //   //        cout << tempfile << endl;
  //   if (tl.size() == 0) {
  //     continue;
  //   }
  //   for (int j = 0; j < tl.size(); j++) {
  //     if (wcscmp(tl[j].c_str(), ORPRE.c_str()) ==
  //         0)
  //         //当出现或的关系时，继续添加下一个关键词，即关键词树上不出现或标志
  //     {
  //       // cout << "****appear****" << endl;
  //       continue;
  //     }

  //     // wcout << tl[j] << endl;
  //     // std::map<std::std::wstring, long long>::iterator iter =
  //     // Minganci_iddict.find(Unicode2Utf8(tl[j]));
  //     minganci_num = Minganci_iddict.count(tl[j]);
  //     // std::map<std::std::wstring, long long>::iterator iter =
  //     Minganci_iddict.find(tl[j]); if (minganci_num == 0)
  //     //当前map中的key不存在,才在map中添加信息，并同时向AC自动机中插入数据
  //     {
  //       // wcout << tl[j] << endl;
  //       Minganci_iddict.insert({tl[j], word_cnt});
  //       t_ac.insert(tl[j]);
  //       word_cnt += 1;
  //     }
  //   }
  //   /* add by cl */
  //   sensitives.insert({id, tl});
  //   /* */
  // }
  // t_ac.build_ac_automation();

  // return sensitives;
}
