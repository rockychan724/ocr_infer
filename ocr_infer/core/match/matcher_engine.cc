#include "ocr_infer/core/match/matcher_engine.h"

#include <algorithm>

#include "glog/logging.h"
#include "ocr_infer/core/match/ac/ac_matcher.h"
#include "ocr_infer/core/match/fuzzy/fuzzy_matcher.h"
#include "ocr_infer/core/util/code_convert.h"
#include "ocr_infer/core/util/file_handle.h"

void GetSubstr(std::vector<int> &src, int pos, std::vector<int> substr,
               std::vector<std::vector<int>> &vec) {
  if (pos == src.size()) {
    vec.emplace_back(substr);
    return;
  }

  int i = src[pos];
  GetSubstr(src, pos + 1, substr, vec);
  substr.emplace_back(i);
  GetSubstr(src, pos + 1, substr, vec);
}

MatcherEngine::MatcherEngine(const std::string &keyword_dir) {
  LOG(INFO) << "Initializing matcher...";
  // TODO: use AcMatcher or FuzzyMatcher
  matcher_ = std::make_unique<AcMatcher>();
  rule_tree_ = std::make_unique<RuleTree<int, std::vector<int>>>();

  ReadKeyword(keyword_dir);

  LOG(INFO) << "Matcher init over!";
}

KeywordId MatcherEngine::Match(const std::vector<std::string> &texts) {
  std::vector<std::wstring> utf16_texts = BatchUtf8ToUtf16(texts);

  int matched = 0;
  int single_length;  // 控制每一个词条里面出现的敏感词个数
  std::vector<KeywordId> res_match;
  std::vector<int> hit_word_vector;
  std::wstring dict_key;
  std::vector<std::vector<int>> sub_hit_word;  // 命中关键词的子序列
  std::vector<int> init_sub_word;              // 初始化的空vector
  std::vector<int> hit_result;
  std::vector<int> comb_result;

  std::vector<std::wstring> hit_words;
  for (size_t i = 0; i < utf16_texts.size(); i++) {
    std::vector<std::wstring> results = matcher_->Parse(utf16_texts[i]);
    if (results.size() > 0) {
      hit_words.insert(hit_words.end(), results.begin(), results.end());
    }
  }

  for (const auto &result : hit_words) {
    auto iter = keyword_to_wordid_.find(result);
    if (iter != keyword_to_wordid_.end()) {
      hit_word_vector.emplace_back(iter->second);  // 得到命中关键词的向量序列
    }
  }

  std::sort(hit_word_vector.begin(),
            hit_word_vector.end());  // 对命中的关键词排序
  hit_word_vector.erase(
      std::unique(hit_word_vector.begin(), hit_word_vector.end()),
      hit_word_vector.end());  // 去除命中重复的关键词
  if (hit_word_vector.size() > 2) {
    if (hit_word_vector.size() > 6) {
      single_length = 6;
    } else {
      single_length = hit_word_vector.size();
    }
    for (int k = 1; k <= single_length; k++) {  // 查找组合数。
      // k代表在hit_word_vector中查找几个数的组合
      comb_result = CombBack(hit_word_vector, hit_word_vector.size(), k);
      if (comb_result[0] == 1) {
        res_match.emplace_back(comb_result[1]);
        break;
      }
    }
  } else {
    GetSubstr(hit_word_vector, 0, init_sub_word,
              sub_hit_word);  // 得到命中关键词的子序列
    for (int j = 0; j < sub_hit_word.size(); j++) {
      if (sub_hit_word[j].size() == 0) continue;
      hit_result = rule_tree_->find(sub_hit_word[j]);
      if (hit_result[0] == 1) {
        auto it =
            find(invalid_rule_.begin(), invalid_rule_.end(), hit_result[1]);
        if (it != invalid_rule_.end()) {  // 如果是失效id，则继续下一个
          continue;
        } else {  // 如果不是失效id，则此时命中的id就是命中的关键词序列
          res_match.emplace_back(hit_result[1]);
          break;
        }
      }
    }
  }

  return res_match.empty() ? 0 : res_match[0];
}

int MatcherEngine::AddKeyword(KeywordId rule_id, const std::string &keyword,
                              int flag) {
  return 0;
}

int MatcherEngine::DeleteKeyword(KeywordId rule_id, int flag) { return 0; }

/**
 * 注意区分 rule id 和 word id：
 * 1. 一个关键词文件对应一个 rule，因此 rule id 是文件 id；
 * 2. keyword 是关键词文件中的每一行，因此 word id 是所有关键词的编码 id。
 */
void MatcherEngine::ReadKeyword(std::string keyword_dir) {
  const std::wstring ANDPRE = L"#AND";
  const std::wstring ORPRE = L"#OR";
  const std::wstring NOTPRE = L"#NOT";
  if (keyword_dir.back() != '/') {
    keyword_dir += "/";
  }
  std::vector<std::string> filename =
      GetFilesV1(keyword_dir, "txt", false, false);
  int word_id = 0;
  for (int i = 0; i < filename.size(); i++) {
    std::string file_path = keyword_dir + filename[i];
    std::vector<std::wstring> tl = ReadUnicodeFile(file_path);
    if (tl.size() == 0) {
      continue;
    }
    KeywordId rule_id = std::stoi(filename[i]);  // file id
    CHECK_GT(rule_id, 0) << "File id must be greater than 0!";
    CHECK(ruleid_to_keyword_.find(rule_id) == ruleid_to_keyword_.end())
        << "file id " << rule_id << " is duplicate, please check " << rule_id
        << ".txt!";
    ruleid_to_keyword_.insert({rule_id, tl});

    std::vector<int> rule;
    for (int j = 0; j < tl.size(); j++) {
      // 当出现或的关系时，代表一条规则已经完成,支持添加多条规则;
      // 继续添加下一个关键词，即关键词树上不出现或标志
      if (wcscmp(tl[j].c_str(), ORPRE.c_str()) == 0) {
        std::sort(rule.begin(), rule.end());
        rule_tree_->insert(rule, static_cast<int>(rule_id));
        rule.clear();
        continue;
      }
      // 当前map中的key不存在,才在map中添加信息，并同时向AC自动机中插入数据
      if (keyword_to_wordid_.find(tl[j]) == keyword_to_wordid_.end()) {
        keyword_to_wordid_.insert({tl[j], word_id});
        matcher_->Insert(tl[j]);
        word_id += 1;
      } else {
        rule.emplace_back(keyword_to_wordid_[tl[j]]);
      }
    }
    if (!rule.empty()) {
      std::sort(rule.begin(), rule.end());
      rule_tree_->insert(rule, rule_id);
    }
  }
  matcher_->BuildAcAutomation();
}

std::vector<int> MatcherEngine::CombBack(std::vector<int> &hit_word_vector,
                                         int m, int r) {
  std::vector<int> result;
  std::vector<int> hit_result;
  result.clear();

  int *a = new int[r];  // 创建一个新数组存储构成组合的3个数
  int i, j, k = 0;
  i = 0, a[i] = 1;
  do {
    // a[i]-i<+m-r+1
    // 是判断a[i]的值是否在范围之内如i=0时,a[i]最大可以为倒数第r个数,这样本组数的最大值才不至于大过m
    if (a[i] <= m - r + 1 + i) {
      if (i == r - 1)  // 到了最底层，就一直运行这一分支
      {
        std::vector<int> sub_hit;
        for (j = 0; j < r; j++) {
          sub_hit.emplace_back(hit_word_vector[a[j] - 1]);
        }
        hit_result = rule_tree_->find(sub_hit);
        if (hit_result[0] == 1) {
          // cout << "hit_rule_id: "<< hit_result[1]<<endl;
          // 查找当前命中的id是否位于失效ID里面
          auto it = std::find(invalid_rule_.begin(), invalid_rule_.end(),
                              hit_result[1]);
          if (it == invalid_rule_.end()) {  // 没有在失效id里面
            result.emplace_back(1);
            result.emplace_back(hit_result[1]);
            return result;
          }
        }
        a[i]++;
        sub_hit.clear();
        continue;
      }
      i++;    // 前进到下一层试探,深搜
      a[i] = a[i - 1] + 1;
    } else {  // 回溯到上一层进行试探
      if (i == 0) {
        result.emplace_back(0);
        return result;  // 已经找到了所有层的解
      }
      a[--i]++;  // 前一层的数字增加1,继续进行向前试探,其总共就r层
    }
  } while (1);

  delete[] a;
}
