#ifndef OCR_INFER_CORE_MATCH_AC_AC_MATCHER_H_
#define OCR_INFER_CORE_MATCH_AC_AC_MATCHER_H_

#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

// TODO: 和 parallel ac 项目进行对比
class AcMatcher {
 public:
  AcMatcher(bool case_sensitive = true);  // TODO: true or false?

  int Insert(std::wstring word);

  int BuildAcAutomation();

  void SetCaseSensibility(bool case_sensitive);

  std::vector<std::wstring> Parse(std::wstring text);

 private:
  struct TreeNode {
    std::wstring wstr;  // 记录从根节点到当前节点的路径构成的字符序列
    int count;  // 记录当前节点是否构成一个词语
    std::map<wchar_t, std::shared_ptr<TreeNode>> next;
    std::shared_ptr<TreeNode> fail;  // 失配指针
  };

  // 节点队列，用于构造各个节点的失配指针
  std::queue<std::shared_ptr<TreeNode>> nodes_queue;
  // 头结点
  std::shared_ptr<TreeNode> root_;

  bool case_sensitive_;

  void InitNode(std::shared_ptr<TreeNode> node);

  std::wstring CaseTransform(const std::wstring &src);
};

#endif  // OCR_INFER_CORE_MATCH_AC_AC_MATCHER_H_
