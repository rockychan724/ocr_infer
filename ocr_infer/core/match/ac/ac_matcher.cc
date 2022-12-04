#include "ocr_infer/core/match/ac/ac_matcher.h"

AcMatcher::AcMatcher(bool case_sensitive) : case_sensitive_(case_sensitive) {
  root_ = std::make_shared<TreeNode>();
  InitNode(root_);
}

// insert a word into this tree
int AcMatcher::Insert(std::wstring word) {
  if (!case_sensitive_) {
    word = CaseTransform(word);
  }
  std::shared_ptr<TreeNode> temp = root_;
  std::shared_ptr<TreeNode> new_node;
  for (size_t i = 0; i < word.length(); i++) {
    wchar_t key = word[i];
    if (temp->next.find(key) == temp->next.end()) {
      new_node.reset(new TreeNode());
      InitNode(new_node);
      temp->next[key] = new_node;
      temp->next[key]->wstr = temp->wstr + key;
    }
    temp = temp->next[key];
  }
  temp->count = 1;
  return 0;  // TODO: check status
}

// use BFS algorithm to build all fail pointers in this tree
int AcMatcher::BuildAcAutomation() {
  nodes_queue.push(root_);
  while (!nodes_queue.empty()) {
    std::shared_ptr<TreeNode> temp = nodes_queue.front();  // 获取队头结点
    nodes_queue.pop();
    // 遍历当前节点的所有子节点
    for (auto iter = temp->next.begin(); iter != temp->next.end(); iter++) {
      wchar_t key = iter->first;
      std::shared_ptr<TreeNode> value = iter->second;
      value->fail = nullptr;  // for add sensitive words
      // 若是第一层中的节点，则把节点的失配指针指向root
      if (temp == root_) {
        value->fail = root_;
      } else {
        std::shared_ptr<TreeNode> p = temp->fail;
        while (p != nullptr) {
          if (p->next.find(key) != p->next.end()) {
            value->fail = p->next[key];
            break;
          }
          p = p->fail;
        }
        if (p == nullptr) {
          value->fail = root_;
        }
      }
      // 将当前节点temp的各个子节点加入队列
      nodes_queue.push(iter->second);
    }
  }
}

void AcMatcher::SetCaseSensibility(bool case_sensitive) {
  case_sensitive_ = case_sensitive;
}

std::vector<std::wstring> AcMatcher::Parse(std::wstring text) {
  if (!case_sensitive_) {
    text = CaseTransform(text);
  }
  std::vector<std::wstring> result;
  std::shared_ptr<TreeNode> p = root_;
  for (size_t i = 0; i < text.length(); i++) {
    wchar_t key = text[i];
    while (p->next.find(key) == p->next.end() && p != root_) {
      p = p->fail;
    }
    if (p->next.find(key) == p->next.end()) {
      p = root_;
    } else {
      p = p->next[key];
    }
    std::shared_ptr<TreeNode> temp = p;
    while (temp != root_) {
      if (temp->count > 0) {
        result.emplace_back(temp->wstr);
      } else {
        break;
      }
      temp = temp->fail;
    }
  }
  return result;
}

// TODO: 放入 TreeNode 的构造函数
// init one node
void AcMatcher::InitNode(std::shared_ptr<TreeNode> node) {
  node->wstr = L"";
  node->count = 0;
  node->next.clear();
  node->fail = nullptr;
}

std::wstring AcMatcher::CaseTransform(const std::wstring &src) {
  std::wstring dest = src;
  for (size_t i = 0; i < dest.length(); i++) {
    if (int(dest[i]) >= 'a' && int(dest[i]) <= 'z') {
      dest[i] -= 32;
    }
  }
  return dest;
}
