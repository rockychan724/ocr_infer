#ifndef OCR_INFER_CORE_MATCH_RULE_TREE_H_
#define OCR_INFER_CORE_MATCH_RULE_TREE_H_

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

template <class KeyType, class ContainerType>
class RuleTree;

template <class KeyType, class ContainerType>
class TreeNode {
 private:
  friend class RuleTree<KeyType, ContainerType>;
  bool complete_;  //判断这个节点是否完成
  int wordID_;     //保存这个节点的数据,设置规则id
  std::unordered_map<KeyType, std::shared_ptr<TreeNode<KeyType, ContainerType>>>
      next_;

 public:
  TreeNode() : complete_(false) {}
  std::shared_ptr<TreeNode<KeyType, ContainerType>> get_child(
      const KeyType& key) const  //根据key获取子节点
  {
    auto itr = next_.find(key);
    if (itr == next_.end()) {
      return std::shared_ptr<TreeNode<KeyType, ContainerType>>();
    } else {
      return itr->second;
    }
  }
  void set_child(const KeyType& key) {  //设置子节点
    next_[key] = std::make_shared<
        TreeNode<KeyType, ContainerType>>();  //得到一个指向trie_node的指针
  }
  bool is_complete() const { return complete_; }
  void set_complete(const bool& complete) { complete_ = complete; }
  int getID() const {
    return wordID_;  //得到id，是一个vector
  }
  void setID(const int& wordID) { wordID_ = wordID; }
  /*std::vector<int> getEd0() const
  {
      return Ed0_;
  }
  void setEd0(const int& ed0)
  {
      Ed0_.push_back(ed0);
  }*/
};

template <class KeyType, class ContainerType>
class RuleTree {
 private:
  std::shared_ptr<TreeNode<KeyType, ContainerType>> root_;  //根节点
 public:
  RuleTree() {  //构造函数
    root_ = std::make_shared<
        TreeNode<KeyType, ContainerType>>();  //得到根节点并设置根节点为false
    root_->set_complete(false);
  }

  void insert(const ContainerType& container);
  void insert(const ContainerType& container, const int& wordID);
  // void insert(const ContainerType& container, const int& wordID, const int&
  // ed0);
  std::vector<int> find(const ContainerType& container) const;

  // 20160628
  bool findsubstr(const ContainerType& container) const;
  std::vector<int> findsub(const ContainerType& container) const;
  bool findsubstr(const ContainerType& container, const int TOPK) const;
  std::vector<int> findsub(const ContainerType& container,
                           const int TOPK) const;

  // 20160629
  void remove();
  std::vector<std::vector<int>> fuzzymatch(const ContainerType& container,
                                           const int TOPK,
                                           const int editDistance) const;
  std::vector<std::vector<int>> fuzzymatch_one(const ContainerType& container,
                                               const int TOPK,
                                               const int editDistance) const;
};

template <class KeyType, class ContainerType>
void RuleTree<KeyType, ContainerType>::insert(const ContainerType& container) {
  auto current_node = root_;
  // auto i = begin(container);
  auto i = container.begin();
  // while (i != end(container)) {
  while (i != container.end()) {  //一步步操作，直至找到公共前缀不同的字符串
    auto j = current_node->get_child(*i);
    if (j.get() == NULL) {
      break;
    }
    current_node = j;
    ++i;
  }
  // while (i != end(container)) {
  while (i != container.end()) {
    current_node->set_child(*i);  //这时候，没有子节点，需要将其进行插入
    current_node = current_node->get_child(*i);
    ++i;
  }
  current_node->set_complete(true);  //完成插入
}

template <class KeyType, class ContainerType>
void RuleTree<KeyType, ContainerType>::insert(const ContainerType& container,
                                              const int& wordID) {
  auto current_node = root_;
  // auto i = begin(container);
  auto i = container.begin();
  // while (i != end(container)) {
  while (i != container.end()) {
    auto j = current_node->get_child(*i);
    if (j.get() == NULL) {
      break;
    }
    current_node = j;
    ++i;
  }
  // while (i != end(container)) {
  while (i != container.end()) {
    current_node->set_child(*i);
    current_node = current_node->get_child(*i);
    ++i;
  }
  current_node->set_complete(true);
  current_node->setID(wordID);
  // current_node->setEd0(ed0);               //在叶子节点中保存wordID
  // 和ed的信息，规则树里面就是对应的规则id
}

template <class KeyType, class ContainerType>
std::vector<int> RuleTree<KeyType, ContainerType>::find(
    const ContainerType& container)
    const  //返回一个vector，包括该字符串是否在这棵树上面和其对应的具体id
{
  auto current_node = root_;
  std::vector<int> result;
  int word_id;
  for (auto it = container.begin(); it != container.end(); ++it) {
    auto j = current_node->get_child(*it);
    if (j.get() == NULL) {  //直接找不到了
      result.push_back(0);
      result.push_back(0);
      return result;
    }
    current_node = j;
  }
  if (current_node->is_complete()) {  //找到了叶子节点
    result.push_back(1);
    word_id = current_node->getID();
    result.push_back(word_id);
    return result;
  } else {
    result.push_back(0);
    result.push_back(0);
    return result;
  }
}

template <class KeyType, class ContainerType>
void RuleTree<KeyType, ContainerType>::remove()  //
{
  // root_= std::make_shared<TreeNode<KeyType, ContainerType>>();

  root_.reset(new TreeNode<KeyType, ContainerType>());
  root_->set_complete(false);
}

typedef RuleTree<int, std::vector<int>> int_trie;

#endif  // OCR_INFER_CORE_MATCH_RULE_TREE_H_
