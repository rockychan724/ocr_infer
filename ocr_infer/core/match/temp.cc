#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "DirHandle.h"
#include "Trie_rule.h"
#include "codeUtils.h"

using namespace std;

vector<std::wstring> readUnicodeFile_pwy_rule(const char* filename) {
  vector<wstring> re;
  ifstream ifs_(filename, ios::in | ios::binary | ios::ate);
  if (!ifs_) {
    cout << "cannot open file " << filename << endl;
    return re;
  }

  ifs_.seekg(2, ios::beg);

  wstring ws = L"";
  while (true) {
    wchar_t wc = 0x0000;
    ifs_.read((char*)(&wc), 2);

    if (wc == '\r') continue;

    if (wc == 0x0000 || wc == 0x000a) {
      wchar_t* tmp_ws = new wchar_t[ws.size() + 1];
      for (int i = 0; i < ws.size(); i++) {
        tmp_ws[i] = ws[i];
        // cout << setw(4) << setfill('0') << hex << (int)ws[i] << " ";
      }
      // cout << endl;
      tmp_ws[ws.size()] = 0x0000;
      if (ws.size() != 0) {  //&& !isENGString(ws)) {
        wstring tmp_ws_mid = tmp_ws;
        transform(tmp_ws_mid.begin(), tmp_ws_mid.end(), tmp_ws_mid.begin(),
                  towupper);
        // cout<<"add key----"<<Unicode2Utf8(tmp_ws_mid)<<endl;
        re.push_back(tmp_ws_mid);
      }
      ws = L"";
      delete[] tmp_ws;
    } else {
      ws += wc;
    }
    if (wc == 0x0000) break;
  }

  ifs_.close();
  return re;
}

void Trie_SetRuleFromDir(const char* DirName, ori_Trie::int_trie& int_trie,
                         std::map<std::wstring, long long>&
                             Minganci_iddict)  //从路径中设置Trie的路径
{
  const wstring ANDPRE = L"#AND";
  const wstring ORPRE = L"#OR";
  const wstring NOTPRE = L"#NOT";
  string dirname(DirName);
  vector<string> filename = GetFiles(dirname, "txt", false, false);
  vector<int> rule;
  long long word_id;
  for (int i = 0; i < filename.size(); i++) {
    string tempfile = DirName + filename[i];
    vector<wstring> tl = readUnicodeFile_pwy_rule(tempfile.c_str());
    long long id =
        atoi(filename[i].substr(0, filename[i].length() - 4).c_str());
    if (tl.size() == 0) {
      continue;
    }
    for (int j = 0; j < tl.size(); j++) {
      if (wcscmp(tl[j].c_str(), ORPRE.c_str()) ==
          0)  //当出现或的关系时，代表一条规则已经完成,支持添加多条规则
      {
        // cout << "####appear#####" << endl;

        sort(rule.begin(), rule.end());
        int_trie.insert(rule, id);
        rule.clear();
        continue;

      } else if (Minganci_iddict.count(tl[j]) > 0) {
        word_id = Minganci_iddict[tl[j]];
        // cout << "word_id: " << word_id << endl;
        rule.push_back(word_id);
      }
    }
    if (rule.size() != 0) {  //当到达一个规则的末尾时，代表该规则添加已完成
      sort(rule.begin(), rule.end());
      int_trie.insert(rule, id);
      rule.clear();
    }

    // sort(rule.begin(), rule.end()); //对规则中的敏感词id进行升序排列
    /*		cout <<"add_rule: " << id <<endl;
            cout << "######" << endl;
            for (auto r : rule)
            {
                cout << r << endl;
            }*/
    // int_trie.insert(rule, id);                    //规则树上添加该规则
    // rule.clear();
  }
}

void addrule_to_trie(ori_Trie::int_trie& int_trie,
                     std::map<std::wstring, long long>& Minganci_iddict,
                     std::vector<std::wstring> add_rule, long long id) {
  //一次添加一条规则
  const wstring ANDPRE = L"#AND";
  const wstring ORPRE = L"#OR";
  const wstring NOTPRE = L"#NOT";
  std::vector<int> rule;
  long long word_id;
  for (int i = 0; i < add_rule.size(); i++) {
    if (wcscmp(add_rule[i].c_str(), ORPRE.c_str()) == 0) {
      if (rule.size() == 0) {
        continue;
      } else {  //遇到或的关系时了。
        sort(rule.begin(), rule.end());
        int_trie.insert(rule, id);
        rule.clear();
        continue;
      }
    } else if (Minganci_iddict.count(add_rule[i]) > 0) {
      word_id = Minganci_iddict[add_rule[i]];
      rule.push_back(word_id);
    }
  }
  if (rule.size() != 0) {  //当到达一个规则的末尾时，代表该规则添加已完成
    sort(rule.begin(), rule.end());
    int_trie.insert(rule, id);
    rule.clear();
  }
}
