# coding:utf-8
from PIL import ImageFile
import sys
import os
import subprocess
import re
import shutil
from jan2fan import Traditional2Simplified
from tqdm import tqdm


def eval_line_acc(gt_folder="./gt", rel_folder="./rel"):
    gt_files = os.listdir(gt_folder)
    rel_files = os.listdir(rel_folder)
    # print("the gt has files "+str( len(gt_files)) )
    # print("the rel has files "+str( len(rel_files)) )
    not_exist_num = 0
    gt_line_nums = 0
    rel_line_nums = 0
    acc_line_nums = 0

    for item in tqdm(gt_files):
        gt_txt = gt_folder + "/" + item
        rel_txt = rel_folder + "/" + item
        # print('\n\n')
        if not os.path.exists(rel_txt):
            # print(rel_txt+' not exists')
            rel = []
        else:
            rel_f = open(rel_txt)
            rel_line = rel_f.readlines()
            rel_line = filter(lambda x: x.strip(), rel_line)
            rel = [filter_str(line) for line in rel_line]
            rel = j2fconvert(rel)
            rel = [line for line in rel if len(line) > 0]
        with open(gt_txt) as gt_f:
            gt_line = gt_f.readlines()
            gt_line = filter(lambda x: x.strip(), gt_line)
            gt = [filter_str(line) for line in gt_line]
            gt = j2fconvert(gt)
            gt = [line for line in gt if len(line) > 0]

            # print('gt:{}'.format(gt))
            # print('rel:{}'.format(rel))
            acc_num = 0
            for line in gt:
                if line in rel:
                    acc_num += 1

            gt_line_nums += len(gt)
            rel_line_nums += len(rel)
            acc_line_nums += acc_num
            # print("-------acc,gt,rel----------",acc_num,len(gt),len(rel))
        if len(rel) != 0 and len(gt) != 0:
            single_recall = acc_num / len(gt)
            single_precesion = acc_num / len(rel)
        if acc_num == 0:
            single_recall = 0
            single_precesion = 0
        if (single_recall + single_precesion) != 0:
            single_F = (single_recall * single_precesion * 2) / (
                single_recall + single_precesion
            )
        else:
            single_F = 0
        # print('%16s'%rel_txt.split("/")[-1]+'     recall: ' + '%0.4f' % single_recall + '      precesion: '\
        #                          + '%0.4f' % single_precesion + '       F: ' + '%0.4f' % single_F)
        # if single_F < 1:
        #     with open('error/error.txt','a') as wf:
        #         wf.write('\n' + rel_txt.split("/")[-1]+'\n')
        #         wf.writelines(gt_line)
        #         wf.writelines(rel_line)

    recall = acc_line_nums / gt_line_nums
    precesion = acc_line_nums / rel_line_nums
    # print("------acc_line_nums,gt_line_nums,rel_line_nums-----------",acc_line_nums,gt_line_nums,rel_line_nums)
    if (recall + precesion) < 0.000001:
        F = 0
    else:
        F = (recall * precesion * 2) / (recall + precesion)
    # print('gt_line_nums: ' + '%0.4f' % gt_line_nums)
    # print('rel_line_nums: ' + '%0.4f' % rel_line_nums)
    # print('acc_line_nums: ' + '%0.4f' % acc_line_nums )
    # print("\nfinal eval:")
    # print('total   recall: ' + '%0.4f' % recall\
    #                          + '        precesion: ' + '%0.4f' % precesion+ '       F: ' + '%0.4f' % F)
    return recall, precesion, F


# eval_line_acc()


def trrCach(gt_trr):  # t
    gt_cach = []
    gt_lnum = []
    for gtline in gt_trr:
        for gtc in gtline:
            incach = True
            idx = -1
            try:
                idx = gt_cach.index(gtc)
            except ValueError:
                incach = False

            if incach:
                gt_lnum[idx] += 1
            else:
                gt_cach.append(gtc)
                gt_lnum.append(int(1))
    return gt_cach, gt_lnum


def charMatch(tg, rect):  # t
    tg_cach, tg_lnum = tg
    rect_cach, rect_lnum = rect
    I = 0
    O = 0
    for idx in range(0, len(tg_cach)):
        tgc = tg_cach[idx]
        tgn = tg_lnum[idx]
        incach = True
        idy = -1
        try:
            idy = rect_cach.index(tgc)
        except ValueError:
            incach = False
        if incach:
            rcn = rect_lnum[idy]
            if rcn < tgn:
                I += rcn
                O += tgn - rcn
            else:
                I += tgn
        else:
            O += tgn
    return I, O


def trrsCompare(gt_trr, res_trr):  # t
    gt_trr = [filter_str(line) for line in gt_trr]
    res_trr = [filter_str(line) for line in res_trr]
    gt_trr = j2fconvert(gt_trr)
    res_trr = j2fconvert(res_trr)
    # print('gt:{}'.format(gt_trr))
    # print('res:{}'.format(res_trr))
    gt = trrCach(gt_trr)
    res = trrCach(res_trr)
    # print('gt:{}'.format(gt))
    # print('res:{}'.format(res))
    TP1, FN = charMatch(gt, res)
    TP2, FP = charMatch(res, gt)
    return TP1, FN, FP


def eval_char_acc(gt_folder="./gt", rel_folder="./rel"):
    gt_files = os.listdir(gt_folder)
    rel_files = os.listdir(rel_folder)
    TP_total = 0
    FN_total = 0
    FP_total = 0
    for item in tqdm(gt_files):
        gt_txt = gt_folder + "/" + item
        rel_txt = rel_folder + "/" + item.replace("gt_", "")
        if not os.path.exists(rel_txt):
            # print(rel_txt+' not exists')
            rel = [""]
        else:
            rel_f = open(rel_txt)
            rel = rel_f.readlines()
            rel = filter(lambda x: x.strip(), rel)
        # print(rel_txt.split("/")[-1])
        with open(gt_txt) as gt_f:
            gt = gt_f.readlines()
            gt = filter(lambda x: x.strip(), gt)

            try:
                tp, fn, fp = trrsCompare(gt, rel)
            except Exception as e:
                print(e.args)
                continue
            TP_total += tp
            FN_total += fn
            FP_total += fp
        if (tp + fn) != 0 and (tp + fp) != 0:
            single_recall = tp / (tp + fn)
            single_precesion = tp / (tp + fp)
        if tp == 0:
            single_recall = 0
            single_precesion = 0
        if (single_recall + single_precesion) != 0:
            single_F = (single_recall * single_precesion * 2) / (
                single_recall + single_precesion
            )
        else:
            single_F = 0
        # print('%16s'%rel_txt.split("/")[-1]+'      recall: ' + '%0.4f' % single_recall + '      precesion: ' + '%0.4f' % single_precesion + '      F: ' + '%0.4f' % single_F)

    # print('TP: '+str(TP_total))
    # print('FN: '+str(FN_total))
    # print('FP: '+str(FP_total))
    precesion = TP_total / (TP_total + FP_total)
    recall = TP_total / (TP_total + FN_total)
    F = (recall * precesion * 2) / (recall + precesion)

    # print("\nfinal eval:")
    # print('total   recall: ' + '%0.4f' % recall \
    #                  +'     precesion: ' + '%0.4f' % precesion+ '       F: ' + '%0.4f' % F)

    return recall, precesion, F


def get_txt_file(imagepath):
    """
    find image files in test data path
    :return: list of files found
    """
    files = []
    exts = ["txt"]
    for parent, dirnames, filenames in os.walk(imagepath):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print("Find {} txt".format(len(files)))
    return files


def filter_txt(test_path, result_path):
    test_txt_list = get_txt_file(test_path)
    result_txt_list = get_txt_file(result_path)

    test = "./acc/T/"
    result = "./acc/R/"
    if os.path.exists(test):
        shutil.rmtree(test)
    os.makedirs(test)
    if os.path.exists(result):
        shutil.rmtree(result)
    os.makedirs(result)
    # if (not os.path.exists(test)):
    #     os.makedirs(test)
    # if (not os.path.exists(result)):
    #     os.makedirs(result)

    for test_txt in test_txt_list:
        dis_filter_path = test + test_txt.split("/")[-1]
        # print(dis_filter_path)
        stcmd_filter = (
            "cat " + test_txt + " | sed 's/.*,.*,.*,.*,.*,.*,//g'>" + dis_filter_path
        )
        subprocess.call(stcmd_filter, shell=True)

    for result_txt in result_txt_list:
        dis_filter_path = result + result_txt.split("/")[-1]
        stcmd_filter = (
            "cat " + result_txt + " | sed 's/.*,.*,.*,.*,.*,.*,//g'>" + dis_filter_path
        )
        subprocess.call(stcmd_filter, shell=True)

    return test, result  # 过滤掉前面的坐标信息


def filter_repeat(line, min_num=2, max_num=5, repeat_times=3):
    repeat_length = len(line) // repeat_times
    if repeat_length < max_num:
        max_num = repeat_length
    for slice_length in range(min_num, max_num):
        for index in range(0, len(line) - slice_length * repeat_times + 1):
            begin = index
            end = index + slice_length
            cut = line[begin:end]
            sign = True
            for i in range(0, repeat_times - 1):
                begin += slice_length
                end += slice_length
                # if cut=='ab':print(line[begin:end])
                if not cut == line[begin:end]:
                    sign = False
            if sign:
                # print('find:'+cut)
                pattern = r"(" + cut + ")" + "+"
                # print(pattern)
                line = re.sub(pattern, cut, line)
    # print(line)
    return line


def ch_is_alnum(ch):
    if 48 <= ord(ch) and ord(ch) <= 57:
        return True
    if 65 <= ord(ch) and ord(ch) <= 90:
        return True
    if 97 <= ord(ch) and ord(ch) <= 122:
        return True
    return False


def j2fconvert(fan_lst):
    jian_lst = []
    for fan_string in fan_lst:
        jan_string = Traditional2Simplified(fan_string)
        jian_lst.append(jan_string)
    return jian_lst


def filter_str(line, delete_repeat=True, delete_alnum=False, delete_common_char=True):
    line = line.strip()
    # print('\nori:   '+line)
    line = re.sub(r"\s+", " ", line)

    line = line.replace("，", ",")
    line = line.replace("。", ".")
    line = line.replace("？", "?")
    Set = set(",.? ")
    # filter other chars
    # print(line)
    # print(len(line))
    # line1 = u"我可从来没爷们儿过"
    # print(len(str(line1)))
    # for ch in line:

    #     if not u'\u4e00' <= ch <= u'\u9fff' and not ch.isalnum() and ch not in Set:
    #         line = line.replace(ch,'')
    if delete_common_char:
        # for ch in line:
        #     if not u'\u4e00' <= ch <= u'\u9fff' and not ch_is_alnum(ch):
        #         line = line.replace(ch,'')
        pass

    if delete_alnum:  # delete al and number
        for ch in line:
            if ch_is_alnum(ch):
                line = line.replace(ch, "")

    if delete_repeat:
        line = filter_repeat(line)
    # print('filter:'+line)
    return line


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        result_path = sys.argv[2]
    else:
        test_path = "../testdata/e2e/gt/"
        result_path = "../output/rec_output/"

    test, result = filter_txt(test_path, result_path)
    print("\nline eval : ")
    line_recall, line_precesion, line_F = eval_line_acc(
        gt_folder=test, rel_folder=result
    )
    print(
        "\n###########################################################################"
    )
    print("\nchar eval: ")
    world_recall, world_precesion, world_F = eval_char_acc(
        gt_folder=test, rel_folder=result
    )

    print("\nline final eval:")
    print(
        "total   recall: "
        + "%0.4f" % line_recall
        + "        precesion: "
        + "%0.4f" % line_precesion
        + "       F: "
        + "%0.4f" % line_F
    )

    print("\nworld final eval:")
    print(
        "total   recall: "
        + "%0.4f" % world_recall
        + "     precesion: "
        + "%0.4f" % world_precesion
        + "       F: "
        + "%0.4f" % world_F
    )

    # os.system('rm -rf ' + test)
    # os.system('rm -rf ' + result)
