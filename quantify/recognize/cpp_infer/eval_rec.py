# -*- coding: utf-8 -*-

from edit_distance import edit_distance


def calculate_accuracy(labels, preds):
    correct_num = 0
    norm_edit_dist = 0.0
    save_info = {}
    for i, (l, p) in enumerate(zip(labels, preds)):
        label = l.replace(" ", "")
        pred = p.replace(" ", "")
        pred = pred.replace("UNKNOWN", "")
        # print(f"gt:{label} pred:{pred}")
        edit_dist = edit_distance(label, pred)[0]
        if edit_dist == 0:
            correct_num += 1
        norm_edit_dist += edit_dist / max(len(label), len(pred), 1)
    total_num = len(labels)
    # return line acc and char acc
    return (
        correct_num / total_num,
        1 - norm_edit_dist / total_num,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        gt_file = sys.argv[1]
        pred_file = sys.argv[2]
    else:
        gt_file = "../../../testdata/rec/sub_mixed_test.txt"
        pred_file = "inference_output/rec_result.txt"
    gt_list = []
    pred_list = []
    with open(gt_file, "r", encoding="utf-8") as f1, open(
        pred_file, "r", encoding="utf-8"
    ) as f2:
        image_names = []
        for line in f1.readlines():
            line_splits = line.strip().split(" ")
            image_names.append(line_splits[0])
            gt_list.append(" ".join(line_splits[1:]))

        temp_dict = {}
        for line in f2.readlines():
            line_splits = line.strip().split(" ")
            temp_dict[line_splits[0]] = " ".join(line_splits[1:])
        # sorted_dict = dict(sorted(temp_dict.items(), key=lambda kv: image_names.index(kv[0])))
        # pred_list = list(sorted_dict.values())
        for i, k in enumerate(image_names):
            if k in temp_dict.keys():
                print(f"{k} gt:{gt_list[i]} pred:{temp_dict[k]}")
                pred_list.append(temp_dict[k])
            else:
                print(f"{k} gt:{gt_list[i]}")
    print(len(gt_list), len(pred_list))
    assert len(gt_list) == len(pred_list)
    line_acc, char_acc = calculate_accuracy(gt_list, pred_list)
    print(f"line_acc: {line_acc} char_acc: {char_acc}\n")
