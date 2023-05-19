#! /usr/bin/python3
# -*- coding: utf-8 -*-


# requirements:
# python3 -m pip install edit_distance numpy Polygon3


from glob import glob
import os

from edit_distance import edit_distance
import numpy as np
import Polygon as plg


def eval_detect(gt_dir, pred_dir):
    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
        resBoxes = np.empty([1, 8], dtype="int32")
        resBoxes[0, 0] = int(points[0])
        resBoxes[0, 4] = int(points[1])
        resBoxes[0, 1] = int(points[2])
        resBoxes[0, 5] = int(points[3])
        resBoxes[0, 2] = int(points[4])
        resBoxes[0, 6] = int(points[5])
        resBoxes[0, 3] = int(points[6])
        resBoxes[0, 7] = int(points[7])
        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg.Polygon(pointMat)

    def get_intersection(pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def get_union(pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)

    def get_intersection_over_union(gt_polygon, pred_polygon):
        try:
            return get_intersection(gt_polygon, pred_polygon) / get_union(
                gt_polygon, pred_polygon
            )
        except:
            return 0

    total_matched_num = 0
    total_gt_num = 0
    total_pred_num = 0
    total_char_acc = 0.0

    for gt_path in glob(os.path.join(gt_dir, "*.txt")):
        gt_polgons = []
        gt_texts = []
        pred_polgons = []
        pred_texts = []
        single_matched_num = 0
        single_char_acc = 0.0

        # read gt
        with open(gt_path, "r", encoding="utf-8") as gt:
            for line in gt.readlines():
                line = line.strip()
                if line == "":
                    continue
                line = line.split(",")
                points = line[:8]
                text = ",".join(line[8:])  # TODO: eval text recognition
                gt_polgons.append(polygon_from_points(points))
                gt_texts.append(text)
                total_gt_num += 1

        # read prediction
        base_name = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, base_name)
        if not os.path.exists(pred_path):
            print(f"{pred_path} does not exit!")
            continue
        with open(pred_path, "r", encoding="utf-8") as pred:
            for line in pred.readlines():
                line = line.strip()
                if line == "":
                    continue
                line = line.split(",")
                points = line[:8]
                text = ",".join(line[8:])  # TODO: eval text recognition
                pred_polgons.append(polygon_from_points(points))
                pred_texts.append(text)
                total_pred_num += 1

        # calculate precision, recall and f1
        if len(gt_polgons) > 0 and len(pred_polgons) > 0:
            gt_visit_flag = [False] * len(gt_polgons)
            pred_visit_flag = [False] * len(pred_polgons)
            for gt_index in range(len(gt_polgons)):
                if gt_visit_flag[gt_index]:
                    continue
                # find the best candidate with max iou
                max_iou, max_iou_index = -1, -1
                for pred_index in range(len(pred_polgons)):
                    if pred_visit_flag[pred_index]:
                        continue
                    iou = get_intersection_over_union(
                        gt_polgons[gt_index], pred_polgons[pred_index]
                    )
                    if iou > max_iou:
                        max_iou, max_iou_index = iou, pred_index
                if max_iou > evaluation_params["IOU_CONSTRAINT"]:
                    gt_visit_flag[gt_index] = True
                    pred_visit_flag[max_iou_index] = True
                    single_matched_num += 1
                    edit_dist = edit_distance(
                        gt_texts[gt_index], pred_texts[pred_index]
                    )[0]
                    single_char_acc += 1 - edit_dist / max(
                        len(gt_texts[gt_index]), len(pred_texts[pred_index]), 1
                    )
            total_matched_num += single_matched_num
            total_char_acc += single_char_acc
            # single_precision = single_matched_num / len(pred_polgons)
            # single_recall = single_matched_num / len(gt_polgons)
            # single_f1 = 2 * single_precision * single_recall / (single_precision + single_recall) if (single_precision + single_recall) > 0 else 0
            # print(f'{base_name} ===> "single_precision": {single_precision}, "single_recall": {single_recall}, "single_f1": {single_f1}, "single_char_acc": {single_char_acc / len(gt_polgons)}')
    precision = total_matched_num / total_pred_num if total_pred_num > 0 else 0
    recall = total_matched_num / total_gt_num if total_gt_num > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    char_acc = total_char_acc / total_gt_num
    print(f'Totally, \nDetect "precision": {precision}, "recall": {recall}, "f1": {f1}')
    # print(f'End-to-End Recognize "char_acc": {char_acc}')  # not correct


def rename():
    dirs = ["eval_v2/gt", "eval_v2/preds"]
    for dir in dirs:
        for f in os.listdir(dir):
            new_name = "_".join(f.split("_")[1:])
            os.rename(os.path.join(dir, f), os.path.join(dir, new_name))


if __name__ == "__main__":
    import sys

    evaluation_params = {"IOU_CONSTRAINT": 0.5}
    if len(sys.argv) > 1:
        gt_dir_ = sys.argv[1]
        preds_dir_ = sys.argv[2]
    else:
        gt_dir_ = "../../../testdata/e2e/gt"
        preds_dir_ = "inference_output/preds"
    print(gt_dir_, preds_dir_)
    eval_detect(gt_dir_, preds_dir_)
