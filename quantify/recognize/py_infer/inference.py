import math
import os
import time

import cv2
import numpy as np
import torch
from edit_distance import edit_distance

from crnn import CRNN_OCR_for_cnc_trt


class InferenceWrapper:
    def __init__(self, height, width, dict_path, device) -> None:
        self.height = height
        self.width = width
        self.device = device
        self.word_dict = ["blank"]
        for ch in open(dict_path, "r", encoding="utf-8").readlines():
            ch = ch.replace("\n", "")
            self.word_dict.append(ch)
        self.word_dict.append("UNKNOWN")

    def preprocess(self, img_paths):
        x_batch = np.zeros([len(img_paths), self.height, self.width, 1])
        for i, img_path in enumerate(img_paths):
            temp_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h, w = temp_img.shape
            if h > 1.5 * w:  # It's a vertical image if h > 1.5*w
                temp_img = np.rot90(temp_img, 1)
                h, w = w, h
            hnew = self.height
            wnew = int(1.0 * hnew / h * w)
            if self.width > wnew:
                temp_img = cv2.resize(
                    temp_img, (wnew, hnew), interpolation=cv2.INTER_CUBIC
                )
                x_batch[i, :, :wnew, 0] = temp_img[:, :]
            else:
                temp_img = cv2.resize(
                    temp_img, (self.width, self.height), interpolation=cv2.INTER_CUBIC
                )
                x_batch[i, :, :, 0] = temp_img[:, :]
        return x_batch

    def inference(self, model, x_batch_in):
        x_batch = (
            torch.from_numpy(np.array(x_batch_in))
            .permute(0, 3, 1, 2)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        start_time = time.time()
        preds = model(x_batch)
        if self.device == torch.device("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()
        preds = preds.cpu().type(torch.IntTensor).numpy()
        texts = self.decode(preds)
        return texts, end_time - start_time

    def decode(self, preds):
        b, n = preds.shape
        texts = []
        for pred in preds:
            text = ""
            for i in range(n):
                if pred[i] <= 0 or (i > 0 and pred[i] == pred[i - 1]):
                    continue
                text += self.word_dict[pred[i]]
            texts.append(text)
        return texts


def calculate_accuracy(labels, preds, p_list, save_file):
    correct_num = 0
    norm_edit_dist = 0.0
    save_info = {}
    for i, (l, p) in enumerate(zip(labels, preds)):
        label = l.replace(" ", "")
        pred = p.replace(" ", "")
        pred = pred.replace("UNKNOWN", "")
        edit_dist = edit_distance(label, pred)[0]
        if edit_dist == 0:
            correct_num += 1
        norm_edit_dist += edit_dist / max(len(label), len(pred), 1)
        save_info[p_list[i]] = [edit_dist, l, p]

    # save detail
    if save_file is not None:
        save_info = dict(sorted(save_info.items(), key=lambda kv: kv[1][0]))
        with open(save_file, "w", encoding="utf-8") as f_w:
            for k, v in save_info.items():
                f_w.write("{} ed:{} gt:{} pred:{}\n".format(k, v[0], v[1], v[2]))

    total_num = len(labels)
    return (
        correct_num / total_num,
        1 - norm_edit_dist / total_num,
    )  # line acc and char acc


def evaluate():
    inference_wrapper = InferenceWrapper(
        opt.image_height, opt.image_width, opt.dict_file, device
    )

    # read label
    img_path = []
    img_label = []
    with open(opt.testset_gt, "r", encoding="utf-8") as f_r:
        for line in f_r.readlines():
            line = line.rstrip("\n")
            seg = line.split(" ")
            img_path.append(os.path.join(opt.testset_dir, seg[0]))
            img_label.append(" ".join(seg[1:]))

    ocr_model = CRNN_OCR_for_cnc_trt(opt.input_channel, opt.class_number)
    ocr_model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model = ocr_model.to(device)

    # inference
    batch_num = math.ceil(len(img_label) / opt.batch_size)
    sum_gt = []
    sum_pred = []
    batch_time_lst = []
    p_list = []
    time_count = 0
    for i in range(batch_num):
        x_batch = inference_wrapper.preprocess(
            img_path[i * opt.batch_size : (i + 1) * opt.batch_size]
        )
        rst, batch_time = inference_wrapper.inference(model, x_batch)
        sum_gt.extend(img_label[i * opt.batch_size : (i + 1) * opt.batch_size])
        sum_pred.extend(rst)
        batch_time_lst.append(batch_time)
        time_count += len(x_batch)
        p_list.extend(img_path[i * opt.batch_size : (i + 1) * opt.batch_size])
        print(
            "{}th/{} batch, inference time: {} ms, average time: {} ms/image".format(
                i + 1, batch_num, batch_time * 1000, batch_time * 1000 / len(x_batch)
            )
        )

    # calculate accuracy and save result
    line_acc, char_acc = calculate_accuracy(
        sum_gt, sum_pred, p_list, os.path.join(opt.output_dir, "prediction.txt")
    )
    speed = sum(batch_time_lst) / time_count * 1000  # ms/image

    with open(
        os.path.join(opt.output_dir, "eval_result.txt"), "w", encoding="utf-8"
    ) as f:
        save_info = f"line_acc: {line_acc}\nchar_acc: {char_acc}\nspeed: {speed} ms/image <=> {1000/speed} FPS\n"
        print(save_info)
        f.write(save_info)


if __name__ == "__main__":
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    # MLU model config
    parser.add_argument(
        "--model_path",
        type=str,
        default="../../weights/torch_weight/crnn.pth",
        help="the path quantized model",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument(
        "--use_gpu", type=str2bool, default=True, help="whether to use GPU"
    )

    # network config
    parser.add_argument(
        "--input_channel", type=int, default=1, help="input channel number"
    )
    parser.add_argument("--class_number", type=int, default=10910, help="class number")
    parser.add_argument("--image_height", type=int, default=48, help="image height")
    parser.add_argument("--image_width", type=int, default=480, help="image width")

    # other config
    parser.add_argument(
        "--dict_file",
        type=str,
        default="../../../data/rec_dict/dict_cjke.txt",
        help="the path to testset",
    )
    parser.add_argument(
        "--testset_dir",
        type=str,
        default="../../../testdata/rec/sub_mixed_test",
        help="the path to testset",
    )
    parser.add_argument(
        "--testset_gt",
        type=str,
        default="../../../testdata/rec/sub_mixed_test.txt",
        help="the path to testset gt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_output",
        help="the path to eval output",
    )
    opt = parser.parse_args()

    device = torch.device(
        "cuda" if opt.use_gpu and torch.cuda.is_available() else "cpu"
    )
    print(f"****** use {opt.model_path}")

    os.makedirs(opt.output_dir, exist_ok=True)

    with torch.no_grad():
        evaluate()
