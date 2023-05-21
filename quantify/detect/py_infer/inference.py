# ! /usr/bin/python
# -*- coding: utf-8 -*-
import math
import os

from collections import OrderedDict
from glob import glob
import time

import cv2
import numpy as np
import torch

# from DB_model_mobilenet import BasicModel

from DB_model_resnet import BasicModel
from DB_postprocess import SegDetectorRepresenter


def format_output(batch, output):
    batch_boxes, batch_scores = output
    for index in range(len(batch["filename"])):
        filename = batch["filename"][index]
        result_file_name = filename.split("/")[-1].split(".")[0] + ".txt"
        result_file_path = os.path.join(opt["output_dir"], "preds", result_file_name)
        boxes = batch_boxes[index]
        scores = batch_scores[index]
        with open(result_file_path, "wt") as res:
            for i in range(boxes.shape[0]):
                score = scores[i]
                if score < opt["box_thresh"]:
                    continue
                box = boxes[i, :, :].reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                # res.write(result + "," + str(score) + "\n")
                res.write(result + "\n")


def demo_visualize(image_path, output):
    boxes, _ = output
    # boxes = boxes[0]
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_shape = original_image.shape
    pred_canvas = original_image.copy().astype(np.uint8)
    pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

    for box in boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)

    return pred_canvas


def do_infer():
    model = BasicModel()

    # if torch version >= 1.6
    state = torch.load(opt["model_path"], map_location=opt["device"])
    new_state = OrderedDict()
    for k, v in state.items():
        name = k[13:]
        new_state[name] = v
    model.load_state_dict(new_state)
    # if torch versuib < 1.6
    # state = torch.load(opt["model_path"], map_location=opt["device"])
    # model.load_state_dict(state)

    model = model.to(opt["device"])

    pp = SegDetectorRepresenter(opt)

    img_paths = glob("../../../testdata/e2e/image/*.jpg")
    batch_size = opt["batch_size"]
    batch_num = math.ceil(len(img_paths) / batch_size)
    total_inference_time = 0
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    for i in range(batch_num):
        batch = {"filename": [], "image": None, "shape": []}
        images = []
        for img_path in img_paths[i * batch_size : (i + 1) * batch_size]:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype("float32")
            batch["filename"].append(img_path)
            batch["shape"].append(img.shape[:2])
            img = cv2.resize(img, (opt["image_height"], opt["image_width"]))
            img = (img - RGB_MEAN) / 255.0
            images.append(img)
        batch_image = (
            torch.from_numpy(np.array(images))
            .permute(0, 3, 1, 2)
            .type(torch.FloatTensor)
            .to(opt["device"])
        )
        print(
            "batch_image:",
            batch_image.shape,
            batch_image.dtype,
            batch_image.device.type,
        )
        start = time.time()
        preds = model(batch_image)
        if opt["device"] == torch.device("cuda"):
            torch.cuda.synchronize()
        batch_time = time.time() - start
        total_inference_time += batch_time
        print(
            "{}th/{} batch, inference time: {} ms, average time: {} ms/image".format(
                i + 1, batch_num, batch_time * 1000, batch_time * 1000 / len(images)
            )
        )
        preds = preds.detach().cpu().type(torch.FloatTensor).numpy()
        # print(preds[0][0][0])
        # print(preds.dtype)
        # print(preds.shape)

        start = time.time()
        output = pp.represent(batch, preds)
        batch_time = time.time() - start
        print(
            "    {}th/{} batch, postprocess time: {} ms, average time: {} ms/image".format(
                i + 1, batch_num, batch_time * 1000, batch_time * 1000 / len(images)
            )
        )
        format_output(batch, output)
        for j, img_path in enumerate(batch["filename"]):
            core_name = os.path.splitext(os.path.basename(img_path))[0]
            if opt["debug"]:
                np.savetxt(
                    os.path.join(opt["output_dir"], "probs", core_name + ".txt"),
                    preds[j][0],
                )  # cost much time and storage space
            vis_image = demo_visualize(img_path, (output[0][j], output[1][j]))
            cv2.imwrite(
                os.path.join(opt["output_dir"], "vis", core_name + ".jpg"), vis_image
            )
    speed = total_inference_time * 1000 / (batch_size * batch_num)
    print("inference speed: {} ms/image <=> {} fps".format(speed, 1000 / speed))


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
    parser.add_argument(
        "--model_path", type=str, default="", help="the path quantized model"
    )
    parser.add_argument("--debug", type=str2bool, default=False, help="debug")
    parser.add_argument("--box_thresh", type=float, default=0.5, help="box thresh")

    parser.add_argument("--batch_size", type=int, default=50, help="batch size")

    parser.add_argument(
        "--output_dir", type=str, default="inference_output", help="save all output"
    )
    parser.add_argument("--image_height", type=int, default=512, help="image height")
    parser.add_argument("--image_width", type=int, default=512, help="image width")

    opt = vars(parser.parse_args())

    os.makedirs(opt["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(opt["output_dir"], "preds"), exist_ok=True)
    os.makedirs(os.path.join(opt["output_dir"], "vis"), exist_ok=True)
    if opt["debug"]:
        os.makedirs(os.path.join(opt["output_dir"], "probs"), exist_ok=True)

    opt["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt["model_path"] = "../../weights/torch_weight/db_resnet.pth"
    print(f"****** use {opt['model_path']}")

    with torch.no_grad():
        do_infer()
