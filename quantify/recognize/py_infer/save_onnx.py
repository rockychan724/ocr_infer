import torch
import onnx
import yaml
from crnn import CRNN_OCR_for_cnc_trt

if __name__ == "__main__":
    src_pth = "../../data/torch_weight/crnn.pth"
    dst_onnx = "../../data/onnx/crnn.onnx"
    config = {
        "model_path": src_pth,
        "dict_path": "../testdata/dict_cjke.txt",
        "batch_size": 32,
        "input_channel": 1,
        "class_number": 10910,
        "image_height": 48,
        "image_width": 480,
    }
    device = torch.device("cuda")
    model = CRNN_OCR_for_cnc_trt(config["input_channel"], config["class_number"]).to(
        device
    )
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    x = torch.randn((1, 1, 48, 480)).to(device)
    torch.onnx.export(
        model,
        x,
        dst_onnx,
        export_params=True,
        training=False,
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}},
        opset_version=10,
    )
    onnxmodel = onnx.load(dst_onnx)
    onnx.checker.check_model(onnxmodel, full_check=True)
