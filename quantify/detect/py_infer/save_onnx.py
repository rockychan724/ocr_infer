from collections import OrderedDict

import torch.onnx
import onnx

# from DB_model_mobilenet import BasicModel_for_trt

from DB_model_resnet import BasicModel_for_trt


def save_onnx():
    src_pth = "../../weights/torch_weight/db_mobilenet.pth"
    dst_onnx = "../../weights/onnx/db_mobilenet.onnx"
    model = BasicModel_for_trt().cuda()
    state = torch.load(src_pth)
    new_state = OrderedDict()
    for k, v in state.items():
        name = k[13:]
        new_state[name] = v
    model.load_state_dict(new_state)
    model.eval()
    dummy_input = torch.zeros((1, 3, 512, 512)).cuda()
    torch.onnx.export(
        model,
        dummy_input,
        dst_onnx,
        export_params=True,
        training=False,
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={"inputs": {0: "batch"}, "outputs": {0: "batch"}},
        opset_version=11,
    )
    omodel = onnx.load(dst_onnx)
    onnx.checker.check_model(omodel)


if __name__ == "__main__":
    save_onnx()
