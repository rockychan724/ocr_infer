from collections import OrderedDict
import torch

from DB_resnet_model import BasicModel


def convert_high_version_pytorch_to_low():
    model = BasicModel()
    state = torch.load("data/weights/model_epoch_125_minibatch_46000.pth", map_location=torch.device("cuda"))
    new_state = OrderedDict()
    for k, v in state.items():
        name = k[13:]
        new_state[name] = v
    model.load_state_dict(new_state)
    # model.load_state_dict(state, False)
    torch.save(model.state_dict(), "data/weights/model_epoch_125_minibatch_46000_unzipped.pth", _use_new_zipfile_serialization=False)

if __name__ == "__main__":
    convert_high_version_pytorch_to_low()
