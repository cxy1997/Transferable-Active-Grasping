import logging
import matplotlib.pyplot as plt
import numpy as np
from functools import partial


def load_snapshot(snapshot_file, depth_fusion):
    """Load a training snapshot"""
    print("--- Loading model from snapshot")

    # Create network
    norm_act = partial(InPlaceABN, activation="leaky_relu", slope=.01)
    model_dict = dict()
    if depth_fusion == 'no-depth':
        model_dict['body'] = models.__dict__["net_wider_resnet38_a2"](
            norm_act=norm_act,
            dilation=(1, 2, 4, 4)
        )
    elif depth_fusion == 'pixel-concat':
        model_dict['body'] = models.__dict__["net_wider_resnet38_a2"](
            norm_act=norm_act,
            dilation=(1, 2, 4, 4),
            channels_in=4
        )
    elif depth_fusion == 'feature-concat':
        model_dict['body'] = models.__dict__["net_wider_resnet38_a2"](
            norm_act=norm_act,
            dilation=(1, 2, 4, 4)
        )
        model_dict['depth_body'] = models.__dict__["net_wider_resnet38_a2"](
            norm_act=norm_act,
            dilation=(1, 2, 4, 4),
            channels_in=1
        )
    if depth_fusion == 'feature-concat':
        model_dict['head'] = DeeplabV3(8192, 256, 256, norm_act=norm_act, pooling_size=(84, 84))
    else:
        model_dict['head'] = DeeplabV3(4096, 256, 256, norm_act=norm_act, pooling_size=(84, 84))

    # Load snapshot and recover network state
    data = torch.load(snapshot_file)
    if depth_fusion == 'feature-concat' or depth_fusion == 'no_depth':
        model_dict['body'].load_state_dict(data["state_dict"]["body"])

    if depth_fusion == 'pixel-concat' or depth_fusion == 'no_depth':
        model_dict['head'].load_state_dict(data["state_dict"]["head"])

    return model_dict


def setup_logger(logger_name, log_file, level=logging.INFO, verbose=False):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    if verbose:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        l.addHandler(streamHandler)
    return l


if __name__ == '__main__':
    pass