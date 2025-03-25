import os
import sys

import torch.nn as nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../"))

def group_norm(channel_num, group_num=32):
    assert channel_num%group_num==0, "Invalid parameters for group normalization!"
    return nn.GroupNorm(group_num, channel_num)

