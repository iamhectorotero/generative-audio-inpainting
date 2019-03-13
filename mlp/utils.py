import os
import math
import operator as op

import torch

from itertools import accumulate
from functools import reduce


def get_all_files(root):
    if os.path.isfile(root):
        return [root]
    else:
        return reduce(op.add, map(lambda c: get_all_files(root + "/" + c), os.listdir(root)), [])

    
def build_stroke_purge_mask(patch_width, patch_height, ms, fs, nperseg=256, channels=2):
    pixels = math.floor(ms * (1 / 500) * (fs / nperseg))
    left_offset = patch_width // 2 - pixels // 2
    
    mask = torch.ones((channels, patch_height, patch_width), dtype=torch.uint8)
    mask[:,:, left_offset:left_offset + pixels] = 0
    
    return mask


def acc_to_idx(acc, value):
    for i, v in enumerate(acc):
        if value < v:
            return i
        
    return len(acc) - 1