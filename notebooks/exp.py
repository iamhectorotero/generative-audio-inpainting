from multiprocessing import Pool

import pickle as pk
import matplotlib.pyplot as plt
import importlib
import numpy as np
import torch
import resource
import sys

from torch.utils.data import DataLoader

from rml import training
from rml import utils
from rml.utils import plotting

from mlp import audio
from mlp import normalization
from mlp import utils as mlp
from mlp.dataset import WAVAudioDS, PolarPreprocessing, Pipeline, MultiSet

from unet.unet.unet_parts import *


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 1)

    def forward(self, inn):
        x1 = self.inc(inn)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        return x
    
    
fs = 48000
bs = 64
stroke_width = int(sys.argv[1])
patch_width = int(sys.argv[2])
patch_height = 64

files_train = pk.load(open("train.pk", "rb"))
files_valid = pk.load(open("valid.pk", "rb"))

device = utils.utils.device

print(f"Starting {stroke_width}_{patch_width}")

resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
utils.utils.set_is_notebook(False)

stroke_mask = mlp.build_stroke_purge_mask(patch_width, patch_height, stroke_width, fs, channels=1)
stroke_mask_not = ~stroke_mask

purge_mask = stroke_mask.float()
keep_mask = stroke_mask_not.float().to(device)

preprocess = PolarPreprocessing(
    normalization.norm_mag, 
    normalization.norm_phase, 
    patch_width, 
    patch_height,
    include_phase=False
)

print("Loading dataset")

with Pool(8) as p:
    train_dss = []
    
    for i in range(9):
        train_dss.append(WAVAudioDS(files_train[i*4000:(i+1)*4000], lambda x: x * purge_mask, preprocess, patch_width, p))

    ds_train = MultiSet(train_dss)
    ds_test = WAVAudioDS(files_valid, lambda x: x * purge_mask, preprocess, patch_width, p, random_patches=False)

dl_train = DataLoader(ds_train, batch_size=bs, num_workers=8, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=bs, num_workers=8, shuffle=False)

mse = torch.nn.MSELoss()

def MSEGapLoss(pred, targ):
    pred_gap = pred * keep_mask
    targ_gap = targ * keep_mask
    
    return mse(pred_gap[pred_gap != 0], targ_gap[targ_gap != 0])
    
model = UNet().to(device)
optim = torch.optim.Adam(model.parameters())

print(f"Starting optimization")

train_data = training.optimize(16, model, optim, MSEGapLoss, dl_train, dl_test, 
                               store_pt=False, epoch_save_path=f"gap_context/{stroke_width}_{patch_width}", 
                               no_epoch_info_bar=True)

print(f"Finished optimization")

torch.save(train_data, f"gap_context/{stroke_width}_{patch_width}_data_full.pt")
torch.save(model.state_dict(), f"gap_context/{stroke_width}_{patch_width}_model_final.pt")