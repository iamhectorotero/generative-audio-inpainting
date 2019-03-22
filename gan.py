import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import pickle
from multiprocessing import Pool
import resource

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from pix2pix.pix2pix_model import *
from unet.unet.unet_parts import *
from mlp import audio
from mlp import normalization
from mlp import utils as mlp
from mlp.dataset import WAVAudioDS, PolarPreprocessing, Pipeline, MultiSet

import torch.nn as nn
import torch.nn.functional as F
import torch

epoch = 0 # epoch to start training from
n_epochs = 30 # number of epochs of training
dataset_name = 'VCTK' # name of the dataset
batch_size = 4 # size of the batches
lr = 0.0002 # adam: learning rate
b1 = 0.5 # adam: decay of first order momentum of gradient
b2 = 0.999 # adam: decay of first order momentum of gradient
decay_epoch = 100 # epoch from which to start lr decay
n_cpu = 4 # number of cpu threads to use during batch generation
img_height = 64 # size of image height
img_width = 64 # size of image width
channels = 1 # number of image channels
sample_interval = 20000 # interval between sampling of images from generators
checkpoint_interval = 1 # interval between model checkpoints

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 10

# Calculate output of image discriminator (PatchGAN)
patch = (1, img_height//2**4, img_width//2**4)

class UNet(nn.Module):
    def __init__(self, leaky_relu=False, dropout=0.0):
        super(UNet, self).__init__()
        self.inc = inconv(1, 64, leaky_relu=leaky_relu)
        self.down1 = down(64, 128, leaky_relu=leaky_relu)
        self.down2 = down(128, 256, leaky_relu=leaky_relu)
        self.down3 = down(256, 512, leaky_relu=leaky_relu, dropout=dropout)
        self.down4 = down(512, 512, leaky_relu=leaky_relu, dropout=dropout)
        self.up1 = up(1024, 256, bilinear=True, leaky_relu=leaky_relu, dropout=dropout)
        self.up2 = up(512, 128, bilinear=True, leaky_relu=leaky_relu, dropout=dropout)
        self.up3 = up(256, 64, bilinear=True, leaky_relu=leaky_relu)
        self.up4 = up(128, 64, bilinear=True, leaky_relu=leaky_relu)
        self.outc = outconv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
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
    
pretrained_generator = UNet(leaky_relu=False, dropout=0.0).to(device)
generator = UNet(leaky_relu=True, dropout=0.5).to(device)
discriminator = Discriminator(in_channels=channels).to(device)

if cuda:
    generator = generator.cuda()
    pretrained_generator = pretrained_generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if cuda:
    pretrained_generator.load_state_dict(torch.load('32_64_model_final.pt'))
else:
    pretrained_generator.load_state_dict(torch.load('32_64_model_final.pt', map_location='cpu'))
    
if epoch != 0:
    # Load pretrained models
    results = pickle.load(open("results.pkl", "rb"))
    accuracies = pickle.load(open("accuracies.pkl", "rb"))
    generator.load_state_dict(torch.load('generator_%d.pth' % (epoch)))
    discriminator.load_state_dict(torch.load('discriminator_%d.pth' % (epoch)))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_G.load_state_dict(torch.load('optimizerG_%d.pth' % (epoch)))
    optimizer_D.load_state_dict(torch.load('optimizerD_%d.pth' % (epoch)))
else:
    # Initialize weights
    #generator.apply(weights_init_normal)
    results = []
    accuracies = []
    if cuda:
        generator.load_state_dict(torch.load('32_64_model_final.pt'))
    else:
        generator.load_state_dict(torch.load('32_64_model_final.pt', map_location='cpu'))
    discriminator.apply(weights_init_normal)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

fs = 48000
bs = batch_size
stroke_width = 32
patch_width = img_width
patch_height = img_height
nperseg = 256

train_files = pickle.load(open("train.pk", "rb"))
val_files = pickle.load(open("valid.pk", "rb"))

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
        print('Progress:', i)
        train_dss.append(WAVAudioDS(train_files[i*4000:(i+1)*4000], mk_source=lambda x: x * purge_mask, 
                                    preprocess=preprocess, patch_width=patch_width, proc_pool=p, 
                                    nperseg=256, random_patches=True))

    ds_train = MultiSet(train_dss)
    ds_test = WAVAudioDS(val_files, mk_source=lambda x: x * purge_mask, preprocess=preprocess, 
                          patch_width=patch_width, proc_pool=p, nperseg=256, random_patches=False)

val_dataloader = DataLoader(ds_test, batch_size=bs, num_workers=8, shuffle=False)
dataloader = DataLoader(ds_train, batch_size=bs, num_workers=8, shuffle=True)

print('Loading data done')

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(epoch, batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs[0].type(Tensor))
    real_B = Variable(imgs[1].type(Tensor))

    recon_B = generator(real_A.to(device)).detach()
    recon_B = real_A + keep_mask*recon_B
    recon_B_old = pretrained_generator(real_A.to(device)).detach()
    recon_B_old = real_A + keep_mask*recon_B_old
    
    img_sample = torch.cat((real_A.detach(), recon_B.detach(), recon_B_old.detach(), real_B.detach()), -2)
    torch.save(img_sample, '%s_%s.pt' % (epoch, batches_done))
    
    
print('Start training')

prev_time = time.time()
for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        real_A = Variable(batch[0].type(Tensor)) # Gap
        real_B = Variable(batch[1].type(Tensor)) # Original (No gap)
        # Adversarial ground truths
        valid = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        
        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        noise = np.random.uniform()
        noise_thresh = 0.05*np.exp(-epoch*0.05)
        if noise < noise_thresh:
            valid_soft = Variable(Tensor(np.random.uniform(0.85, 1.0, size=(real_A.size(0), *patch))), requires_grad=False)
            fake_soft = Variable(Tensor(np.random.uniform(0.0, 0.15, size=(real_A.size(0), *patch))), requires_grad=False)
        else:
            valid_soft = Variable(Tensor(np.random.uniform(0.0, 0.15, size=(real_A.size(0), *patch))), requires_grad=False)
            fake_soft = Variable(Tensor(np.random.uniform(0.85, 1.0, size=(real_A.size(0), *patch))), requires_grad=False)

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid_soft)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake_soft)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        correct = 0
        correct = torch.sum(pred_real.view(bs, -1).mean(1, keepdim=True)<0.5).item()
        correct += torch.sum(pred_fake.view(bs, -1).mean(1, keepdim=True)>0.5).item()
        accuracies.append(correct/(bs*2))
        acc_thresh = 0.85

        if np.mean(accuracies[-20:]) < acc_thresh:
            loss_D.backward()
            optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" %
                                                        (epoch, n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_pixel.item(), loss_GAN.item(),
                                                        time_left))
        results.append((epoch, loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item()))
        if batches_done % 10000 == 0:
            with open('results.pkl', 'wb') as fp:
                pickle.dump(results, fp)
            with open('accuracies.pkl', 'wb') as fp:
                pickle.dump(accuracies, fp)

        # If at sample interval save image
        if batches_done % sample_interval == 0:
            sample_images(epoch, batches_done)

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'generator_%d.pth' % (epoch))
        torch.save(discriminator.state_dict(), 'discriminator_%d.pth' % (epoch))
        torch.save(optimizer_G.state_dict(), 'optimizerG_%d.pth' % (epoch))
        torch.save(optimizer_D.state_dict(), 'optimizerD_%d.pth' % (epoch))

with open('results.pkl', 'wb') as fp:
    pickle.dump(results, fp)
    
print('Training done')
