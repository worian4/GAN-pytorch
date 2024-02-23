import asyncio
import functools
import itertools
import math
import random

import os
import numpy as np

from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from torchvision import transforms
import torch
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from skimage import io
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch import nn
import torch.optim as optim
from torchvision.utils import save_image
import torchvision.utils as vutils

import models

manualSeed = random.randint(1, 10000)
print("\nRandom Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

nz = 100
image_size = 64
ngpu = 1
device = torch.device("cuda:0")
n_extra_layers_g = 1

netG = models._netG_1(ngpu, nz, 3, image_size, n_extra_layers_g).to(device)
netG.apply(weights_init)
netG.load_state_dict(torch.load("pth_weights_direction"))
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
with torch.no_grad():
    res = netG(fixed_noise).detach().cpu()
save_image(vutils.make_grid(res, padding=2, normalize=True), 'output.png')
