# -*- coding: UTF-8 -*-
from __future__ import print_function

import torch
import torch.nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import torch.backends.cudnn as cudnn
from PIL import Image
from miscc.utils import mkdir_p

import pickle
import random
import os
import numpy as np
from model import G_NET
from miscc.config import cfg, cfg_from_file

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def save_singleimages(images, save_dir):
    for i in range(images.size(0)):
        s_tmp = '%s/bird' %\
            (save_dir)
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            mkdir_p(folder)

        fullpath = '%s_%d.png' % (s_tmp, i)
        # range from [-1, 1] to [0, 255]
        img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
        ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(fullpath)


cfg_from_file('./stackGAN_code/cfg/eval_birds.yml')
save_dir = './display'
txt_dir = './embeddings/txt_embedding.t7'
manualSeed = random.randint(1, 120)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

gpus = [0]
num_gpus = 1
torch.cuda.set_device(gpus[0])
cudnn.benchmark = True
batch_size = cfg.batch_size

netG = G_NET()
netG.apply(weights_init)
netG = torch.nn.DataParallel(netG, device_ids=gpus)
state_dict = \
    torch.load(cfg.TRAIN.NET_G,
               map_location=lambda storage, loc: storage)
netG.load_state_dict(state_dict)

nz = cfg.GAN.Z_DIM
noise = Variable(torch.FloatTensor(batch_size, nz))

netG.cuda()
netG.eval()
noise = noise.cuda()

t_embeddings = load_lua(txt_dir)
t_embeddings = t_embeddings.view(-1,1024)

t_embeddings = t_embeddings.repeat(4,1)
t_embeddings = Variable(t_embeddings).cuda()

embedding_dim = t_embeddings.size(1)

noise.data.resize_(batch_size, nz)
noise.data.normal_(0, 1)
images, _, _ = netG(noise, t_embeddings)
save_singleimages(images[-1], save_dir)
