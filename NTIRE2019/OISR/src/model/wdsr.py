import math

from model import common

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def make_model(args, parent=False):
    return WDSR(args)

class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, block_feats, wn, res_scale=1, act=nn.ReLU(True)):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        body.append(
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

class WDSR(nn.Module):
    def __init__(self, args):
        super(WDSR, self).__init__()
        # hyper-params
        self.args = args
        scale = args.scale[0]
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [args.r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1])

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(args.n_colors, n_feats, 3, padding=3//2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, args.block_feats, wn=wn, res_scale=args.res_scale, act=act))

        # define tail module
        tail = []
        out_feats = scale*scale*args.n_colors
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))
        # tail.append(nn.PixelShuffle(scale))

        skip = []
        skip.append(
            wn(nn.Conv2d(args.n_colors, out_feats, 5, padding=5//2))
        )
        # skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        x = (x - self.rgb_mean.cuda()*255)/127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x*127.5 + self.rgb_mean.cuda()*255
        return x
