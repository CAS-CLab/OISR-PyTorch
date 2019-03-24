from model import common

import torch.nn as nn
import torch

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        # self.url = url['r{}f{}x{}'.format(n_resblocks, n_feats, scale)]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        self.m_body0 = common.ResBlock(
                    conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
        self.m_downsample0 = conv(n_feats, n_feats//4, 1)
        self.m_body1 = common.ResBlock(
                    conv, n_feats//4, kernel_size, act=act, res_scale=args.res_scale)
        self.m_downsample1 = conv(n_feats//4, n_feats, 1)
        # self.tail1 = conv(n_feats//4, n_feats//16, kernel_size)
        self.m_body2 = common.ResBlock(
                    conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
        self.m_downsample2 = conv(n_feats, n_feats//4, 1)
        self.m_body3 = common.ResBlock(
                    conv, n_feats//4, kernel_size, act=act, res_scale=args.res_scale)
        self.m_downsample3 = conv(n_feats//4, n_feats, 1)
        # self.tail3 = conv(n_feats//4, n_feats//16, kernel_size)
        self.m_body4 = common.ResBlock(
                    conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
        self.m_downsample4 = conv(n_feats, n_feats//4, 1)
        self.m_body5 = common.ResBlock(
                    conv, n_feats//4, kernel_size, act=act, res_scale=args.res_scale)
        self.m_downsample5 = conv(n_feats//4, n_feats, 1)
        # self.tail5 = conv(n_feats//4, n_feats//16, kernel_size)

        m_body = [conv(n_feats, n_feats, kernel_size)]

        # define tail module
        m_tail = [
            # common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        # self.refine = conv(n_feats//4, args.n_colors, kernel_size)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        body0 = self.m_body0(x)
        m_downsample0 = self.m_downsample0(body0)
        body1 = self.m_body1(m_downsample0)
        m_downsample1 = self.m_downsample1(body1)
        # m_tail1 = self.tail1(body1)
        body2 = self.m_body2(m_downsample1+body0)
        m_downsample2 = self.m_downsample2(body2)
        body3 = self.m_body3(m_downsample2+m_downsample0)
        m_downsample3 = self.m_downsample3(body3)
        # m_tail3 = self.tail1(body3)
        body4 = self.m_body4(m_downsample3+m_downsample1)
        m_downsample4 = self.m_downsample4(body4)
        body5 = self.m_body5(m_downsample4+m_downsample2+m_downsample0)
        m_downsample5 = self.m_downsample5(body5)
        # m_tail5 = self.tail1(body5)
        res = self.body(m_downsample5+m_downsample3+m_downsample1)
        res += x

        y = self.tail(res)
        # y = torch.cat([x, m_tail1, m_tail3, m_tail5], 1)
        # y = self.refine(x)
        y = self.add_mean(y)

        return y

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

