import argparse
import os
import platform
import sys
from copy import deepcopy
from einops import rearrange, repeat
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from models.grit import Transformer as grit_transformer
from utils.caption.caption_utils import get_end_token, create_caption_and_mask, create_src_mask
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.model_utils import DET_LAYERS, CAP_LAYERS, find_layer
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
from utils.tal.anchor_generator import make_anchors, dist2bbox

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class DDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class DualDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = max((ch[self.nl] // 4, self.reg_max * 4, 16)), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4 * self.reg_max, 1)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2])

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class DualDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4), nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2])
        #y = torch.cat((dbox2, cls2.sigmoid()), 1)
        #return y if self.export else (y, d2)
        #y1 = torch.cat((dbox, cls.sigmoid()), 1)
        #y2 = torch.cat((dbox2, cls2.sigmoid()), 1)
        #return [y1, y2] if self.export else [(y1, d1), (y2, d2)]
        #return [y1, y2] if self.export else [(y1, y2), (d1, d2)]

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class TripleDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 3  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = max((ch[self.nl] // 4, self.reg_max * 4, 16)), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        c6, c7 = max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), max((ch[self.nl * 2], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4 * self.reg_max, 1)) for x in ch[self.nl:self.nl*2])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])
        self.cv6 = nn.ModuleList(
            nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3), nn.Conv2d(c6, 4 * self.reg_max, 1)) for x in ch[self.nl*2:self.nl*3])
        self.cv7 = nn.ModuleList(
            nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)
        self.dfl3 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        d3 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
            d3.append(torch.cat((self.cv6[i](x[self.nl*2+i]), self.cv7[i](x[self.nl*2+i])), 1))
        if self.training:
            return [d1, d2, d3]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
        dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2, d3])

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class TripleDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 3  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        c6, c7 = make_divisible(max((ch[self.nl * 2] // 4, self.reg_max * 4, 16)), 4), \
                                max((ch[self.nl * 2], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3, g=4),
                          nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)) for x in ch[:self.nl])
        #self.cv3 = nn.ModuleList(
        #    nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1), DWConv(c3, c3, 3), Conv(c3, c3, 1), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3, g=4),
                          nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl:self.nl*2])
        #self.cv5 = nn.ModuleList(
        #    nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])
        self.cv5 = nn.ModuleList(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c5, 1), DWConv(c5, c5, 3), Conv(c5, c5, 1), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:self.nl*2])

        #self.cv6 = nn.ModuleList(
        #    nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3, g=4),
        #                  nn.Conv2d(c6, 4 * self.reg_max, 1, groups=4)) for x in ch[self.nl*2:self.nl*3])
        #self.cv7 = nn.ModuleList(
        #    nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nc, 1)) for x in ch[self.nl*2:self.nl*3])
        self.cv6 = deepcopy(self.cv4)
        self.cv7 = deepcopy(self.cv5)
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)
        self.dfl3 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        d3 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
            d3.append(torch.cat((self.cv6[i](x[self.nl*2+i].detach()), self.cv7[i](x[self.nl*2+i].detach())), 1))
        if self.training:
            return [d1, d2, d3]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box3, cls3 = torch.cat([di.view(shape[0], self.no, -1) for di in d3], 2).split((self.reg_max * 4, self.nc), 1)
        dbox3 = dist2bbox(self.dfl3(box3), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1), torch.cat((dbox3, cls3.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2, d3])
        #y = torch.cat((dbox3, cls3.sigmoid()), 1)
        #return y if self.export else (y, d3)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv6, m.cv7, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class Segment(Detect):
    # YOLO Segment head for segmentation models
    def __init__(self, nc=80, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class DSegment(DDetect):
    # YOLO Segment head for segmentation models
    def __init__(self, nc=80, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch[:-1], inplace)
        self.nl = len(ch)-1
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Conv(ch[-1], self.nm, 1)  # protos
        self.detect = DDetect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch[:-1])

    def forward(self, x):
        p = self.proto(x[-1])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x[:-1])
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class DualDSegment(DualDDetect):
    # YOLO Segment head for segmentation models
    def __init__(self, nc=80, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch[:-2], inplace)
        self.nl = (len(ch)-2) // 2
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Conv(ch[-2], self.nm, 1)  # protos
        self.proto2 = Conv(ch[-1], self.nm, 1)  # protos
        self.detect = DualDDetect.forward

        c6 = max(ch[0] // 4, self.nm)
        c7 = max(ch[self.nl] // 4, self.nm)
        self.cv6 = nn.ModuleList(nn.Sequential(Conv(x, c6, 3), Conv(c6, c6, 3), nn.Conv2d(c6, self.nm, 1)) for x in ch[:self.nl])
        self.cv7 = nn.ModuleList(nn.Sequential(Conv(x, c7, 3), Conv(c7, c7, 3), nn.Conv2d(c7, self.nm, 1)) for x in ch[self.nl:self.nl*2])

    def forward(self, x):
        p = [self.proto(x[-2]), self.proto2(x[-1])]
        bs = p[0].shape[0]

        mc = [torch.cat([self.cv6[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2),
              torch.cat([self.cv7[i](x[self.nl+i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)]  # mask coefficients
        d = self.detect(self, x[:-2])
        if self.training:
            return d, mc, p
        return (torch.cat([d[0][1], mc[1]], 1), (d[1][1], mc[1], p[1]))


class Panoptic(Detect):
    # YOLO Panoptic head for panoptic segmentation models
    def __init__(self, nc=80, sem_nc=93, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch, inplace)
        self.sem_nc = sem_nc
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.uconv = UConv(ch[0], ch[0]//4, self.sem_nc+self.nc)
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)


    def forward(self, x):
        p = self.proto(x[0])
        s = self.uconv(x[0])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p, s
        return (torch.cat([x, mc], 1), p, s) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p, s))


class DPanoptic(DDetect):
    # YOLO Panoptic head for panoptic segmentation models
    def __init__(self, nc=80, sem_nc=93, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch[:-2], inplace)
        self.sem_nc = sem_nc
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Conv(ch[-1], self.nm, 1)  # protos
        self.uconv = nn.Conv2d(ch[-2], self.sem_nc+self.nc, 1)
        self.detect = DDetect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch[:-2])


    def forward(self, x):
        p = self.proto(x[-1])
        s = self.uconv(x[-2])
        bs = p.shape[0]

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x[:-2])
        if self.training:
            return x, mc, p, s
        return (torch.cat([x, mc], 1), p, s) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p, s))


class DualDPanoptic(TripleDDetect):
    # YOLO Panoptic head for panoptic segmentation models
    def __init__(self, nc=80, sem_nc=93, nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, ch[:-4], inplace)
        self.nl = (len(ch)-4) // 3
        self.sem_nc = sem_nc
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Conv(ch[-2], self.nm, 1)  # protos
        self.proto2 = Conv(ch[-1], self.nm, 1)  # protos
        self.uconv = nn.Conv2d(ch[-4], self.sem_nc+self.nc, 1)
        self.uconv2 = nn.Conv2d(ch[-3], self.sem_nc+self.nc, 1)
        self.detect = TripleDDetect.forward

        c8 = max(ch[0] // 4, self.nm)
        c9 = max(ch[self.nl] // 4, self.nm)
        self.cv8 = nn.ModuleList(nn.Sequential(Conv(x, c8, 3), Conv(c8, c8, 3), nn.Conv2d(c8, self.nm, 1)) for x in ch[:self.nl])
        self.cv9 = nn.ModuleList(nn.Sequential(Conv(x, c9, 3), Conv(c9, c9, 3), nn.Conv2d(c9, self.nm, 1)) for x in ch[self.nl:self.nl*2])


    def forward(self, x):
        p = [self.proto(x[-2]), self.proto2(x[-1])]
        s = [self.uconv(x[-4]), self.uconv2(x[-3])]
        bs = p[0].shape[0]

        #xa, xb = x[3].chunk(2, 1)
        #x[3] = torch.cat((xa, xb.detach()), 1)
        #xa, xb = x[4].chunk(2, 1)
        #x[4] = torch.cat((xa, xb.detach()), 1)
        #xa, xb = x[5].chunk(2, 1)
        #x[5] = torch.cat((xa, xb.detach()), 1)
        #xa, xb = x[6].chunk(2, 1)
        #x[6] = torch.cat((xa, xb.detach()), 1)
        #xa, xb = x[7].chunk(2, 1)
        #x[7] = torch.cat((xa, xb.detach()), 1)
        #xa, xb = x[8].chunk(2, 1)
        #x[8] = torch.cat((xa, xb.detach()), 1)

        mc = [torch.cat([self.cv8[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2),
              torch.cat([self.cv9[i](x[self.nl+i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)]  # mask coefficients

        d = self.detect(self, x[:-4])
        if self.training:
            return d, mc, p, s
        return (torch.cat([d[0][-1], mc[-1]], 1), (d[1][-1], mc[-1], p[-1], s[-1]))
        #return (torch.cat([d[0][-2], mc[-1]], 1), (d[1][-1], mc[-1], p[-1], s[-1]))
        #return (torch.cat([d[0][-2], mc[-1]], 1), (d[1][-1], mc[-1], p[-1], s[-1]))


class Grit(nn.Module):
    def __init__(self, ch = (), inplace = True):
        super().__init__()

        self.nl = len(ch)-3
        self.reg_max = 16
        self.nc = 80
        self.no = self.nc + 4*self.reg_max
        self.k = 300

        # default config settings
        self.config = {
            'use_gri_feat': True,
            'use_reg_feat': True,
            'grid_feat_dim': 512,
            'beam_size': 5,
            'beam_len': 20,
            'dropout': 0.2,
            'attn_dropout': 0.2,

            # coco vocab
            'vocab_size': 10202,
            'pad_idx': 0,  # [PAD]
            'bos_idx': 2,  # [CLS]
            'eos_idx': 3,  # [SEP]

            # bert vocab
            #'vocab_size': 30522,
            #'pad_idx': 0,  # [PAD]
            #'bos_idx': 101,  # [CLS]
            #'eos_idx': 102,  # [SEP]

            'max_len': 128,
            'd_model': 256,
            'n_heads': 8,
            'd_ff': 512,

            'grid_net': {
                'n_memories': 1,
                'n_layers': 1,
            },

            'cap_generator': {
                'decoder_name': 'parallel', # sequential concat
                'n_layers': 1,
            },
        }

        self.grit_transformer = grit_transformer(config = self.config)

        c3 = max(ch[0], 128)  # channels
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), Conv(c3, c3, 1)) for x in ch[:self.nl])

        c4 = max(ch[-3], 128)  # channels
        self.cv4 = nn.Sequential(Conv(ch[-3], c4, 1), Conv(c4, c4, 1))

        c5 = c3
        self.cv5 = nn.Sequential(Conv(ch[-2], c3, 1), Conv(c3, c3, 1))

        self.src_mask = None
        self.seq = None

        self.use_beam_search = False
        self.beam_size = 5
        self.out_size = 1
        self.return_probs = False

    def forward(self, x):

        det = x[-1][0][-1] if self.training else x[-1][1][0] #[-1]
        sem = x[-1][-1][-1] if self.training else x[-1][1][-1]
        xsem = x[-2]
        xs = x[-3]
        x = x[:-3]
        shape = x[0].shape

        for i in range(self.nl):
            x[i] = self.cv3[i](x[i])
        xs = self.cv4(xs)
        xsem = self.cv5(xsem)

        # Caption
        gri_feat = xs.float()
        gri_mask = [F.interpolate(self.src_mask[None].float(), size = f.shape[-2:]).to(torch.bool)[0] for l, f in enumerate(gri_feat)]  # masks [[B, Hi, Wi]]

        #reg_feat = None
        #for df in x:
        #    df = rearrange(df, 'b c h w -> b (h w) c')
        #    reg_feat = df.float() if (reg_feat is None) else torch.cat((reg_feat, df.float()), dim = 1)
        # select top k
        _, cls = torch.cat([xi.detach().view(shape[0], self.no, -1) for xi in det], 2).split((self.reg_max * 4, self.nc), 1)
        _, ki = torch.topk(torch.max(cls, dim = 1)[0], k = self.k)  # shape = (b, k)
        reg_feat = torch.cat([rearrange(xi, 'b c h w -> b (h w) c').float() for xi in x[: self.nl]], dim = 1)#shape = (b, P3 + P4 + P5, c)
        shape = reg_feat.shape  # shape = (b, P3 + P4 + P5, c)
        reg_feat = torch.gather(reg_feat, dim = 1, index = ki.unsqueeze(2).expand(shape[0], self.k, shape[-1]))  # shape = (b, k, c)


        sem = rearrange(sem.detach().float(), 'b c h w -> (h w) c b')
        _, idx = sem.max(dim = 0)
        idx = idx.permute(1, 0)

        xsem = rearrange(xsem.float(), 'b c h w -> b c (h w)')
        #sem_feat = xsem.index_select(1, idx).permute(0, 2, 1)  # b, cls, c
        #print(sem_feat.shape)
        sem_feat = torch.cat([xsem[i].index_select(1, idx[i]).unsqueeze(0) for i in range(xsem.shape[0])]).permute(0, 2, 1)  # b, cls, c
        #print(sem_feat.shape)
        reg_feat = torch.cat([reg_feat, sem_feat], dim = 1)

        images = {
            'gri_feat': rearrange(gri_feat, 'b c h w -> b (h w) c'),
            'gri_mask': repeat(gri_mask[-1], 'b h w -> b 1 1 (h w)'),
            'reg_feat': reg_feat,
            'reg_mask': reg_feat.data.new_full((reg_feat.shape[0], 1, 1, reg_feat.shape[1]), 0).bool(),
        }

        self.grit_transformer = self.grit_transformer.to(gri_feat.device)
        return self.grit_transformer(
            images,
            seq = self.seq,
            use_beam_search = self.use_beam_search,
            max_len = self.config['beam_len'],
            eos_idx = self.config['eos_idx'],
            beam_size = self.beam_size,
            out_size = self.out_size,
            return_probs = self.return_probs,
        )

    def set_params(self, src_mask, tgt, tgt_mask, use_beam_search = False, beam_size = 5, out_size = 1, return_probs = False):
        self.src_mask = src_mask
        # self.seq = NestedTensor(tgt, tgt_mask)
        self.seq = tgt
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size
        self.out_size = out_size
        self.return_probs = return_probs


class DualGrit(nn.Module):
    def __init__(self, ch = (), inplace = True):
        super().__init__()

        self.nl = (len(ch)-2) // 2

        # default config settings
        self.config = {
            'use_gri_feat': True,
            'use_reg_feat': True,
            'grid_feat_dim': 512,
            'beam_size': 5,
            'beam_len': 20,
            'dropout': 0.2,
            'attn_dropout': 0.2,

            # coco vocab
            'vocab_size': 10202,
            'pad_idx': 0,  # [PAD]
            'bos_idx': 2,  # [CLS]
            'eos_idx': 3,  # [SEP]

            # bert vocab
            #'vocab_size': 30522,
            #'pad_idx': 0,  # [PAD]
            #'bos_idx': 101,  # [CLS]
            #'eos_idx': 102,  # [SEP]

            'max_len': 128,
            'd_model': 256,
            'n_heads': 8,
            'd_ff': 512,

            'grid_net': {
                'n_memories': 1,
                'n_layers': 2,
            },

            'cap_generator': {
                'decoder_name': 'parallel', # sequential concat
                'n_layers': 2,
            },
        }

        self.grit_transformer = grit_transformer(config = self.config)
        self.grit_transformer2 = grit_transformer(config = self.config)

        c3 = max(ch[self.nl], 128)  # channels
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3)) for x in ch[:self.nl])

        c4 = max(ch[-1], 128)  # channels
        self.cv4 = nn.Sequential(Conv(ch[-2], c4, 3), Conv(c4, c4, 3))

        c5 = max(ch[self.nl], 128)  # channels
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3)) for x in ch[self.nl:2*self.nl])

        c6 = max(ch[-1], 128)  # channels
        self.cv6 = nn.Sequential(Conv(ch[-1], c6, 3), Conv(c6, c6, 3))

        self.src_mask = None
        self.seq = None

        self.use_beam_search = False
        self.beam_size = 5
        self.out_size = 1
        self.return_probs = False

    def forward(self, x):

        #xs = x[-1].detach().clone()
        #x = [xi.detach().clone() for xi in x[:-1]]
        xs2 = x[-1]
        x2 = x[self.nl:2*self.nl]
        xs = x[-2]
        x = x[:self.nl]

        for i in range(self.nl):
            x[i] = self.cv3[i](x[i])
            x2[i] = self.cv5[i](x2[i])
        xs = self.cv4(xs)
        xs2 = self.cv6(xs2)

        # Caption
        gri_feat = xs.float()
        gri_mask = [F.interpolate(self.src_mask[None].float(), size = f.shape[-2:]).to(torch.bool)[0] for l, f in enumerate(gri_feat)]  # masks [[B, Hi, Wi]]
        gri_feat2 = xs2.float()
        gri_mask2 = [F.interpolate(self.src_mask[None].float(), size = f.shape[-2:]).to(torch.bool)[0] for l, f in enumerate(gri_feat2)]  # masks [[B, Hi, Wi]]

        reg_feat = None
        for df in x:
            df = rearrange(df, 'b c h w -> b (h w) c')
            reg_feat = df.float() if (reg_feat is None) else torch.cat((reg_feat, df.float()), dim = 1)
        reg_feat2 = None
        for df in x2:
            df = rearrange(df, 'b c h w -> b (h w) c')
            reg_feat2 = df.float() if (reg_feat2 is None) else torch.cat((reg_feat2, df.float()), dim = 1)

        images = {
            'gri_feat': rearrange(gri_feat, 'b c h w -> b (h w) c'),
            'gri_mask': repeat(gri_mask[-1], 'b h w -> b 1 1 (h w)'),
            'reg_feat': reg_feat,
            'reg_mask': reg_feat.data.new_full((reg_feat.shape[0], 1, 1, reg_feat.shape[1]), 0).bool(),
        }

        images2 = {
            'gri_feat': rearrange(gri_feat2, 'b c h w -> b (h w) c'),
            'gri_mask': repeat(gri_mask2[-1], 'b h w -> b 1 1 (h w)'),
            'reg_feat': reg_feat2,
            'reg_mask': reg_feat2.data.new_full((reg_feat2.shape[0], 1, 1, reg_feat2.shape[1]), 0).bool(),
        }

        self.grit_transformer = self.grit_transformer.to(gri_feat.device)
        self.grit_transformer2 = self.grit_transformer2.to(gri_feat2.device)
        if self.training:
            return [self.grit_transformer(
                images,
                seq = self.seq,
                use_beam_search = self.use_beam_search,
                max_len = self.config['beam_len'],
                eos_idx = self.config['eos_idx'],
                beam_size = self.beam_size,
                out_size = self.out_size,
                return_probs = self.return_probs,
            ),
            self.grit_transformer2(
                images2,
                seq = self.seq,
                use_beam_search = self.use_beam_search,
                max_len = self.config['beam_len'],
                eos_idx = self.config['eos_idx'],
                beam_size = self.beam_size,
                out_size = self.out_size,
                return_probs = self.return_probs,
            )]
        else:
            return self.grit_transformer(
                images,
                seq = self.seq,
                use_beam_search = self.use_beam_search,
                max_len = self.config['beam_len'],
                eos_idx = self.config['eos_idx'],
                beam_size = self.beam_size,
                out_size = self.out_size,
                return_probs = self.return_probs,
            )

    def set_params(self, src_mask, tgt, tgt_mask, use_beam_search = False, beam_size = 5, out_size = 1, return_probs = False):
        self.src_mask = src_mask
        # self.seq = NestedTensor(tgt, tgt_mask)
        self.seq = tgt
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size
        self.out_size = out_size
        self.return_probs = return_probs


class OutputLayer(nn.Module):
    def forward(self, x):
        output_type = ['detect', 'captions']
        output = {}
        for ot, out in zip(output_type, x):
            output[ot] = out

        return output


class BaseModel(nn.Module):
    # YOLO base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (RepConvN)) and hasattr(m, 'fuse_convs'):
                m.fuse_convs()
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        det_layer = find_layer(self.model, DET_LAYERS)
        m = self.model[det_layer]  # Detect()
        if isinstance(m, (
            Detect, DualDetect, TripleDetect, DDetect, DualDDetect, TripleDDetect,
            Segment, DSegment, DualDSegment,
            Panoptic, DPanoptic, DualDPanoptic,
            Grit, DualGrit
        )):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
            # m.grid = list(map(fn, m.grid))
        return self


class DetectionModel(BaseModel):
    # YOLO detection model
    def __init__(self, cfg='yolo.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        self.init_phase = True

        # Build strides, anchors
        self.det_layer = find_layer(self.model, DET_LAYERS)
        self.cap_layer = find_layer(self.model, CAP_LAYERS)

        m = self.model[self.det_layer]  # Detect()
        if isinstance(m, (
            Detect, DDetect, Segment, DSegment, Panoptic, DPanoptic,
            DualDetect, TripleDetect, DualDDetect, TripleDDetect, DualDSegment, DualDPanoptic,
            Grit, DualGrit
        )):
            s = 256  # 2x min stride
            m.inplace = self.inplace

            if isinstance(m, (Detect, DDetect, Segment, DSegment, Panoptic, DPanoptic)):
                forward = lambda x: self.forward(x)['detect'][0] if isinstance(m, (Segment, DSegment, Panoptic, DPanoptic)) else self.forward(x)['detect']
            elif isinstance(m, (DualDetect, TripleDetect, DualDDetect, TripleDDetect, DualDSegment, DualDPanoptic)):
                forward = lambda x: self.forward(x)['detect'][0][0] if isinstance(m, (DualDSegment, DualDPanoptic)) else self.forward(x)['detect'][0]
            elif isinstance(m, (Grit, DualGrit)):
                forward = lambda x: self.forward(x)['captions']
            else:
                forward = lambda x: self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # check_anchor_order(m)
            # m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if self.init_phase and isinstance(self.model[self.cap_layer], (Grit, DualGrit)):
            # set params
            _, src_mask = create_src_mask(x)
            # TODO: get length from hyp
            cap, cap_mask = create_caption_and_mask(128)
            self.model[self.cap_layer].set_params(src_mask, cap, cap_mask)
            self.init_phase = False

        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLO augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y


Model = DetectionModel  # retain YOLO 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLO segmentation model
    def __init__(self, cfg='yolo-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLO classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLO classification model from a YOLO detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLO classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        RepConvN.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv, AConv, ConvTranspose,
            Bottleneck, SPP, SPPF, DWConv, BottleneckCSP, nn.ConvTranspose2d, DWConvTranspose2d, SPPCSPC, ADown, ASCDown,
            RepNCSPELAN4, RepNCSPELAN4B, SPPELAN, PSA}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, SPPCSPC}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        # TODO: channel, gw, gd
        elif m in {
            Detect, DualDetect, TripleDetect, DDetect, DualDDetect, TripleDDetect,
            Segment, DSegment, DualDSegment,
            Panoptic, DPanoptic, DualDPanoptic,
            Grit, DualGrit
        }:
            args.append([ch[x] for x in f])
            # if isinstance(args[1], int):  # number of anchors
            #     args[1] = [list(range(args[1] * 2))] * len(f)
            if m in {Segment, DSegment, DualDSegment, Panoptic, DPanoptic, DualDPanoptic}:
                args[2] = make_divisible(args[2] * gw, 8)
        elif m in {OutputLayer}:
            pass
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolo.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)
    model.eval()

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
