import argparse
import itertools
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import caption.val as validate
from models.common import DetectMultiBackend
from models.yolo import Grit, SegmentationModel
from utils.caption.caption_utils import bert_tokenizer
from utils.general import LOGGER, TQDM_BAR_FORMAT, init_seeds, colorstr, increment_path, intersect_dicts, check_yaml, check_img_size, check_dataset
from utils.model_utils import find_layer, CAP_LAYERS, OUTPUT_LAYERS
from utils.caption.dataloaders import create_dataloader
from utils.torch_utils import select_device, torch_distributed_zero_first, de_parallel

from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from pycocoevalcap.cider.cider import Cider

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None  # check_git_info()

def co_optimizer(model, caption_idx):
    lr, beta, decay = 0.000005, (0.9, 0.99), 0.01  # 5e-6, (0.9, 0.99), 0.01
    g = [], [], []
    cap_g = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for k, v in model.named_modules():
        # Grit Layer
        if f'model.{caption_idx}.' not in k:
            continue
        if isinstance(v, (Grit)):
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                cap_g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                cap_g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                cap_g[0].append(v.weight)
            continue

        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                g[1].append(v.im.implicit)
            else:
                for iv in v.im:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                g[1].append(v.ia.implicit)
            else:
                for iv in v.ia:
                    g[1].append(iv.implicit)

    optimizer = torch.optim.AdamW(g[2], lr = lr, betas = beta, eps = 1e-4, weight_decay = 0.0)

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    optimizer.add_param_group({'params': cap_g[0], 'weight_decay': decay})
    optimizer.add_param_group({'params': cap_g[1], 'weight_decay': 0.0})
    optimizer.add_param_group({'params': cap_g[2], 'weight_decay': 0.0})
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer


def gather_result(value):
    if isinstance(value, torch.Tensor):
        torch.distributed.all_reduce(value, async_op = False)  # compute the sum
        value.mul_(1.0 / torch.distributed.get_world_size())  # compute the avg
    return value

def cider_optimization(model, opt, hyp, device, train_loader, val_loader, model_key, caption_idx):
    # CIDEr Optimization
    optimizer = co_optimizer(model, caption_idx)

    save_dir = Path(opt.save_dir)
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents = True, exist_ok = True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    if ('caption_tokenizer' in hyp) and ('custom' == hyp['caption_tokenizer']):
        tokenizer = bert_tokenizer(
            model = hyp['caption_tokenizer'],
            vocab = hyp['caption_vocab_path'],
            do_lower = True,
        )
    else:
        tokenizer = bert_tokenizer(do_lower = True)

    for epoch in range(opt.epochs):
        running_reward = .0
        running_reward_baseline = .0
        running_loss = .0
        beam_size = hyp['caption_beam_size']
        model.train()

        if -1 != RANK:
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(enumerate(train_loader), total = len(train_loader), bar_format = TQDM_BAR_FORMAT)
        pbar_desc = ('Epoch', 'GPU_mem', 'loss', 'reward', 'reward_baseline')

        # CIDEr
        cider_scorer = Cider()
        best_cider = 0

        LOGGER.info(('\n' + '%20s' * len(pbar_desc)) % pbar_desc)
        for i, (_, ori_imgs, _, _, _, _, _, src_masks, _, _, img_ids) in pbar:  # batch ---------------------------------------------------
            optimizer.zero_grad()
            model_key[caption_idx].set_params(src_masks.to(device), None, None, use_beam_search = True, beam_size = beam_size, out_size = beam_size, return_probs = False)

            ori_imgs = ori_imgs.to(device, non_blocking = True).float() / 255
            outs, log_probs = model(ori_imgs)['captions']

            # Rewards
            caps_gen = tokenizer.get_batch_decoded_captions(outs.view(-1, hyp['caption_beam_len']), skip_special_tokens = True)
            all_caps = train_loader.dataset.get_all_caps(img_ids)
            caps_gt = list(itertools.chain(*([all_caps[img_id]] * beam_size for img_id in img_ids)))

            tk_caps_gen = {}
            tk_caps_gt = {}
            for i, (gen, gt) in enumerate(zip(caps_gen, caps_gt)):
                tk_caps_gen[i] = [gen]
                tk_caps_gt[i] = gt

            batch_size = outs.shape[0]
            reward = cider_scorer.compute_score(tk_caps_gt, tk_caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(batch_size, beam_size)

            reward_baseline = torch.mean(reward, -1, keepdim = True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()

            if -1 != RANK:
                torch.distributed.barrier()

            optimizer.step()

            loss = gather_result(loss) if (-1 != RANK) else loss
            running_loss += loss.item()

            reward = gather_result(reward.mean()) if (-1 != RANK) else reward.mean()
            running_reward += reward.item()

            reward_baseline = gather_result(reward_baseline.mean()) if (-1 != RANK) else reward_baseline.mean()
            running_reward_baseline += reward_baseline.item()

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%20s' * 2 + '%20.4g' * (len(pbar_desc) - 2)) %
                                 (f'{epoch}/{opt.epochs - 1}', mem, running_loss, running_reward, running_reward_baseline))


        if RANK in {-1, 0}:
            results, maps, _ = validate.run(
                data_dict,
                batch_size = batch_size // WORLD_SIZE * 2,
                imgsz = imgsz,
                half = False, # amp,
                model = model,
                single_cls = False,
                dataloader = val_loader,
                save_dir = save_dir,
                plots = False,
                mask_downsample_ratio = opt.mask_ratio,
                overlap = False,
            )
            cider = np.array(results)[10]

            ckpt = {
                'epoch': epoch,
                'best_cider': best_cider,
                'model': deepcopy(de_parallel(model)), #.half(),
                'optimizer': optimizer.state_dict(),
                'opt': vars(opt),
                'date': datetime.now().isoformat(),
            }

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_cider < cider:
                torch.save(ckpt, best)
            if (0 < opt.save_period) and (0 == epoch % opt.save_period):
                torch.save(ckpt, w / f'epoch{epoch}.pt')
            del ckpt

    return model


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type = str, default = ROOT / 'yolo-pan.pt', help = 'initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type = str, default = ROOT / 'data/coco128-seg.yaml', help = 'dataset.yaml path')
    parser.add_argument('--hyp', type = str, default = ROOT / 'data/hyps/hyp.scratch-low.yaml', help = 'hyperparameters path')
    parser.add_argument('--epochs', type = int, default = 100, help = 'total training epochs')
    parser.add_argument('--batch-size', type = int, default = 16, help = 'total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type = int, default = 640, help = 'train, val image size (pixels)')
    parser.add_argument('--resume', nargs = '?', const = True, default = False, help = 'resume most recent training')
    parser.add_argument('--nosave', action = 'store_true', help = 'only save final checkpoint')
    parser.add_argument('--noval', action = 'store_true', help = 'only validate final epoch')
    parser.add_argument('--cache', type = str, nargs = '?', const = 'ram', help = 'image --cache ram/disk')
    parser.add_argument('--device', default = '', help = 'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type = str, choices = ['SGD', 'Adam', 'AdamW', 'LION'], default = 'AdamW', help = 'optimizer')
    parser.add_argument('--workers', type = int, default = 8, help = 'max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default = ROOT / 'runs/train-co', help = 'save to project/name')
    parser.add_argument('--name', default = 'exp', help = 'save to project/name')
    parser.add_argument('--exist-ok', action = 'store_true', help = 'existing project/name ok, do not increment')
    parser.add_argument('--freeze', nargs = '+', type = int, default = [0], help = 'Freeze layers: backbone = 10, first3 = 0 1 2')
    parser.add_argument('--save-period', type = int, default = -1, help = 'Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type = int, default = 0, help = 'Global training seed')
    parser.add_argument('--local_rank', type = int, default = -1, help = 'Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to saving memory')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.hyp = check_yaml(opt.hyp)  # check YAML
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok = opt.exist_ok))
    cuda = torch.cuda.is_available() and ('cpu' != opt.device)

    init_seeds(opt.seed + 1 + RANK, deterministic = True)

    device = select_device(opt.device, batch_size = opt.batch_size)

    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLO Multi-GPU DDP training'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend = "nccl" if dist.is_nccl_available() else "gloo", timeout = timedelta(seconds = 86400))

    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']

    ckpt = torch.load(opt.weights, map_location = 'cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    if ckpt.get('ema'):
        ckpt['model'] = ckpt['ema']
    model = SegmentationModel(opt.cfg or ckpt['model'].yaml, ch = 3, nc = int(data_dict['nc'])).to(device)
    exclude = ['anchor']  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict = False)  # load
    LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {opt.weights}')  # report

    stride = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, s = stride)  # check image size

    model.hyp = hyp

    model_key = de_parallel(model)
    model_key = model_key.model if not isinstance(model_key.model, SegmentationModel) else model_key.model.model
    caption_idx = find_layer(model_key, CAP_LAYERS)
    output_idx = find_layer(model_key, OUTPUT_LAYERS)

    freeze = [f'model.{x}.' for x in (caption_idx, output_idx)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = False  # freeze all layers
        if any(x in k for x in freeze):
            LOGGER.info(f'unfreezing {k}')
            v.requires_grad = True

    def replace_bn(module, name):
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if type(target_attr) == nn.BatchNorm2d:
                fbn = FrozenBatchNorm2d(target_attr.num_features, target_attr.eps)
                fbn.running_mean.data = target_attr.running_mean.data
                fbn.running_var.data = target_attr.running_var.data
                fbn.weight.data = target_attr.weight.data
                fbn.bias.data = target_attr.bias.data
                setattr(module, attr_str, fbn)

        # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
        for name, immediate_child_module in module.named_children():
            replace_bn(immediate_child_module, name)

    replace_bn(model, 'model')

    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        opt.batch_size // WORLD_SIZE,
        stride,
        single_cls = False,
        hyp = hyp,
        rect = False,
        rank = LOCAL_RANK,
        workers = opt.workers,
        prefix = colorstr('train: '),
        mask_downsample_ratio = opt.mask_ratio,
        overlap_mask = False,
        is_train = True,
        pin_memory = cuda,
    )

    if True:#RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            opt.batch_size // WORLD_SIZE * 2,
            stride,
            single_cls = False,
            hyp = hyp,
            cache = None,
            rect = False,
            rank = -1,
            workers = opt.workers * 2,
            pad = 0.5,
            mask_downsample_ratio = opt.mask_ratio,
            overlap_mask = False,
            prefix = colorstr('val: '),
            is_train = False,
            pin_memory = cuda,
        )[0]

    _ = cider_optimization(model, opt, hyp, device, train_loader, val_loader, model_key, caption_idx)
