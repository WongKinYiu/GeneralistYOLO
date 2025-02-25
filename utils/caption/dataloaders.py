import os
import random

import pickle
from pathlib import Path
from PIL import Image, ImageOps

from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool

import cv2
import json
import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader, distributed, BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

from ..augmentations import augment_hsv
from ..dataloaders import InfiniteDataLoader, LoadImagesAndLabels, seed_worker, get_hash, img2label_paths, verify_image_label, HELP_URL, TQDM_BAR_FORMAT, LOCAL_RANK, IMG_FORMATS
from ..general import NUM_THREADS, LOGGER, xyn2xy, xywhn2xyxy, xyxy2xywhn
from ..torch_utils import torch_distributed_zero_first
from ..coco_utils import annToMask, getCocoIds
from ..caption.caption_utils import bert_tokenizer, create_src_mask
from .augmentations import mixup, random_perspective, copy_paste, letterbox

RANK = int(os.getenv('RANK', -1))


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls = False,
    hyp = None,
    augment = False,
    cache = False,
    pad = 0.0,
    rect = False,
    rank = -1,
    workers = 8,
    image_weights = False,
    close_mosaic = False,
    quad = False,
    prefix = '',
    shuffle = False,
    mask_downsample_ratio = 1,
    overlap_mask = False,
    is_train = False,
    pin_memory = False,
):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadCaptions(
            path,
            imgsz,
            batch_size,
            augment = augment,  # augmentation
            hyp = hyp,  # hyperparameters
            rect = rect,  # rectangular batches
            cache_images = cache,
            single_cls = single_cls,
            stride = int(stride),
            pad = pad,
            image_weights = image_weights,
            prefix = prefix,
            downsample_ratio = mask_downsample_ratio,
            overlap = overlap_mask,
            is_train = is_train,
        )


    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    # loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    loader = DataLoader if image_weights or close_mosaic else InfiniteDataLoader
    # generator = torch.Generator()
    # generator.manual_seed(6148914691236517205 + RANK)
    # return loader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=shuffle and sampler is None,
    #     num_workers=nw,
    #     sampler=sampler,
    #     pin_memory=True,
    #     collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if quad else LoadImagesAndLabelsAndMasks.collate_fn,
    #     worker_init_fn=seed_worker,
    #     generator=generator,
    # ), dataset

    if is_train:
        if -1 != rank:
            sampler = distributed.DistributedSampler(dataset, shuffle = shuffle)
            dataloader = loader(
                dataset,
                batch_size = batch_size,
                shuffle = shuffle and sampler is None,
                num_workers = nw,
                sampler = sampler,
                pin_memory = pin_memory,
                collate_fn = LoadCaptions.collate_fn4 if quad else LoadCaptions.collate_fn,
                worker_init_fn = seed_worker,
                # generator = generator,
            )
        else:
            batch_sampler = BatchSampler(
                RandomSampler(dataset), batch_size, drop_last = False
            )
            dataloader = loader(
                dataset,
                # batch_size = batch_size,
                # shuffle = shuffle and sampler is None,
                num_workers = nw,
                batch_sampler = batch_sampler,
                pin_memory = pin_memory,
                collate_fn = LoadCaptions.collate_fn4 if quad else LoadCaptions.collate_fn,
                worker_init_fn = seed_worker,
                # generator = generator,
            )

    else:
        sampler = SequentialSampler(dataset)
        dataloader = loader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle and sampler is None,
            num_workers = nw,
            sampler = sampler,
            pin_memory = pin_memory,
            collate_fn = LoadCaptions.collate_fn4 if quad else LoadCaptions.collate_fn,
            worker_init_fn = seed_worker,
            # generator = generator,
        )

    return dataloader, dataset

def img2stuff_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}stuff{os.sep}'  # /images/, /segmentations/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # for training/testing

    def __init__(
        self,
        path,
        img_size = 640,
        batch_size = 16,
        augment = False,
        hyp = None,
        rect = False,
        image_weights = False,
        cache_images = False,
        single_cls = False,
        stride = 32,
        pad = 0,
        min_items = 0,
        prefix = "",
        overwrite_cache = False,
        downsample_ratio = 1,
        overlap = False,
    ):
        super().__init__(
            path,
            img_size,
            batch_size,
            augment,
            hyp,
            rect,
            image_weights,
            cache_images,
            single_cls,
            stride,
            pad,
            min_items,
            prefix,
            overwrite_cache = overwrite_cache,
        )
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap

        # semantic segmentation
        self.coco_ids = getCocoIds()

        # Check cache
        self.seg_files = img2stuff_paths(self.im_files)  # labels
        p = Path(path)
        cache_path = (p.with_suffix('') if p.is_file() else Path(self.seg_files[0]).parent)
        cache_path = Path(str(cache_path) + '_stuff').with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle = True).item(), True  # load dict
            #assert cache['version'] == self.cache_version  # matches current version
            #assert cache['hash'] == get_hash(self.seg_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_seg_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc = (prefix + d), total = n, initial = n, bar_format = TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert (0 < nf) or (not augment), f'{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        seg_labels, _, self.semantic_masks = zip(*cache.values())
        nl = len(np.concatenate(seg_labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}'

        # Update labels
        self.seg_cls = []
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, semantic_masks) in enumerate(zip(seg_labels, self.semantic_masks)):
            self.seg_cls.append((label[:, 0].astype(int)).tolist())
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                if semantic_masks:
                    self.semantic_masks[i] = semantic_masks[j]
            if single_cls:  # single-class training, merge all classes into 0
                if semantic_masks:
                    self.semantic_masks[i][:, 0] = 0

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        masks = []
        if mosaic:
            # Load mosaic
            img, labels, segments, seg_cls, semantic_masks = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img, labels, segments, seg_cls, semantic_masks = mixup(img, labels, segments, seg_cls, semantic_masks,
                                                                       *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            segments = self.segments[index].copy()
            if len(segments):
                for i_s in range(len(segments)):
                    segments[i_s] = xyn2xy(
                        segments[i_s],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw=pad[0],
                        padh=pad[1],
                    )

            seg_cls = self.seg_cls[index].copy()
            semantic_masks = self.semantic_masks[index].copy()
            #semantic_masks = [xyn2xy(x, ratio[0] * w, ratio[1] * h, padw = pad[0], padh = pad[1]) for x in semantic_masks]
            if len(semantic_masks):
                for ss in range(len(semantic_masks)):
                    semantic_masks[ss] = xyn2xy(
                        semantic_masks[ss],
                        ratio[0] * w,
                        ratio[1] * h,
                        padw = pad[0],
                        padh = pad[1],
                    )

            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels, segments, semantic_masks = random_perspective(
                                                           img,
                                                           labels,
                                                           segments=segments,
                                                           semantic_masks = semantic_masks,
                                                           degrees=hyp["degrees"],
                                                           translate=hyp["translate"],
                                                           scale=hyp["scale"],
                                                           shear=hyp["shear"],
                                                           perspective=hyp["perspective"])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            if self.overlap:
                masks, sorted_idx = polygons2masks_overlap(img.shape[:2],
                                                           segments,
                                                           downsample_ratio=self.downsample_ratio)
                masks = masks[None]  # (640, 640) -> (1, 640, 640)
                labels = labels[sorted_idx]
            else:
                masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)

        masks = (torch.from_numpy(masks) if len(masks) else torch.zeros(1 if self.overlap else nl, img.shape[0] //
                                                                        self.downsample_ratio, img.shape[1] //
                                                                        self.downsample_ratio))
        semantic_masks = polygons2masks(img.shape[:2], semantic_masks, color = 1, downsample_ratio=self.downsample_ratio)
        #semantic_masks = polygons2masks(img.shape[:2], semantic_masks, color = 1, downsample_ratio=1)
        semantic_masks = torch.from_numpy(semantic_masks)
        # TODO: albumentations support
        if self.augment:
            # Albumentations
            # there are some augmentation that won't change boxes and masks,
            # so just be it for now.
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations
            ns = len(semantic_masks)

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
                    masks = torch.flip(masks, dims=[1])
                if ns:
                    semantic_masks = torch.flip(semantic_masks, dims = [1])

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
                    masks = torch.flip(masks, dims=[2])
                if ns:
                    semantic_masks = torch.flip(semantic_masks, dims = [2])

            # Cutouts  # labels = cutout(img, labels, p=0.5)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Combine semantic masks
        semantic_seg_masks = torch.zeros((len(self.coco_ids), img.shape[0] // self.downsample_ratio,
                                          img.shape[1] // self.downsample_ratio), dtype = torch.uint8)
        #semantic_seg_masks = torch.zeros((len(self.coco_ids), img.shape[0], img.shape[1]), dtype = torch.uint8)
        for cls_id, semantic_mask in zip(seg_cls, semantic_masks):
            semantic_seg_masks[cls_id] = (semantic_seg_masks[cls_id].logical_or(semantic_mask)).int()


        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return (torch.from_numpy(img), labels_out, self.im_files[index], shapes, masks, semantic_seg_masks)

    def load_mosaic(self, index):
        # YOLO 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4, seg_cls, semantic_masks4 = [], [], [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        # 3 additional image indices
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels, segments, semantic_masks = self.labels[index].copy(), self.segments[index].copy(), self.semantic_masks[index].copy()

            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            semantic_masks = [xyn2xy(x, w, h, padw, padh) for x in semantic_masks]
            labels4.append(labels)
            segments4.extend(segments)
            seg_cls.extend(self.seg_cls[index].copy())
            semantic_masks4.extend(semantic_masks)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for i in range(len(semantic_masks4)):
            if i < len(segments4):
                np.clip(labels4[:, 1:][i], 0, 2 * s, out = labels4[:, 1:][i])
                np.clip(segments4[i], 0, 2 * s, out = segments4[i])
            np.clip(semantic_masks4[i], 0, 2 * s, out = semantic_masks4[i])
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # 3 additional image indices
        # Augment
        img4, labels4, segments4, seg_cls, semantic_masks4 = copy_paste(img4, labels4, segments4, seg_cls, semantic_masks4, p=self.hyp["copy_paste"])
        img4, labels4, segments4, semantic_masks4 = random_perspective(img4,
                                                      labels4,
                                                      segments4,
                                                      semantic_masks4,
                                                      degrees=self.hyp["degrees"],
                                                      translate=self.hyp["translate"],
                                                      scale=self.hyp["scale"],
                                                      shear=self.hyp["shear"],
                                                      perspective=self.hyp["perspective"],
                                                      border=self.mosaic_border)  # border to remove

        return img4, labels4, segments4, seg_cls, semantic_masks4

    def cache_seg_labels(self, path = Path('./labels_stuff.cache'), prefix = ''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.seg_files, repeat(prefix))),
                        desc = desc,
                        total = len(self.im_files),
                        bar_format = TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.seg_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, masks, semantic_masks = zip(*batch)  # transposed
        batched_masks = torch.cat(masks, 0)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, batched_masks, torch.stack(semantic_masks, 0)


class LoadCaptions(LoadImagesAndLabelsAndMasks):
    def __init__(
        self,
        path,
        img_size = 640,
        batch_size = 16,
        augment = False,
        hyp = None,
        rect = False,
        image_weights = False,
        cache_images = False,
        single_cls = False,
        stride = 32,
        pad = 0,
        prefix = '',
        downsample_ratio = 1,
        overwrite_cache = True,
        overlap = False,
        is_train = False,
    ):
        super().__init__(
            path,
            img_size = img_size,
            batch_size = batch_size,
            augment = augment,
            hyp = hyp,
            rect = rect,
            image_weights = image_weights,
            cache_images = cache_images,
            single_cls = single_cls,
            stride = stride,
            pad = pad,
            prefix = prefix,
            overwrite_cache = overwrite_cache,
            downsample_ratio = downsample_ratio,
            overlap = overlap,
        )

        current_path = os.path.dirname(os.path.abspath(__file__))
        dir_name = (path.split('/')[-1 :])[0].split('.')[0]
        ann_path = Path(str(Path(self.im_files[0]).parent.parent.parent) + f'/annotations/captions_{dir_name}.json')
        with open(ann_path, 'r') as j:
            json_data = json.load(j)

        ann_file_name = '/train_cap.cache' if is_train else '/val_cap.cache'
        ann_cache_path = Path(str(Path(self.im_files[0]).parent.parent.parent) + ann_file_name)
        ann_cache = {}

        self.anns = []
        self.all_caps = {}

        if Path(ann_cache_path).is_file():
            print(f'Load ann cache {ann_cache_path}')
            ann_cache = np.load(ann_cache_path, allow_pickle = True).item()  # load dict

        ann_path = str(ann_path)
        if (0 == len(ann_cache)) or (ann_cache['hash'] != get_hash(ann_path)):
            annotations = json_data['annotations']

            if ('caption_tokenizer' in hyp) and ('custom' == hyp['caption_tokenizer']):
                tokenizer = bert_tokenizer(
                    model = hyp['caption_tokenizer'],
                    vocab = hyp['caption_vocab_path'],
                    do_lower = True,
                )
            else:
                tokenizer = bert_tokenizer(do_lower = True)

            img_files_mapping = {}
            for img_file in self.im_files:
                img_id = int(Path(img_file).stem)
                img_files_mapping[img_id] = img_file

            self.im_files = [] # reset

            for idx in tqdm(range(len(annotations)), desc = 'Loading annotations'):
                img_id = int(annotations[idx]['image_id'])

                #if not is_train:
                    # Only use in val
                if img_id not in self.all_caps:
                    self.all_caps[img_id] = [annotations[idx]['caption']]
                else:
                    self.all_caps[img_id].append(annotations[idx]['caption'])

                # encode captions
                # cap, cap_mask = tokenizer.get_encoded_caption_and_mask(annotations[idx]['caption'], (hyp['caption_max_length'] + 1 if (hyp is not None) else 128 + 1))
                cap, cap_mask = tokenizer.get_encoded_caption_and_mask(annotations[idx]['caption'], (hyp['caption_max_len'] + 1 if (hyp is not None) else 128 + 1))

                if (not is_train) and (img_files_mapping[img_id] in self.im_files):
                    # TODO: add a controller for get "all anns" or "only image list" when validation
                    continue

                ann = annotations[idx]
                ann['caption'] = cap
                ann['caption_mask'] = cap_mask

                # mapping image
                ann['image'] = img_files_mapping[img_id]
                self.im_files.append(img_files_mapping[img_id]) # overwrite for mapping captions

                self.anns.append(ann)

            try:
                ann_cache = {
                    'anns': self.anns,
                    'all_caps': self.all_caps,
                    'im_files': self.im_files,
                    'hash': get_hash(ann_path),
                }
                np.save(ann_cache_path, ann_cache)  # save cache for next time
                ann_cache_path.with_suffix('.cache.npy').rename(ann_cache_path)  # remove .npy suffix
                LOGGER.info(f'New cache created: {ann_cache_path}')
            except Exception as e:
                LOGGER.warning(f'WARNING: Cache directory {ann_cache_path.parent} is not writeable: {e}')  # not writeable

            del annotations
            del img_files_mapping
            del tokenizer
        else:
            self.anns = ann_cache['anns']
            self.all_caps = ann_cache['all_caps']
            self.im_files = ann_cache['im_files'] # reset

        del ann_cache

        self.indices = range(len(self.im_files)) # reset

        ''' reset '''
        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        p = Path(path)
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle = True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items

        # labels, shapes, self.segments = zip(*cache.values())
        labels, shapes, self.segments = [], [], []
        for im_file in self.im_files:
            l, sh, seg = cache[im_file]
            labels.append(l)
            shapes.append(sh)
            self.segments.append(seg)

        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}'
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        # self.im_files = list(cache.keys())  # update
        # self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        # self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into RAM/disk for faster training (WARNING: large datasets may exceed system resources)
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

        # Check cache
        self.seg_files = img2stuff_paths(self.im_files)  # labels
        p = Path(path)
        cache_path = (p.with_suffix('') if p.is_file() else Path(self.seg_files[0]).parent)
        cache_path = Path(str(cache_path) + '_stuff').with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle = True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.seg_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_seg_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc = (prefix + d), total = n, initial = n, bar_format = TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert (0 < nf) or (not augment), f'{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        seg_labels, _, self.semantic_masks = zip(*cache.values())
        nl = len(np.concatenate(seg_labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}'

        self.seg_cls = []
        seg_labels, _, self.semantic_masks = [], [], []
        for im_file in self.im_files:
            sl, _, sm = cache[im_file]
            seg_labels.append(sl)
            self.seg_cls.append((sl[:, 0].astype(int)).tolist())
            self.semantic_masks.append(sm)

        nl = len(np.concatenate(seg_labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}'

        # Update labels
        self.seg_cls = []
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, semantic_masks) in enumerate(zip(seg_labels, self.semantic_masks)):
            self.seg_cls.append((label[:, 0].astype(int)).tolist())
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                if semantic_masks:
                    self.semantic_masks[i] = semantic_masks[j]
            if single_cls:  # single-class training, merge all classes into 0
                if semantic_masks:
                    self.semantic_masks[i][:, 0] = 0
        ''' reset '''

        self.img_transform = tv.transforms.Compose([
            tv.transforms.Lambda(self.under_max),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        img, label, path, shapes, masks, semantic_masks = super().__getitem__(index)

        cap = torch.from_numpy(self.anns[index]['caption'])
        cap_mask = torch.from_numpy(self.anns[index]['caption_mask'])
        img_id = int(self.anns[index]['image_id'])

        ## A
        #ori_img = Image.open(self.im_files[index])

        #if 'RGB' != ori_img.mode:
        #    ori_img = ori_img.convert("RGB")

        #ori_img = self.img_transform(ori_img)

        #return img, ori_img.squeeze(0), label, path, shapes, masks, \
        #    semantic_masks, cap, cap_mask, img_id

        # B
        
        ori_img, _, _ = self.load_image(index)
        ori_img, _, _ = letterbox(ori_img, self.img_size, auto=False, scaleup=False)
        
        #ori_img = cv2.imread(self.im_files[index])
        #ori_img = letterbox(ori_img, self.img_size, stride = self.stride, auto = False)[0]  # padded resize
        ori_img = ori_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        ori_img = np.ascontiguousarray(ori_img)  # contiguous

        return img, torch.from_numpy(ori_img), label, path, shapes, masks, \
            semantic_masks, cap, cap_mask, img_id

    @staticmethod
    def collate_fn(batch):
        img, ori_img, label, path, shapes, masks, semantic_masks, cap, cap_mask, img_id = zip(*batch)
        batched_masks = torch.cat(masks, 0)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()

        samples = [oi for oi in ori_img]
        ori_img, src_mask = create_src_mask(samples, is_train = True)

        batched_caps = torch.stack(cap, 0)
        batched_cap_masks = torch.stack(cap_mask, 0)
        _, lens = batched_caps.shape
        for col_num in range(lens - 1, -1, -1):  # backword
            col = batched_caps[:, col_num]
            if 1 != len(torch.bincount(col)):
                # not all [PAD]
                # to fit the longest length
                batched_caps = batched_caps[:, :col_num + 1]
                batched_cap_masks = batched_cap_masks[:, :col_num + 1]
                break

        return torch.stack(img, 0), ori_img, torch.cat(label, 0), path, shapes, batched_masks, \
            torch.stack(semantic_masks, 0), src_mask, batched_caps, batched_cap_masks, img_id

    #def get_all_caps(self, img_id):
    #    return self.all_caps[img_id]
    
    def get_all_caps(self, img_ids):
        if isinstance(img_ids, list):
            return {int(img_id): self.all_caps[int(img_id)] for img_id in img_ids}
        elif isinstance(img_ids, int):
            return self.all_caps[img_ids]

        return []

    def under_max(self, image):
        if 'RGB' != image.mode:
            image = image.convert("RGB")

        shape = np.array(image.size, dtype = np.float)
        while int(self.img_size) != int(max(shape)):
            scale = self.img_size / max(shape)
            shape = (shape * scale).astype(int)
        image = image.resize(shape)

        return image


def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index