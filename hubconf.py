import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import torch


def _ensure_repo_on_path():
    repo_root = Path(__file__).resolve().parent
    for candidate in (repo_root, repo_root.parent):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return repo_root


_ensure_repo_on_path()


def _load_repo_module(module_name, relative_path):
    repo_root = Path(__file__).resolve().parent
    module_path = repo_root / relative_path
    if not module_path.exists():
        raise ModuleNotFoundError(f'Cannot find module file: {module_path}')

    parent_name = module_name.rpartition('.')[0]
    if parent_name:
        parent_dir = str(repo_root / parent_name.replace('.', '/'))
        parent_module = sys.modules.get(parent_name)
        if parent_module is None:
            parent_module = types.ModuleType(parent_name)
            parent_module.__path__ = [parent_dir]
            sys.modules[parent_name] = parent_module
        elif hasattr(parent_module, '__path__'):
            existing_paths = list(parent_module.__path__)
            if parent_dir not in existing_paths:
                parent_module.__path__ = existing_paths + [parent_dir]
        else:
            parent_module.__path__ = [parent_dir]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f'Cannot create import spec for {module_name}')

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_repo_path(path):
    if not path:
        return path

    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)

    repo_root = Path(__file__).resolve().parent
    candidate = (repo_root / path_obj).resolve()
    if candidate.exists():
        return str(candidate)

    cwd_candidate = (Path.cwd() / path_obj).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return str(candidate)


def _create_src_mask(src):
    from utils.caption.caption_utils import create_src_mask

    if torch.is_tensor(src):
        src = [src]
    elif not isinstance(src, (list, tuple)):
        raise TypeError('src must be a tensor or a list of tensors')

    if 0 == len(src):
        raise ValueError('src must not be empty')

    return create_src_mask(src)


def _stuff_to_panoptic(self, stuff_masks, overlap_mask = None):
    from utils.coco_utils import getCocoIds, getMappingId, getMappingIndex, idToPanopticId

    if not torch.is_tensor(stuff_masks):
        stuff_masks = torch.as_tensor(stuff_masks)

    if 4 == stuff_masks.dim():
        stuff_masks = stuff_masks[0]  # [b, class, h, w] -> [class, h, w]
    if 3 != stuff_masks.dim():
        raise ValueError('stuff_masks must be a 3D tensor of shape [class, h, w]')

    device = getattr(self, 'device', None)
    if device is None:
        try:
            device = next(self.parameters()).device
        except Exception:
            device = torch.device('cpu')

    stuff_masks = stuff_masks.to(device = device)
    instance_num = len(getCocoIds(name = 'instances'))
    panoptic_ids = getCocoIds(name = 'panoptic')
    panoptic_num = len(panoptic_ids)
    panoptic_stuff_num = panoptic_num - instance_num

    if stuff_masks.shape[0] <= instance_num:
        return torch.zeros((0, *stuff_masks.shape[1 :]), dtype = torch.bool, device = device)

    stuff_masks = stuff_masks[instance_num : ].to(device = device)
    panoptic_stuff_masks = torch.zeros((panoptic_stuff_num, *stuff_masks.shape[1 :]), dtype = torch.bool, device = device)

    if overlap_mask is None:
        overlap_mask = torch.zeros(stuff_masks.shape[1 :], dtype = torch.bool, device = device)
    else:
        overlap_mask = overlap_mask.to(device = device, dtype = torch.bool)

    for idx, stuff_mask in enumerate(stuff_masks):
        stuff_id = getMappingId(idx + instance_num)
        panoptic_stuff_id = idToPanopticId(stuff_id)
        if 0 == panoptic_stuff_id:
            continue

        panoptic_stuff_idx = getMappingIndex(panoptic_stuff_id, name = 'panoptic') - instance_num
        if (0 <= panoptic_stuff_idx) and (panoptic_stuff_idx < panoptic_stuff_masks.shape[0]):
            mask = (0 < stuff_mask).to(dtype = torch.bool)
            if overlap_mask.shape == mask.shape:
                mask = torch.logical_and(mask, torch.logical_not(overlap_mask))
            panoptic_stuff_masks[panoptic_stuff_idx] = torch.logical_or(panoptic_stuff_masks[panoptic_stuff_idx], mask)

    return panoptic_stuff_masks


def _process(self, im, augment = False, visualize = False, imgsz = 640, conf_thres = 0.25, iou_thres = 0.45):
    import cv2
    import numpy as np
    import torch.nn.functional as F

    from utils.augmentations import letterbox
    from utils.caption.caption_utils import bert_tokenizer
    from utils.coco_utils import getCocoIds, getCocoName, getMappingId
    from utils.general import non_max_suppression, scale_boxes
    from utils.segment.general import process_mask

    device = getattr(self, 'device', None)
    if device is None:
        try:
            device = next(self.parameters()).device
        except Exception:
            device = torch.device('cpu')

    if isinstance(im, (str, bytes)):
        im0 = cv2.imread(str(im))
        if im0 is None:
            raise FileNotFoundError(f'Could not read image: {im}')
    else:
        if torch.is_tensor(im):
            im = im.detach().cpu().numpy()

        if not isinstance(im, np.ndarray):
            im = np.asarray(im)

        if (4 == im.ndim) and (1 == im.shape[0]):
            im = im[0]  # remove batch

        if (3 == im.ndim) and (1 <= im.shape[0] <= 4) and (im.shape[-1] not in (1, 3, 4)):
            im = np.transpose(im, (1, 2, 0))  # (c, h, w) -> (h, w, c)

        if 2 == im.ndim:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # gray to BGR

        if (3 == im.ndim) and (1 == im.shape[2]):
            im = np.repeat(im, 3, axis = 2)  # gray to BGR

        im0 = im.copy()

    if np.uint8 != im0.dtype:
        im0 = im0.astype(np.uint8)

    im_letterbox, _, _ = letterbox(im0, new_shape = (imgsz, imgsz), auto = False, scaleup = False)
    src_img = torch.from_numpy(im_letterbox).permute(2, 0, 1).contiguous().to(device = device)
    input_img = torch.from_numpy(im_letterbox).permute(2, 0, 1).contiguous().float().to(device = device) / 255.0
    input_img = input_img[None]

    if hasattr(self, 'set_src_mask'):
        self.set_src_mask([src_img])

    with torch.no_grad():
        output = self(input_img, augment = augment, visualize = visualize)

    if isinstance(output, dict):
        pred, out = output['detect'][: 2]
        pred_caption = output.get('captions')
    else:
        pred, out = output[: 2]
        pred_caption = None

    proto = out[2]
    psemasks = out[3]

    pred = non_max_suppression(pred, conf_thres = conf_thres, iou_thres = iou_thres, classes = None, agnostic = False, max_det = 1000, nm = 32)
    pred = pred[0] if pred else None

    if pred is not None and 0 != len(pred):
        det = pred.detach().clone()
        instance_masks = process_mask(proto[0], det[:, 6 :], det[:, : 4], input_img.shape[-2 :], upsample = True)
        if 2 == instance_masks.dim():
            instance_masks = instance_masks.unsqueeze(0)
        instance_boxes = scale_boxes((im_letterbox.shape[0], im_letterbox.shape[1]), det[:, :4], (im0.shape[0], im0.shape[1])).round()
    else:
        det = None
        instance_masks = None
        instance_boxes = None

    if psemasks is not None:
        if 4 == psemasks.dim():
            psemasks = psemasks[0]
        if 3 != psemasks.dim():
            psemasks = psemasks.squeeze(0)
        if 3 == psemasks.dim():
            psemasks = psemasks.unsqueeze(0)
        semantic_mask = psemasks.to(device = device).float()
        semantic_mask = F.interpolate(semantic_mask, size = (im0.shape[0], im0.shape[1]), mode = 'bilinear', align_corners = False)
        semantic_flat = semantic_mask.flatten(start_dim = 1).permute(1, 0)
        max_idx = semantic_flat.argmax(1)
        semantic_one_hot = torch.zeros(semantic_flat.shape, device = device).scatter(1, max_idx.unsqueeze(1), 1.0)
        semantic_one_hot = semantic_one_hot.permute(1, 0).reshape(semantic_mask.shape)
    else:
        semantic_one_hot = None

    label_ids = []
    mask_list = []
    box_list = []
    occupied_mask = torch.zeros((im0.shape[0], im0.shape[1]), dtype = torch.bool, device = device)

    if det is not None:
        for idx, (mask, box) in enumerate(zip(instance_masks, instance_boxes)):
            mask = mask.detach().to(device = device)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size = (im0.shape[0], im0.shape[1]), mode = 'bilinear', align_corners = False)[0, 0]
            mask = (0.5 < mask).to(dtype = torch.bool)
            if 0 == mask.sum():
                continue
            mask = torch.logical_and(mask, torch.logical_not(occupied_mask))
            if 0 == mask.sum():
                continue
            occupied_mask = torch.logical_or(occupied_mask, mask)
            label_ids.append(int(getMappingId(int(det[idx, 5]))))
            mask_list.append(mask.cpu())
            box_list.append(torch.tensor([float(x) for x in box.tolist()], dtype = torch.float32, device = 'cpu'))

    if semantic_one_hot is not None:
        panoptic_stuff_masks = self.stuff_to_panoptic(semantic_one_hot, overlap_mask = occupied_mask)
        panoptic_stuff_masks = panoptic_stuff_masks.to(dtype = torch.bool)
        for idx, mask in enumerate(panoptic_stuff_masks):
            mask = mask.detach().to(device = device)
            if 0 == mask.sum():
                continue
            mask = torch.logical_and(mask, torch.logical_not(occupied_mask))
            if 0 == mask.sum():
                continue
            occupied_mask = torch.logical_or(occupied_mask, mask)
            panoptic_id = getMappingId(idx + len(getCocoIds(name = 'instances')), name = 'panoptic')
            label_ids.append(int(panoptic_id))
            mask_list.append(mask.cpu())
            ys, xs = torch.where(mask)
            if 0 == len(xs):
                box_list.append(torch.tensor([0.0, 0.0, 0.0, 0.0], dtype = torch.float32, device = 'cpu'))
            else:
                box_list.append(torch.tensor([float(xs.min().item()), float(ys.min().item()), float(xs.max().item()), float(ys.max().item())], dtype = torch.float32, device = 'cpu'))

    label_names = [getCocoName(label_id) for label_id in label_ids]
    labels = torch.tensor(label_ids, dtype = torch.int64) if (0 != len(label_ids)) else torch.empty((0,), dtype = torch.int64)
    masks = torch.stack(mask_list, dim = 0) if (0 != len(mask_list)) else torch.empty((0, im0.shape[0], im0.shape[1]), dtype = torch.bool)
    boxes = torch.stack(box_list, dim = 0) if (0 != len(box_list)) else torch.empty((0, 4), dtype = torch.float32)

    caption_text = None
    if pred_caption is not None:
        if isinstance(pred_caption, (list, tuple)):
            pred_caption = pred_caption[0]
        if torch.is_tensor(pred_caption):
            pred_caption = pred_caption.detach().cpu()
        if torch.is_tensor(pred_caption):
            if 1 < pred_caption.dim():
                if 1 == pred_caption.shape[0]:
                    pred_caption = pred_caption[0]
                else:
                    pred_caption = pred_caption[0]
            pred_caption = pred_caption.tolist()
        if not isinstance(pred_caption, list):
            pred_caption = [pred_caption]
        if hasattr(self, 'hyp') and ('caption_tokenizer' in self.hyp) and ('custom' == self.hyp['caption_tokenizer']):
            vocab_path = self.hyp.get('caption_vocab_path')
            if isinstance(vocab_path, str):
                vocab_path = _resolve_repo_path(vocab_path)
                self.hyp['caption_vocab_path'] = vocab_path
            tokenizer = bert_tokenizer(model = self.hyp['caption_tokenizer'], vocab = vocab_path, do_lower = True)
        else:
            tokenizer = bert_tokenizer(do_lower = True)
        try:
            caption_text = tokenizer.get_decoded_caption(pred_caption, skip_special_tokens = True)
        except Exception:
            caption_text = ''

    return labels, label_names, masks, boxes, caption_text


def _get_caption_module(model):
    container = model
    if hasattr(container, 'module'):
        container = container.module
    if hasattr(container, 'model'):
        container = container.model
    if hasattr(container, 'model'):
        container = container.model
    return container


def _find_caption_layer(model):
    from utils.model_utils import find_layer

    container = _get_caption_module(model)
    layer_idx = find_layer(container, ['Caption', 'Grit'])
    return container, layer_idx


def _set_src_mask(self, src, use_beam_search = True, beam_size = 5, out_size = 1, return_probs = False):
    _, src_mask = _create_src_mask(src)
    container, layer_idx = _find_caption_layer(self)
    if layer_idx is None:
        raise AttributeError('No caption layer found in model')

    device = getattr(self, 'device', None)
    if device is None:
        try:
            device = next(self.parameters()).device
        except Exception:
            device = torch.device('cpu')

    container[layer_idx].set_params(
        src_mask.to(device),
        None,
        None,
        use_beam_search = use_beam_search,
        beam_size = beam_size,
        out_size = out_size,
        return_probs = return_probs,
    )
    return src_mask


def caption(weights = 'gyolo.pt', autoshape = True, _verbose = True, device = None):
    from pathlib import Path

    _ensure_repo_on_path()
    common_module = _load_repo_module('models.common', 'models/common.py')
    DetectMultiBackend = common_module.DetectMultiBackend
    from utils.downloads import attempt_download
    from utils.general import LOGGER, check_requirements, logging
    from utils.torch_utils import select_device

    if not _verbose:
        LOGGER.setLevel(logging.WARNING)

    requirements_path = Path(__file__).resolve().parent / 'requirements.txt'
    if requirements_path.exists():
        install_result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path)],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
        )
        if 0 != install_result.returncode:
            LOGGER.warning('WARNING ⚠️ Failed to install requirements from requirements.txt automatically.')
            if install_result.stdout:
                LOGGER.warning(install_result.stdout)

    check_requirements(exclude = ('opencv-python', 'tensorboard', 'thop'))

    if not weights:
        weights = 'gyolo.pt'

    weight_path = Path(weights)
    if not weight_path.is_absolute():
        candidate = Path(__file__).resolve().parent / 'weights' / weight_path.name
        if candidate.exists():
            weight_path = candidate
        else:
            candidate = Path(__file__).resolve().parent / weight_path
            if candidate.exists():
                weight_path = candidate

    if not weight_path.exists():
        weight_path = attempt_download(str(weight_path))

    data_path = Path(__file__).resolve().parent / 'data' / 'coco.yaml'
    device = select_device(device)
    model = DetectMultiBackend(
        str(weight_path),
        device = device,
        dnn = False,
        data = str(data_path),
        fp16 = False,
    )

    if not _verbose:
        LOGGER.setLevel(logging.INFO)

    model = model.to(device)
    setattr(model, 'create_src_mask', _create_src_mask)
    setattr(model, 'set_src_mask', _set_src_mask.__get__(model, type(model)))
    setattr(model, 'stuff_to_panoptic', _stuff_to_panoptic.__get__(model, type(model)))
    setattr(model, 'process', _process.__get__(model, type(model)))
    return model


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Creates or loads a YOLO model

    Arguments:
        name (str): model name 'yolov3' or path 'path/to/best.pt'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLO .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLO model
    """
    from pathlib import Path

    _ensure_repo_on_path()
    common_module = _load_repo_module('models.common', 'models/common.py')
    AutoShape = common_module.AutoShape
    DetectMultiBackend = common_module.DetectMultiBackend
    experimental_module = _load_repo_module('models.experimental', 'models/experimental.py')
    attempt_load = experimental_module.attempt_load
    yolo_module = _load_repo_module('models.yolo', 'models/yolo.py')
    ClassificationModel = yolo_module.ClassificationModel
    DetectionModel = yolo_module.DetectionModel
    SegmentationModel = yolo_module.SegmentationModel
    from utils.downloads import attempt_download
    from utils.general import LOGGER, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(exclude=('opencv-python', 'tensorboard', 'thop'))
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' and not name.is_dir() else name  # checkpoint path
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # detection model
                if autoshape:
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning('WARNING ⚠️ YOLO ClassificationModel is not yet AutoShape compatible. '
                                       'You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224).')
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning('WARNING ⚠️ YOLO SegmentationModel is not yet AutoShape compatible. '
                                       'You will not be able to run inference with this model.')
                    else:
                        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            except Exception:
                model = attempt_load(path, device=device, fuse=False)  # arbitrary model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]  # model.yaml path
            model = DetectionModel(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if not verbose:
            LOGGER.setLevel(logging.INFO)  # reset to default
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e


def custom(path='path/to/model.pt', autoshape=True, _verbose=True, device=None):
    # YOLO custom or local model
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from utils.general import cv2, print_args

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo', help='model name')
    opt = parser.parse_args()
    print_args(vars(opt))

    # Model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    # model = custom(path='path/to/model.pt')  # custom

    # Images
    imgs = [
        'data/images/zidane.jpg',  # filename
        Path('data/images/zidane.jpg'),  # Path
        'https://ultralytics.com/images/zidane.jpg',  # URI
        cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
        Image.open('data/images/bus.jpg'),  # PIL
        np.zeros((320, 640, 3))]  # numpy

    # Inference
    results = model(imgs, size=320)  # batched inference

    # Results
    results.print()
    results.save()
