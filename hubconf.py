import subprocess
import sys

import torch


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

    from models.common import DetectMultiBackend
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

    from models.common import AutoShape, DetectMultiBackend
    from models.experimental import attempt_load
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
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
