import numpy as np
import torch

from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torch import Tensor
from transformers import BertTokenizer
from typing import List, Optional

# Tokenizer
class bert_tokenizer():
    def __init__(self, model: str = 'bert-base-uncased', vocab = None, do_lower = False):
        if 'custom' == model:
            self.tokenizer = custom_tokenizer(vocab, do_lower_case = do_lower)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model, do_lower = do_lower)

    def get_encoded_caption_and_mask(self, cap, max_length):
        cap_encoded = self.tokenizer.encode_plus(
            cap,
            max_length = max_length,
            add_special_tokens = True,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_token_type_ids = False,
            truncation = True,
        )
        caption = np.array(cap_encoded['input_ids'])
        cap_mask = (1 - np.array(cap_encoded['attention_mask'])).astype(bool)

        return caption, cap_mask

    def get_decoded_caption(self, cap, skip_special_tokens = True):
        result = self.tokenizer.decode(cap, skip_special_tokens = skip_special_tokens)
        return result.capitalize()

    def get_batch_decoded_captions(self, caps, skip_special_tokens = True):
        results = self.tokenizer.batch_decode(caps, skip_special_tokens = skip_special_tokens)
        return [result.capitalize() for result in results]

class custom_tokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case = True,
        do_basic_tokenize = True,
        never_split = None,
        unk_token = "[UNK]",
        sep_token = "[SEP]",
        pad_token = "[PAD]",
        cls_token = "[CLS]",
        mask_token = "[MASK]",
        tokenize_chinese_chars = True,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case = do_lower_case,
            do_basic_tokenize = do_basic_tokenize,
            never_split = never_split,
            unk_token = unk_token,
            sep_token = sep_token,
            pad_token = pad_token,
            cls_token = cls_token,
            mask_token = mask_token,
            tokenize_chinese_chars = tokenize_chinese_chars,
            model_max_length = 512,
            **kwargs,
        )

# https://huggingface.co/bert-base-uncased#model-variations
def get_tokenizer(model: str = 'bert-base-uncased', do_lower = False):
    return BertTokenizer.from_pretrained(model, do_lower = do_lower)

def get_start_token(tokenizer = None):
    if tokenizer is None:
        tokenizer = get_tokenizer()

    if isinstance(tokenizer, bert_tokenizer):
        tokenizer = tokenizer.tokenizer

    return tokenizer.convert_tokens_to_ids(tokenizer._cls_token)

def get_end_token(tokenizer = None):
    if tokenizer is None:
        tokenizer = get_tokenizer()

    if isinstance(tokenizer, bert_tokenizer):
        tokenizer = tokenizer.tokenizer

    return tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

def get_pad_token(tokenizer = None):
    if tokenizer is None:
        tokenizer = get_tokenizer()

    if isinstance(tokenizer, bert_tokenizer):
        tokenizer = tokenizer.tokenizer

    return tokenizer.convert_tokens_to_ids(tokenizer._pad_token)

def get_all_special_token(tokenizer = None, with_pad = True):
    tokens = {
        'start': get_start_token(tokenizer = tokenizer),
        'end': get_end_token(tokenizer = tokenizer),
    }

    if with_pad:
        tokens['pad'] = get_pad_token(tokenizer = tokenizer)

    return tokens

# nested tensor
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking = False):
        cast_tensor = self.tensors.to(device, non_blocking = non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking = non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

# caption mask
def create_caption_and_mask(max_length, batch_size = 1):
    caption_template = torch.zeros((batch_size, max_length), dtype = torch.long)
    mask_template = torch.ones((batch_size, max_length), dtype = torch.bool)

    caption_template[:, 0] = get_start_token()
    mask_template[:, 0] = False

    return caption_template, mask_template

# image mask
def create_src_mask(src: List[Tensor], is_train = False):
    def _max_by_axis(the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    max_dim = max(src[0].shape)
    max_size = [3, max_dim, max_dim] if (is_train) else _max_by_axis([list(img.shape) for img in src])
#     max_size = _max_by_axis([list(img.shape) for img in src])
    batch_shape = [len(src)] + max_size
    b, _, h, w = batch_shape
    dtype = src[0].dtype
    device = src[0].device
    tensor = torch.zeros(batch_shape, dtype = dtype, device = device)
    mask = torch.ones((b, h, w), dtype = torch.bool, device = device)
    for img, pad_img, m in zip(src, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], :img.shape[2]] = False
    return tensor, mask

class MaxWHResize:
    def __init__(self, size):
        self.size = size
        self.max_h = size[0]
        self.max_w = size[1]

    def __call__(self, x):
        w, h = x.size
        scale = min(self.max_w / w, self.max_h / h)
        neww = int(w * scale)
        newh = int(h * scale)
        return x.resize((neww, newh), resample = Image.BICUBIC)

# https://huggingface.co/docs/transformers/v4.22.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus
def get_encoded_caption_and_mask(cap, max_length):
    tokenizer = get_tokenizer()
    cap_encoded = tokenizer.encode_plus(
        cap,
        max_length = max_length,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_token_type_ids = False,
        truncation = True,
    )
    caption = np.array(cap_encoded['input_ids'])
    cap_mask = (1 - np.array(cap_encoded['attention_mask'])).astype(bool)

    return caption, cap_mask

# Evaluation
def eval_cap(gt_json: str, ref_json: str):
    coco = COCO(gt_json)
    coco_system_captions = coco.loadRes(ref_json)
    coco_eval = COCOEvalCap(coco, coco_system_captions)
    coco_eval.params['image_id'] = coco_system_captions.getImgIds()

    coco_eval.evaluate()

    print('\nScores:')
    print('=======')
    for metric, score in coco_eval.eval.items():
        print('{}: {:.3f}'.format(metric, score))

    return coco_eval.eval

def cal_bleu4(references, hypotheses):
    print('Calculate BLEU-4 scores....')
    bleu4 = corpus_bleu(references, hypotheses)
    print('Bleu4: ', bleu4)
    return bleu4