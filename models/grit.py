import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from contextlib import contextmanager
from utils.caption.caption_utils import NestedTensor

from typing import Union, Sequence, Tuple

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]

class Module(nn.Module):

    def __init__(self):
        super(Module, self).__init__()
        self._is_stateful = False
        self._state_names = []
        self._state_defaults = dict()
        self.timestep = 0

    def register_state(self, name: str, default: TensorOrNone):
        self._state_names.append(name)
        if default is None:
            self._state_defaults[name] = None
        else:
            self._state_defaults[name] = default.clone().detach()
        self.register_buffer(name, default)

    def states(self):
        for name in self._state_names:
            yield self._buffers[name]
        for m in self.children():
            if isinstance(m, Module):
                yield from m.states()

    def apply_to_states(self, fn):
        for name in self._state_names:
            self._buffers[name] = fn(self._buffers[name])
        for m in self.children():
            if isinstance(m, Module):
                m.apply_to_states(fn)

    def _init_states(self, batch_size: int):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)
                self._buffers[name] = self._buffers[name].unsqueeze(0)
                self._buffers[name] = self._buffers[name].expand([
                    batch_size,
                ] + list(self._buffers[name].shape[1:]))
                self._buffers[name] = self._buffers[name].contiguous()

    def _reset_states(self):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)

    def enable_statefulness(self, batch_size: int):
        for m in self.children():
            if isinstance(m, Module):
                m.enable_statefulness(batch_size)
        self._init_states(batch_size)
        self._is_stateful = True

    def disable_statefulness(self):
        self.timestep = 0
        for m in self.children():
            if isinstance(m, Module):
                m.disable_statefulness()
        self._reset_states()
        self._is_stateful = False

    @contextmanager
    def statefulness(self, batch_size: int):
        self.enable_statefulness(batch_size)
        try:
            yield
        finally:
            self.disable_statefulness()


class ModuleList(nn.ModuleList, Module):
    pass


class ModuleDict(nn.ModuleDict, Module):
    pass

class Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, n_heads, dropout=0.2, n_memories=0):
        super().__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        # * adapted from Meshed-Memory Transformers; n_memories: # mem slots
        if n_memories > 0:
            self.m_k = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
            self.m_v = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))

        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_memories = n_memories
        self.d_k = d_model // n_heads

        self.apply(init_params)

    def forward(self, q, k, v, attention_mask=None):
        # q, k, v: (b, n, d_model), mask: (b, n, n)
        nq, nk = q.shape[1], k.shape[1]

        if self.n_memories > 0:
            m_k = repeat(self.m_k, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.d_k)
            m_v = repeat(self.m_v, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.n_memories)
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)

            k = torch.cat([self.fc_k(k), m_k], 1)
            v = torch.cat([self.fc_v(v), m_v], 1)
            k = rearrange(k, 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(v, 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            # if attention_weights is not None:
            # scores = torch.cat([scores[:, :, :, :nk] * attention_weights, scores[:, :, :, nk:]], dim=-1)
            if attention_mask is not None:
                scores[:, :, :, :nk] = scores[:, :, :, :nk].masked_fill(attention_mask.bool(), -np.inf)
        else:
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)
            k = rearrange(self.fc_k(k), 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(self.fc_v(v), 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # [b h nq nk]
            # if attention_weights is not None:
            # scores = scores * attention_weights
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.bool(), -np.inf)

        p_attn = torch.softmax(scores, -1)
        p_attn = self.dropout(p_attn)

        # [b h nq nk] * [b h nk dv] = [b h nq dv] -> [b nq h dv] -> [b nq h*dv]
        out = rearrange(torch.matmul(p_attn, v), 'b h nq dv -> b nq (h dv)')

        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MemoryAttention(nn.Module):

    def __init__(self, d_model, n_heads, n_memories, dropout=0.0):
        # * adapted from Meshed-Memory Transformers; n_memories: # mem slots
        super().__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        if n_memories > 0:
            self.m_k = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
            self.m_v = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_memories = n_memories
        self.d_k = d_model // n_heads

        self.apply(init_params)

    def forward(self, q, k, v, attention_mask=None, attention_weights=None):
        # q, k, v: (b, n, d_model), mask: (b, n, n) - True indicates masking

        b_s, nq = q.shape[:2]
        nk = k.shape[1]
        if self.n_memories > 0:
            m_k = repeat(self.m_k, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.d_k)
            m_v = repeat(self.m_v, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.n_memories)
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)

            k = torch.cat([self.fc_k(k), m_k], 1)
            v = torch.cat([self.fc_v(v), m_v], 1)
            k = rearrange(k, 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(v, 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                scores = torch.cat([scores[:, :, :, :nk] * attention_weights, scores[:, :, :, nk:]], dim=-1)
            if attention_mask is not None:
                scores[:, :, :, :nk] = scores[:, :, :, :nk].masked_fill(attention_mask.bool(), -np.inf)
        else:
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)
            k = rearrange(self.fc_k(k), 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(self.fc_v(v), 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # [b h nq nk]
            if attention_weights is not None:
                scores = scores * attention_weights
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.bool(), -np.inf)

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # [b h nq nk] * [b h nk dv] = [b h nq dv] -> [b nq h dv] -> [b nq h*dv]
        out = rearrange(torch.matmul(p_attn, v), 'b h nq dv -> b nq (h dv)')
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadAttention(Module):

    def __init__(self, d_model, n_heads, dropout=.1, n_memories=0, can_be_stateful=False):
        super().__init__()

        self.attention = Attention(d_model=d_model, n_heads=n_heads, dropout=dropout, n_memories=n_memories)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:  # store prev computed K & V for fast inference
            self.register_state('running_keys', torch.zeros((1, d_model)))
            self.register_state('running_values', torch.zeros((1, d_model)))

    def forward(self, queries, keys, values, attention_mask=None):
        if self.can_be_stateful and self._is_stateful:
            # keys, values:             from the current input token: [B, 1, D]
            # running_keys, values:     from prev tokens: [B, t-1, D]
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            self.running_values = torch.cat([self.running_values, values], 1)
            if self.timestep == 0:
                keys = self.running_keys = self.running_keys[:, 1:]  # [B t D]
                values = self.running_values = self.running_values[:, 1:]  # [B t D]
            else:
                keys = self.running_keys  # [B t D]
                values = self.running_values  # [B t D]

            self.timestep += 1

        out = self.attention(queries, keys, values, attention_mask)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000**(2 * dim / d_model))
    cos = torch.cos(input / 10000**(2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out


class FeedForward(nn.Module):

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        out = self.dropout(out)
        out = self.layer_norm(input + out)
        return out




def init_params(module):
    for name, param in module.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'm_' in name:  # for memory
            nn.init.normal_(param, mean=0, std=0.01)


class Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, n_heads, dropout=0.2, n_memories=0):
        super().__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        # * adapted from Meshed-Memory Transformers; n_memories: # mem slots
        if n_memories > 0:
            self.m_k = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
            self.m_v = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))

        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_memories = n_memories
        self.d_k = d_model // n_heads

        self.apply(init_params)

    def forward(self, q, k, v, attention_mask=None):
        # q, k, v: (b, n, d_model), mask: (b, n, n)
        nq, nk = q.shape[1], k.shape[1]

        if self.n_memories > 0:
            m_k = repeat(self.m_k, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.d_k)
            m_v = repeat(self.m_v, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.n_memories)
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)

            k = torch.cat([self.fc_k(k), m_k], 1)
            v = torch.cat([self.fc_v(v), m_v], 1)
            k = rearrange(k, 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(v, 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            # if attention_weights is not None:
            # scores = torch.cat([scores[:, :, :, :nk] * attention_weights, scores[:, :, :, nk:]], dim=-1)
            if attention_mask is not None:
                scores[:, :, :, :nk] = scores[:, :, :, :nk].masked_fill(attention_mask.bool(), -np.inf)
        else:
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)
            k = rearrange(self.fc_k(k), 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(self.fc_v(v), 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # [b h nq nk]
            # if attention_weights is not None:
            # scores = scores * attention_weights
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.bool(), -np.inf)

        p_attn = torch.softmax(scores, -1)
        p_attn = self.dropout(p_attn)

        # [b h nq nk] * [b h nk dv] = [b h nq dv] -> [b nq h dv] -> [b nq h*dv]
        out = rearrange(torch.matmul(p_attn, v), 'b h nq dv -> b nq (h dv)')

        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MemoryAttention(nn.Module):

    def __init__(self, d_model, n_heads, n_memories, dropout=0.0):
        # * adapted from Meshed-Memory Transformers; n_memories: # mem slots
        super().__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        if n_memories > 0:
            self.m_k = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
            self.m_v = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_memories = n_memories
        self.d_k = d_model // n_heads

        self.apply(init_params)

    def forward(self, q, k, v, attention_mask=None, attention_weights=None):
        # q, k, v: (b, n, d_model), mask: (b, n, n) - True indicates masking

        b_s, nq = q.shape[:2]
        nk = k.shape[1]
        if self.n_memories > 0:
            m_k = repeat(self.m_k, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.d_k)
            m_v = repeat(self.m_v, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.n_memories)
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)

            k = torch.cat([self.fc_k(k), m_k], 1)
            v = torch.cat([self.fc_v(v), m_v], 1)
            k = rearrange(k, 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(v, 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                scores = torch.cat([scores[:, :, :, :nk] * attention_weights, scores[:, :, :, nk:]], dim=-1)
            if attention_mask is not None:
                scores[:, :, :, :nk] = scores[:, :, :, :nk].masked_fill(attention_mask.bool(), -np.inf)
        else:
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)
            k = rearrange(self.fc_k(k), 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(self.fc_v(v), 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # [b h nq nk]
            if attention_weights is not None:
                scores = scores * attention_weights
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.bool(), -np.inf)

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # [b h nq nk] * [b h nk dv] = [b h nq dv] -> [b nq h dv] -> [b nq h*dv]
        out = rearrange(torch.matmul(p_attn, v), 'b h nq dv -> b nq (h dv)')
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadAttention(Module):

    def __init__(self, d_model, n_heads, dropout=.1, n_memories=0, can_be_stateful=False):
        super().__init__()

        self.attention = Attention(d_model=d_model, n_heads=n_heads, dropout=dropout, n_memories=n_memories)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:  # store prev computed K & V for fast inference
            self.register_state('running_keys', torch.zeros((1, d_model)))
            self.register_state('running_values', torch.zeros((1, d_model)))

    def forward(self, queries, keys, values, attention_mask=None):
        if self.can_be_stateful and self._is_stateful:
            # keys, values:             from the current input token: [B, 1, D]
            # running_keys, values:     from prev tokens: [B, t-1, D]
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            self.running_values = torch.cat([self.running_values, values], 1)
            if self.timestep == 0:
                keys = self.running_keys = self.running_keys[:, 1:]  # [B t D]
                values = self.running_values = self.running_values[:, 1:]  # [B t D]
            else:
                keys = self.running_keys  # [B t D]
                values = self.running_values  # [B t D]

            self.timestep += 1

        out = self.attention(queries, keys, values, attention_mask)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out


class BaseCaptioner(Module):

    def __init__(self):
        super(BaseCaptioner, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError

    def forward(self, images, seq, *args):
        device = images.device
        b_s = images.size(0)
        seq_len = seq.size(1)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, seq, *args, mode='teacher_forcing')
            outputs.append(out)

        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1)
        return outputs


class GeneratorLayer(Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, n_memories=0):
        super().__init__()

        self.self_att = MultiHeadAttention(d_model, n_heads, dropout, n_memories=n_memories, can_be_stateful=True)
        self.pwff = FeedForward(d_model, d_ff, dropout)


class ParallelAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, activation='sigmoid', n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)

        self.vis_att1 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.vis_att2 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.activation = activation

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):
        self_att = self.self_att(x, x, x, mask_x)
        self_att = self_att * mask_pad

        enc_att1 = self.vis_att1(self_att, y1, y1, mask_y1) * mask_pad
        enc_att2 = self.vis_att2(self_att, y2, y2, mask_y2) * mask_pad

        # [B, N, D]
        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att2], -1)))

        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2) / np.sqrt(2)
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class ConcatAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)
        self.vis_att = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)

    def forward(self, x, y, mask_pad, mask_x, mask_y):
        out = self.self_att(x, x, x, mask_x) * mask_pad
        out = self.vis_att(out, y, y, mask_y) * mask_pad
        out = self.pwff(out) * mask_pad
        return out


class SequentialAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)

        self.vis_att1 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.vis_att2 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.pwff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):
        out = self.self_att(x, x, x, mask_x) * mask_pad
        out = self.vis_att1(out, y1, y1, mask_y1) * mask_pad
        out = self.vis_att2(out, y2, y2, mask_y2) * mask_pad
        ff = self.pwff(out)
        ff = ff * mask_pad
        return ff


class CaptionGenerator(Module):
    GENERATOR_LAYER = {
        'concat': ConcatAttentionLayer,
        'parallel': ParallelAttentionLayer,
        'sequential': SequentialAttentionLayer,
    }

    def __init__(self,
                 vocab_size,
                 max_len,
                 n_layers,
                 pad_idx,
                 d_model=512,
                 n_heads=8,
                 d_ff=2048,
                 dropout=.1,
                 decoder_name='parallel',
                 cfg=None):
        super().__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)

        self.cfg = cfg
        self.decoder_name = decoder_name
        generator_layer = self.GENERATOR_LAYER[self.decoder_name]

        self.layers = ModuleList([generator_layer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.N = n_layers

        self.register_state('running_mask_x', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def get_seq_inputs(self, input):
        # input (b_s, seq_len); when use beam search: input [BB 1]
        b_s, seq_len = input.shape[:2]
        mask_pad = (input != self.pad_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_x = mask_x.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_x = mask_x + (input == self.pad_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_x = mask_x.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_x = torch.cat([self.running_mask_x, mask_x], -1)
            mask_x = self.running_mask_x

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_pad.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
        x = self.word_emb(input) + self.pos_emb(seq)

        return x, mask_x, mask_pad

    def forward(self, input, vis_inputs):
        x, mask_x, mask_pad = self.get_seq_inputs(input)

        if self.decoder_name == 'concat':
            y = torch.cat([vis_inputs['grid_feat'], vis_inputs['reg_feat']], dim=1)
            mask_y = torch.cat([vis_inputs['gri_mask'], vis_inputs['reg_mask']], dim=3)

            for layer in self.layers:
                x = layer(x, y, mask_pad, mask_x, mask_y)

        if self.decoder_name == 'sequential':
            y1 = vis_inputs['gri_feat']
            y2 = vis_inputs['reg_feat']
            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']

            for layer in self.layers:
                x = layer(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)

        if self.decoder_name == 'parallel':
            y1 = vis_inputs['gri_feat']
            y2 = vis_inputs['reg_feat']
            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']

            for layer in self.layers:
                x = layer(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


class TransformerLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, n_memories=0):
        super().__init__()

        self.mhatt = MultiHeadAttention(d_model, n_heads, dropout, n_memories=n_memories)
        self.pwff = FeedForward(d_model, d_ff, dropout)

    def forward(self, q, k, v, mask=None):
        out = self.mhatt(q, k, v, mask)
        out = self.pwff(out)
        return out


class GridFeatureNetwork(nn.Module):

    def __init__(self, n_layers, d_in=1024, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, n_memories=0):
        super().__init__()
        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_heads, d_ff, dropout, n_memories=n_memories) for _ in range(n_layers)])

    def forward(self, input, mask=None):
        out = self.layer_norm(self.dropout(F.relu(self.fc(input))))

        outs = []
        for layer in self.layers:
            out = layer(out, out, out, mask)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, mask

class Transformer(BaseCaptioner):

    def __init__(self, config=None):
        super(Transformer, self).__init__()

        self.grid_net = GridFeatureNetwork(
            n_layers=config['grid_net']['n_layers'],
            d_in=config['grid_feat_dim'],
            d_model = config['d_model'],
            n_heads = config['n_heads'],
            dropout=config['dropout'],
            d_ff=config['d_ff'],
        )
        self.cap_generator = CaptionGenerator(
            n_layers=config['cap_generator']['n_layers'],
            vocab_size=config['vocab_size'],
            max_len=config['max_len'],
            pad_idx=config['pad_idx'],
            d_model = config['d_model'],
            n_heads = config['n_heads'],
            dropout=config['dropout'],
            cfg=config['cap_generator'],
            d_ff=config['d_ff'],
        )

        self.config = config
        self.bos_idx = config['bos_idx']
        self.use_reg_feat = config['use_reg_feat']
        self.use_gri_feat = config['use_gri_feat']
        self.cached_features = False

        if self.use_gri_feat:
            self.register_state('gri_feat', None)
            self.register_state('gri_mask', None)

        if self.use_reg_feat:
            self.register_state('reg_feat', None)
            self.register_state('reg_mask', None)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                images,
                seq,
                use_beam_search=False,
                max_len=20,
                eos_idx=3,
                beam_size=5,
                out_size=1,
                return_probs=False,
                **kwargs):
        if not use_beam_search:
            vis_inputs = images

            if self.config['use_gri_feat']:
                gri_feat, _ = self.grid_net(vis_inputs['gri_feat'], vis_inputs['gri_mask'])
                vis_inputs['gri_feat'] = gri_feat[:, -1]

            dec_output = self.cap_generator(seq, vis_inputs)
            return dec_output
        else:  # Run beam_search in the following code
            batch_size, device = self.get_bs_device(images)

            # the mask of the current word (whether it != eos or not), it = 1 if != <eos>
            self.seq_mask = torch.ones((batch_size, beam_size, 1), device=device)

            # the cummulative sum of log probs up to the current word [B, Beam, 1]
            self.seq_logprob = torch.zeros((batch_size, 1, 1), device=device)

            # log probs of all beam_size selected words: [[B, Beam, 1] * max_len]
            self.log_probs = []
            self.selected_words = None

            if return_probs:
                self.all_log_probs = []

            # selected words at each timestep: [[B, Beam, 1] * max_len]
            outputs = []

            with self.statefulness(batch_size):
                for timestep in range(max_len):
                    images, outputs = self.iter(
                        timestep=timestep,
                        samples=images,
                        outputs=outputs,
                        return_probs=return_probs,
                        batch_size=batch_size,
                        beam_size=beam_size,
                        eos_idx=eos_idx,
                        **kwargs,
                    )

            # Sort result
            seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)

            # sum log_probs = seq_logprob
            # outputs = log_probs shape = [B, Beam, Len], the following is to sorted the order of which sequence.
            outputs = torch.cat(outputs, -1)  # [B, Beam, Len]
            outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, max_len))
            log_probs = torch.cat(self.log_probs, -1)
            log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_len))
            if return_probs:
                all_log_probs = torch.cat(self.all_log_probs, 2)
                all_log_probs = torch.gather(
                    all_log_probs, 1,
                    sort_idxs.unsqueeze(-1).expand(batch_size, beam_size, max_len, all_log_probs.shape[-1]))

            outputs = outputs.contiguous()[:, :
                                           out_size]  # [B Beam Len] -> [B, :topk, Len] select only the top k sentences
            log_probs = log_probs.contiguous()[:, :out_size]  # [B Beam Len] -> [B Len] select only the top k sentences
            if out_size == 1:
                outputs = outputs.squeeze(1)  # [B :topk, len] = [B, len] if topk = 1
                log_probs = log_probs.squeeze(1)

            if return_probs:
                return outputs, log_probs, all_log_probs
            else:
                return outputs, log_probs

    def step(self, timestep, prev_output, samples, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if timestep == 0:
                vis_inputs = samples

                if self.config['use_gri_feat']:
                    self.gri_feat, self.gri_mask = self.grid_net(vis_inputs['gri_feat'], vis_inputs['gri_mask'])
                    self.gri_feat = self.gri_feat[:, -1]

                if self.config['use_reg_feat']:
                    self.reg_feat = vis_inputs['reg_feat']
                    self.reg_mask = vis_inputs['reg_mask']

                # If t = 0, enc_output = [B, N, D], init_tokens = [B, 1]
                # Else t > 0, enc_output = [BB, N, D], it = prev_output (t-1) = [BB, 1]
                _feat = getattr(self, 'gri_feat', self.reg_feat)
                it = _feat.data.new_full((_feat.shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        vis_inputs = {}
        if self.config['use_gri_feat']:
            vis_inputs['gri_feat'] = self.gri_feat
            vis_inputs['gri_mask'] = self.gri_mask

        if self.config['use_reg_feat']:
            vis_inputs['reg_feat'] = self.reg_feat
            vis_inputs['reg_mask'] = self.reg_mask

        return self.cap_generator(it, vis_inputs)

    def get_bs_device(self, samples):
        if isinstance(samples, dict):
            key = 'gri_feat' if 'gri_feat' in samples else 'reg_feat'
            batch_size = samples[key].shape[0]
            device = samples[key].device
        elif isinstance(samples, NestedTensor):
            batch_size = samples.tensors.shape[0]
            device = samples.tensors.device
        return batch_size, device

    def init_state(self, batch_size, device):
        return [torch.zeros((batch_size, 0), dtype=torch.long, device=device), None, None]

    def select(self, t, candidate_logprob, beam_size, **kwargs):
        candidate_logprob = rearrange(candidate_logprob, 'B Beam V -> B (Beam V)')
        selected_logprob, selected_idx = torch.sort(candidate_logprob, -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob  # [B Beam]

    def _expand_state(self, selected_beam, cur_beam_size, batch_size, beam_size):

        def fn(tensor):
            shape = [int(sh) for sh in tensor.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            tensor = torch.gather(tensor.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                                  beam.expand(*([batch_size, beam_size] + shape[1:])))
            tensor = tensor.view(*([-1] + shape[1:]))
            return tensor

        return fn

    def iter(self, timestep, samples, outputs, return_probs, batch_size, beam_size=5, eos_idx=3, **kwargs):
        cur_beam_size = 1 if timestep == 0 else beam_size

        word_logprob = self.step(timestep, self.selected_words, samples, None, mode='feedback', **kwargs)
        word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)  # [BB V] -> [B Beam V]
        candidate_logprob = self.seq_logprob + word_logprob  # [B Beam V]

        # Mask sequence if it reaches EOS
        if timestep > 0:
            _selected_words = self.selected_words.view(batch_size, cur_beam_size)  # [BB, 1] -> [B Beam]
            # mask = 0 if it is eos, else 1.
            mask = repeat((_selected_words != eos_idx).float(), 'B Beam -> B Beam V', V=1)
            self.seq_mask = self.seq_mask * mask  # [B Beam V] V=1
            word_logprob = word_logprob * self.seq_mask  # [B Beam V]
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999  # [B Beam V]
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)  # [B Beam V]
            # After <EOS>, we want to make all predictions to <UNK>.
            # When decoding, we will remove all predictions after <EOS>

        selected_idx, selected_logprob = self.select(timestep, candidate_logprob, beam_size, **kwargs)
        selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='floor')  # [B Beam]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]  # [B Beam]

        # save the states of the selected beam
        self.apply_to_states(self._expand_state(selected_beam, cur_beam_size, batch_size, beam_size))  # [BB, ...]

        self.seq_logprob = repeat(selected_logprob, 'B Beam -> B Beam L', L=1)
        beam_exp = repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.seq_mask = torch.gather(self.seq_mask, 1, beam_exp)
        outputs = [torch.gather(o, 1, beam_exp) for o in outputs]
        outputs.append(repeat(selected_words, 'B Beam -> B Beam L', L=1))

        if return_probs:
            if timestep == 0:
                # [B Beam V] -> [B Beam 1 V]
                self.all_log_probs.append(word_logprob.expand((batch_size, beam_size, -1)).unsqueeze(2))
            else:  # [B Beam V] -> [B Beam 1 V]
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        beam_exp = repeat(selected_beam, 'B Beam -> B Beam V', V=word_logprob.shape[-1])
        this_word_logprob = torch.gather(word_logprob, 1, beam_exp)
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))

        beam_exp = repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.log_probs = [torch.gather(o, 1, beam_exp) for o in self.log_probs]

        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)  # [B*Beam, 1]

        return samples, outputs
