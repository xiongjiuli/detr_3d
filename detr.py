# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn
import sys
import torchio as tio

# sys.path.append('D:\Work_file\detr_my\\nets\\backbone.py')
# print(sys.path)
# from . import ops
# from .backbone import build_backbone, FrozenBatchNorm2d
# from .ops import NestedTensor, nested_tensor_from_tensor_list, unused
# from .transformer import build_transformer

from IPython import embed





'''
the transformer
'''
#* =========================================================================================

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        # 625, batch_size, 256 => ...(x6)... => 625, batch_size, 256
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Self-Attention模块
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN模块
        # Implementation of Feedforward model
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 添加位置信息
        # 625, batch_size, 256 => 625, batch_size, 256
        q = k = self.with_pos_embed(src, pos)
        # 使用自注意力机制模块
        # 625, batch_size, 256 => 625, batch_size, 256
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # 添加残差结构
        # 625, batch_size, 256 => 625, batch_size, 256
        src = src + self.dropout1(src2)
        
        # 添加FFN结构
        # 625, batch_size, 256 => 625, batch_size, 2048 => 625, batch_size, 256
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 添加残差结构
        # 625, batch_size, 256 => 625, batch_size, 256
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        # q自己做一个self-attention
        self.self_attn      = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # q、k、v联合做一个self-attention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN模块
        # Implementation of Feedforward model
        self.linear1        = nn.Linear(d_model, dim_feedforward)
        self.dropout        = nn.Dropout(dropout)
        self.linear2        = nn.Linear(dim_feedforward, d_model)

        self.norm1          = nn.LayerNorm(d_model)
        self.norm2          = nn.LayerNorm(d_model)
        self.norm3          = nn.LayerNorm(d_model)
        self.dropout1       = nn.Dropout(dropout)
        self.dropout2       = nn.Dropout(dropout)
        self.dropout3       = nn.Dropout(dropout)

        self.activation         = _get_activation_fn(activation)
        self.normalize_before   = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        #---------------------------------------------#
        #   q自己做一个self-attention
        #---------------------------------------------#
        # tgt + query_embed
        # 100, batch_size, 256 => 100, batch_size, 256
        q = k = self.with_pos_embed(tgt, query_pos)
        # q = k = v = 100, batch_size, 256 => 100, batch_size, 256
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # 添加残差结构
        # 100, batch_size, 256 => 100, batch_size, 256
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        #---------------------------------------------#
        #   q、k、v联合做一个self-attention
        #---------------------------------------------#
        # q = 100, batch_size, 256, k = 625, batch_size, 256, v = 625, batch_size, 256
        # 输出的序列长度以q为准 => 100, batch_size, 256
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # 添加残差结构
        # 100, batch_size, 256 => 100, batch_size, 256
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
         
        #---------------------------------------------#
        #   做一个FFN
        #---------------------------------------------#
        # 100, batch_size, 256 => 100, batch_size, 2048 => 100, batch_size, 256
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        # 定义用到的transformer的encoder层，然后定义使用到的norm层
        encoder_layer   = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm    = nn.LayerNorm(d_model) if normalize_before else None
        # 构建Transformer的Encoder，一共有6层
        self.encoder    = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 定义用到的transformer的decoder层，然后定义使用到的norm层
        decoder_layer   = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm    = nn.LayerNorm(d_model)
        # 构建Transformer的Decoder，一共有6层
        self.decoder    = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.d_model    = d_model
        self.nhead      = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, d, h, w = src.shape
        # embed()
        # batch_size, 256, 25, 25 => batch_size, 256, 625 => 625, batch_size, 256
        src         = src.flatten(2).permute(2, 0, 1)
        # batch_size, 256, 25, 25 => batch_size, 256, 625 => 625, batch_size, 256
        pos_embed   = pos_embed.flatten(2).permute(2, 0, 1)
        # 100, 256 => 100, 1, 256 => 100, batch_size, 256
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # batch_size, 25, 25 => batch_size, 625
        mask        = mask.flatten(1)
        # embed()
        # 100, batch_size, 256
        tgt         = torch.zeros_like(query_embed)
        
        # 625, batch_size, 256 => 625, batch_size, 256
        memory      = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # 625, batch_size, 256 => 6, 100, batch_size, 256
        hs          = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        # 6, 100, batch_size, 256 => 6, batch_size, 100, 256
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, d, h, w)

def build_transformer(hidden_dim=256, dropout=0.1, nheads=8, dim_feedforward=2048, enc_layers=6, dec_layers=6, pre_norm=True):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )















'''
the ops
'''
#* ==============================================================================================
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
import torch.distributed as dist
import torchvision
from torch import Tensor
# from torchvision.ops.boxes import box_area



def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = torch.clip(boxes, 0, 1)
    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])




def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)

def box_xyzxyz_to_cxcyczwhd(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
         (x1 - x0), (y1 - y0), (z1 - z0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    whd = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    whd = (rb - lt).clamp(min=0)  # [N,M,2]
    area = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]

    return iou - (area - union) / area

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        try:
            if torchvision._is_tracing():
                # nested_tensor_from_tensor_list() does not export well to ONNX
                # call _onnx_nested_tensor_from_tensor_list() instead
                return _onnx_nested_tensor_from_tensor_list(tensor_list)
        except:
            pass

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def wrap(callback):
    def f(*args, **kwargs):
        r = callback(*args, **kwargs)
        return r
    return f

unused = torch.jit.unused if hasattr(torch.jit, "unused") else wrap

# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res











'''
the Backbone 
'''
#* ==============================================================================================

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Dict, List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

# from .ops import NestedTensor, is_main_process
from resnet_3d import Backbone
from IPython import embed



class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class PositionEmbeddingSine(nn.Module):
    """
    这是一个更标准的位置嵌入版本,按照sine进行分布
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats  = num_pos_feats
        self.temperature    = temperature
        self.normalize      = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x           = tensor_list.tensors
        # print(f'x.shape is {x.shape}')
        mask        = tensor_list.mask[:, 0 : (x.shape[2]), 0 : (x.shape[3]), 0 : (x.shape[4])]
        assert mask is not None
        not_mask    = ~mask
        z_embed     = not_mask.cumsum(1, dtype=torch.float32)
        y_embed     = not_mask.cumsum(2, dtype=torch.float32)
        # print(f'y_embed.shape is {y_embed.shape}')
        x_embed     = not_mask.cumsum(3, dtype=torch.float32)
        # embed()
        if self.normalize:
            eps     = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        # print(f'the pos.shape is {pos.shape}') # (batch=3, c=128*3, 4, 4, 4)
        # embed()
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    创建可学习的位置向量
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x       = tensor_list.tensors
        h, w    = x.shape[-2:]
        i       = torch.arange(w, device=x.device)
        j       = torch.arange(h, device=x.device)
        x_emb   = self.col_embed(i)
        y_emb   = self.row_embed(j)
        pos     = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

def build_position_encoding(position_embedding, hidden_dim=256):
    # 创建位置向量
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding

class FrozenBatchNorm2d(torch.nn.Module):
    """
    冻结固定的BatchNorm2d。
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w       = self.weight.reshape(1, -1, 1, 1)
        b       = self.bias.reshape(1, -1, 1, 1)
        rv      = self.running_var.reshape(1, -1, 1, 1)
        rm      = self.running_mean.reshape(1, -1, 1, 1)
        eps     = 1e-5
        scale   = w * (rv + eps).rsqrt()
        bias    = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    """
    用于指定返回哪个层的输出
    这里返回的是最后一层
    """
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers   = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers   = {'layer4': "0"}
            
        # 用于指定返回的层
        self.body           = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels   = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs                           = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m           = tensor_list.mask
            assert m is not None
            mask        = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name]   = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    """
    用于将主干和位置编码模块进行结合
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs                = self[0](tensor_list)
        # xs                      = nested_tensor_from_tensor_list(xs)
        out: List[NestedTensor] = []
        pos                     = []
        for name, x in xs.items():
        # for x in xs:
            # embed()
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

def build_backbone(backbone, position_embedding, hidden_dim, train_backbone=True, pretrained=False):
    # 创建可学习的位置向量还是固定的按'sine'排布的位置向量
    position_embedding  = build_position_encoding(position_embedding, hidden_dim)
    # 创建主干
    # backbone            = Backbone(backbone, train_backbone, False, False, pretrained=pretrained)
    backbone            = Backbone(name='resnet50')
    
    # 用于将主干和位置编码模块进行结合
    model               = Joiner(backbone, position_embedding)
    # model.num_channels = backbone.num_channels
    return model


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    # print(tensor_list[0].shape)
    if tensor_list[0].ndim == 4:
        try:
            if torchvision._is_tracing():
                # nested_tensor_from_tensor_list() does not export well to ONNX
                # call _onnx_nested_tensor_from_tensor_list() instead
                # return _onnx_nested_tensor_from_tensor_list(tensor_list)
                pass
        except:
            pass

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, d, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, d, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)






'''
the detr
'''
#* =========================================================================================================



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DETR(nn.Module):
    def __init__(self, backbone, position_embedding, hidden_dim, num_classes, num_queries, aux_loss=False, pretrained=False):
        super().__init__()
        # 要使用的主干
        self.backbone       = build_backbone(backbone, position_embedding, hidden_dim, pretrained=pretrained)
        self.input_proj     = nn.Conv3d(2048, hidden_dim, kernel_size=1)
        # self.input_proj     = nn.Conv3d(2048, 384, kernel_size=1)
        self.pos            = nn.Conv3d(384, hidden_dim, kernel_size=1) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # 要使用的transformers模块
        self.transformer    = build_transformer(hidden_dim=hidden_dim, pre_norm=False)
        hidden_dim          = self.transformer.d_model
        
        # 输出分类信息
        self.class_embed    = nn.Linear(hidden_dim, num_classes + 1)
        # 输出回归信息
        self.bbox_embed     = MLP(hidden_dim, hidden_dim, 6, 3)
        # 用于传入transformer进行查询的查询向量
        self.query_embed    = nn.Embedding(num_queries, hidden_dim)
        
        # 查询向量的长度与是否使用辅助分支
        self.num_queries    = num_queries
        self.aux_loss       = aux_loss

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # 传入主干网络中进行预测
        # batch_size, 3, 800, 800 => batch_size, 2048, 25, 25
        features, pos = self.backbone(samples)
        # embed()
        # 将网络的结果进行分割，把特征和mask进行分开
        # batch_size, 2048, 25, 25, batch_size, 25, 25
        src, mask = features[-1].decompose()
        # embed()
        mask = mask[:, 0 : (src.shape[2]), 0 : (src.shape[3]), 0 : (src.shape[4])] #! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        assert mask is not None
        # embed()
        # 将主干的结果进行一个映射，然后和查询向量和位置向量传入transformer。
        # batch_size, 2048, 25, 25 => batch_size, 256, 25, 25 => 6, batch_size, 100, 256
        # embed() self.pos(pos[-1]) is to turn torch.Size([3, 384, 4, 4, 4])   to     torch.Size([3, 256, 4, 4, 4])
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, self.pos(pos[-1]))[0]

        # 输出分类信息
        # 6, batch_size, 100, 256 => 6, batch_size, 100, 21
        outputs_class = self.class_embed(hs)
        # 输出回归信息
        # 6, batch_size, 100, 256 => 6, batch_size, 100, 4
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # 只输出transformer最后一层的内容
        # batch_size, 100, 21, batch_size, 100, 4
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, FrozenBatchNorm2d):
                m.eval()




'''
the detr_training
'''
from scipy.optimize import linear_sum_assignment
from functools import partial


def convert_empty_lists_to_tensors(lst):
    for i in range(len(lst)):
        if isinstance(lst[i], list):
            if len(lst[i]) == 0:
                lst[i] = torch.tensor([])  # 将空列表转换为张量
            else:
                lst[i] = convert_empty_lists_to_tensors(lst[i])  # 递归处理嵌套的列表
    return lst


class HungarianMatcher(nn.Module):
    """
    此Matcher计算真实框和网络预测之间的分配
    因为预测多于目标,对最佳预测进行1对1匹配。
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        # 这是Cost中分类错误的相对权重
        self.cost_class = cost_class
        # 这是Cost中边界框坐标L1误差的相对权重
        self.cost_bbox = cost_bbox
        # 这是Cost中边界框giou损失的相对权重
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        # 获得输入的batch_size和query数量
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 将预测结果的batch维度进行平铺
        # [batch_size * num_queries, num_classes]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        # [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  
        # print(f'out_prob is {out_prob}')
        # print(f'out_bbox is {out_bbox}')
        # print(f'out_bbox.shape is {out_bbox.shape}')
        # 将真实框进行concat
        # print(f'the targets is {targets}')
        # for v in targets:
        #     print(f'v is {v}')
        #     print(f'v["labels"] is {v["labels"]}')
        #     tgt_ids = torch.cat(v["labels"]).cuda()
        tgt_ids = torch.cat([v["labels"] for v in targets]).cuda()
        tgt_bbox = torch.cat([v["boxes"] for v in targets]).cuda()
        print(f'tgt_ids is {tgt_ids}')
        print(f'tgt_bbox is {tgt_bbox}')
        # 计算分类成本。预测越准值越小。
        cost_class = -out_prob[:, tgt_ids]
        print(f'cost_class.shape is {cost_class.shape}')
        # 计算预测框和真实框之间的L1成本。预测越准值越小。
        # embed()
        out_bbox = out_bbox.float()  # 将 out_bbox 转换为浮点张量
        tgt_bbox = tgt_bbox.float() 
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        print(f'cost_bbox.shape is {cost_bbox.shape}')
        # 计算预测框和真实框之间的IOU成本。预测越准值越小。
        cost_giou = -generalized_box_iou(box_cxcyczwhd_to_xyzxyz(out_bbox), box_cxcyczwhd_to_xyzxyz(tgt_bbox))
        # 最终的成本矩阵
        print(f'cost_giou.shape is {cost_giou.shape}')
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # 对每一张图片进行指派任务，也就是找到真实框对应的num_queries里面最接近的预测结果，也就是指派num_queries里面一个预测框去预测某一个真实框
        # embed()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # for i, c in enumerate(C.split(sizes, -1)):
        #     embed()
        #     indices = linear_sum_assignment(c[i])
        print(f'after the matcher the indice is {indices}')
        # 返回指派的结果
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]





class SetCriterion(nn.Module):
    """ 
    计算DETR的损失。该过程分为两个步骤:
    1、计算了真实框和模型输出之间的匈牙利分配
    2、根据分配结果计算损失
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        # 类别数量
        self.num_classes    = num_classes
        # 用于匹配的匹配类HungarianMatcher
        self.matcher        = matcher
        # 损失的权值分配
        self.weight_dict    = weight_dict
        # 背景的权重
        self.eos_coef       = eos_coef
        # 需要计算的损失
        self.losses         = losses
        # 种类的权重
        empty_weight        = torch.ones(self.num_classes + 1)
        empty_weight[-1]    = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        # 首先计算不属于辅助头的损失
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # 通过matcher计算每一个图片，预测框和真实框的对应情况
        indices = self.matcher(outputs_without_aux, targets)

        # 计算这个batch中所有图片的总的真实框数量
        # 计算所有节点的目标框的平均数量，以实现标准化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 在辅助损失的情况下，我们对每个中间层的输出重复此过程。
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # 根据名称计算损失
        loss_map = {
            'labels'        : self.loss_labels,
            'cardinality'   : self.loss_cardinality,
            'boxes'         : self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        # 获得输出中的分类部分
        src_logits          = outputs['pred_logits']

        # 找到预测结果中有对应真实框的预测框
        idx                 = self._get_src_permutation_idx(indices)
        # embed()
        
        # embed()
        # 获得整个batch所有框的类别
        # for t, (_, J) in zip(targets, indices):
        #     # embed()
        #     t["labels"] = t["labels"].cuda()
        #     print(f'J is {J}, t["labels"] is {t["labels"]}')
        #     target_classes_o    = torch.cat([t["labels"][J]])


        target_classes_o    = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes      = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        # 将其中对应的预测框设置为目标类别，否则为背景
        # embed()
        indices = [tuple(tensor.cuda() for tensor in tensors) for tensors in indices]
        target_classes_o = target_classes_o.cuda()
        # print(f'target_classes is {target_classes}')
        # print(f'indices is {indices}')
        # print(f'idx is {idx}')
        # print(f'target_classes_o is {target_classes_o}')


        target_classes[idx] = target_classes_o

        # 计算交叉熵
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        # print(f'the label loss')
        # embed()
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits     = outputs['pred_logits']
        device          = pred_logits.device
        
        # 计算每个batch真实框的数量
        tgt_lengths     = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # 计算不是背景的预测数
        card_pred       = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # 然后将不是背景的预测数和真实情况做一个l1损失
        # print(f'pred_logits is {pred_logits}')
        # print(f'targets is {targets}')
        # print(f'card_pred.float() is {card_pred.float()}, the tgt_lengths.float() is {tgt_lengths.float()}')
        card_err        = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses          = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        # 找到预测结果中有对应真实框的预测框
        idx             = self._get_src_permutation_idx(indices)
        # 将预测结果中有对应真实框的预测框取出
        src_boxes       = outputs['pred_boxes'][idx]
        # 取出真实框
        target_boxes    = torch.cat([t['boxes'][i].cuda() for t, (_, i) in zip(targets, indices)], dim=0)
        
        # 预测框和所有的真实框计算l1的损失
        loss_bbox       = F.l1_loss(src_boxes, target_boxes, reduction='none')
        # 计算giou损失
        loss_giou       = 1 - torch.diag(generalized_box_iou(box_cxcyczwhd_to_xyzxyz(src_boxes), box_cxcyczwhd_to_xyzxyz(target_boxes)))
        # 返回两个损失
        losses              = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        # print(f'the bbox loss')
        # embed()
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx   = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx     = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx   = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx     = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

def build_loss(num_classes, dec_layers=6, aux_loss=False):
    # 用到的真实框与预测框的匹配器
    matcher                     = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    # 不同损失的权重
    weight_dict                 = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    # TODO this is a hack
    if aux_loss:
        aux_weight_dict         = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    # 要计算的三个内容
    losses      = ['labels', 'boxes', 'cardinality']
    
    # 构建损失的类
    criterion   = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.1, losses=losses)
    return criterion

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch, lr_scale_ratio):
    lr = lr_scheduler_func(epoch)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * lr_scale_ratio[i]

def data_set():
    nii_data = tio.ScalarImage('/public_bme/data/xiongjl/detr/data/output.nii')
    data = nii_data.data.unsqueeze(0)    #* crop (128, 128, 128) -> resize(256, 256, 256) 
    data[data>1] = 1
    data[data<0] = 0
    targets = [
        {
            "labels": torch.tensor([0, 0]).cuda(),
            "boxes": torch.tensor([[146./256., 222./256., 42./256., 11./256., 11./256., 7./256.], [236./256., 6./256., 209./256., 3.6/256., 3.6/256., 2.2/256.]]).cuda(),
        }
    ]
    #* output.nii
    #* （0.5703， 0.867， 0.164， 0.0429， 0.0429, 0.02734）, 
    # * (0.9218， 0.0234， 0.816， 0.01406， 0.01406, 8.59e-3)
    # * [236./256., 6./256., 209./256., 3.6/256., 3.6/256., 2.2/256.]
    return data, targets

if __name__ == '__main__':

    from detr import DETR
    from IPython import embed
 

    # samples = torch.randn((3, 1, 256, 256, 256)) # (batch, channel, w, h, d)
    nii_data = tio.ScalarImage('.\\image\\nii_data\\output.nii')
    data = nii_data.data.unsqueeze(0)    #* crop (128, 128, 128) -> resize(256, 256, 256)
    # embed()

    net = DETR(backbone='resnet50', position_embedding='sine', hidden_dim=256, num_classes=1, num_queries=100)

    out = net(data)
    #* out['pred_logits'].shape , [batch_size, num_queries, num_classes] , out['pred_boxes'].shape , [batch_size, num_queries, 4]
    # embed()
    # print(f"the out shape is {out['pred_logits'].shape}" )

    detr_loss = build_loss(1)

    targets = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[146., 222., 42., 11., 11., 7.]])
        },
        # {
        #     "labels": torch.tensor([1]),
        #     "boxes": torch.tensor([[236., 6., 209., 3.6, 3.6, 2.2]])
        # },
        # {
        #     "labels": torch.tensor([1]),
        #     "boxes": torch.tensor([[21., 22., 23., 4., 5., 6.]])
        # }
    ]




    loss_value  = detr_loss(out, targets)

    # embed()
