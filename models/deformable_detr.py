# ------------------------------------------------------------------------
# OW-DETR: Open-world Detection Transformer
# Akshita Gupta^, Sanath Narayan^, K J Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah
# https://arxiv.org/pdf/2112.01513.pdf
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Deformable DETR model and criterion classes.
"""
from matplotlib.path import Path
import torch
import torch.nn.functional as F
from torch import nn

import math
import pickle
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)  # , sigmoid_focal_loss_CA)
from .deformable_transformer import build_deforamble_transformer
import copy
import heapq
import operator
import os
from copy import deepcopy
# from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
import torchvision
import torch.nn.functional as F
import torch

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

import torch.nn as nn
import torch


class CustomClassPredictor(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(CustomClassPredictor, self).__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.weight_decoder = nn.Parameter(torch.rand(hidden_dim))
        self.weight_ring = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, decoder_feat, ring_feat):

        combined_feat = torch.cat([decoder_feat, ring_feat], dim=-1)
        combined_feat = torch.relu(self.fc1(combined_feat))
        return self.classifier(combined_feat)

def create_outer_mask(outer_region, inner_coords, outer_coords):

    outer_mask = torch.ones_like(outer_region)

    y1_rel = inner_coords[0] - outer_coords[0]
    y2_rel = inner_coords[1] - outer_coords[0]
    x1_rel = inner_coords[2] - outer_coords[2]
    x2_rel = inner_coords[3] - outer_coords[2]   

    if y1_rel >= 0 and y2_rel <= outer_region.shape[1] and x1_rel >= 0 and x2_rel <= outer_region.shape[2]:
        outer_mask[:, y1_rel:y2_rel, x1_rel:x2_rel] = 0

    return outer_mask
class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 unmatched_boxes=False, novelty_cls=False, featdim=1024):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.class_embed = nn.Linear(hidden_dim, num_classes)  # 类别嵌入
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 边界框嵌入
        self.num_feature_levels = num_feature_levels

        self.featdim = featdim
        self.unmatched_boxes = unmatched_boxes
        self.novelty_cls = novelty_cls
        if self.novelty_cls:
            self.nc_class_embed = nn.Linear(hidden_dim, 1)  # 新颖类别的嵌入

        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),  # 1x1卷积用于通道数变换
                    nn.GroupNorm(32, hidden_dim),  # 归一化层，32为组数
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        if self.novelty_cls:
            self.nc_class_embed.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        self.class_embed = _get_clones(CustomClassPredictor(hidden_dim, num_classes),num_pred)
        self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        self.transformer.decoder.bbox_embed = self.bbox_embed

        if two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def expand_boxes(self, boxes, scale=1.2):
        center = (boxes[..., :2] + boxes[..., 2:]) / 2
        size = boxes[..., 2:] - boxes[..., :2]
        new_size = size * scale
        new_boxes = torch.cat([center - new_size / 2, center + new_size / 2], dim=-1)  # 创建新的盒子坐标
        return new_boxes

    def generate_box_mask(self, boxes, height, width):

        batch_size, num_boxes, _ = boxes.size()
        masks = torch.zeros((batch_size, 1, height, width), dtype=torch.float32, device=boxes.device)
        scale = torch.tensor([width, height, width, height], device=boxes.device).float()
        boxes_scaled = boxes * scale
        boxes_scaled = boxes_scaled.long()
        x1 = torch.clamp(boxes_scaled[:, :, 0] - boxes_scaled[:, :, 2] // 2, 0, width - 1)
        y1 = torch.clamp(boxes_scaled[:, :, 1] - boxes_scaled[:, :, 3] // 2, 0, height - 1)
        x2 = torch.clamp(boxes_scaled[:, :, 0] + boxes_scaled[:, :, 2] // 2, 0, width - 1)
        y2 = torch.clamp(boxes_scaled[:, :, 1] + boxes_scaled[:, :, 3] // 2, 0, height - 1)
        for idx in range(batch_size):
            for jdx in range(num_boxes):
                masks[idx, 0, y1[idx, jdx]:y2[idx, jdx], x1[idx, jdx]:x2[idx, jdx]] = 1
        return masks

    def extract_ring_features(self, outputs_coords, expanded_coords, memory, spatial_shapes, valid_ratios):
        batch_size, num_queries, _ = outputs_coords.shape
        num_layers = spatial_shapes.size(0)
        _, total_features, C = memory.shape
        dtype = memory.dtype
        device = memory.device
        feature_maps = []
        start_index = 0
        for i in range(num_layers):
            H, W = spatial_shapes[i][0], spatial_shapes[i][1]  # 获取每层特征图的高度和宽度
            feature_length = H * W
            feature_map = memory[:, start_index:start_index + feature_length].view(-1, C, H, W)
            feature_maps.append(feature_map)
            start_index += feature_length

        ring_features = torch.zeros((batch_size, num_queries, C), dtype=dtype, device=device)

        xmin_outer = expanded_coords[:, :, 0:1] - expanded_coords[:, :, 2:3] / 2
        xmax_outer = expanded_coords[:, :, 0:1] + expanded_coords[:, :, 2:3] / 2
        ymin_outer = expanded_coords[:, :, 1:2] - expanded_coords[:, :, 3:4] / 2
        ymax_outer = expanded_coords[:, :, 1:2] + expanded_coords[:, :, 3:4] / 2

        xmin_inner = outputs_coords[:, :, 0:1] - outputs_coords[:, :, 2:3] / 2
        xmax_inner = outputs_coords[:, :, 0:1] + outputs_coords[:, :, 2:3] / 2
        ymin_inner = outputs_coords[:, :, 1:2] - outputs_coords[:, :, 3:4] / 2
        ymax_inner = outputs_coords[:, :, 1:2] + outputs_coords[:, :, 3:4] / 2

        for k in range(num_layers):
            feature_map = feature_maps[k]
            H, W = feature_map.shape[2], feature_map.shape[3]

            x1_outer = torch.clamp((xmin_outer * W).long(), 0, W)
            x2_outer = torch.clamp((xmax_outer * W).long(), 0, W)
            y1_outer = torch.clamp((ymin_outer * H).long(), 0, H)
            y2_outer = torch.clamp((ymax_outer * H).long(), 0, H)

            x1_inner = torch.clamp((xmin_inner * W).long(), 0, W)
            x2_inner = torch.clamp((xmax_inner * W).long(), 0, W)
            y1_inner = torch.clamp((ymin_inner * H).long(), 0, H)
            y2_inner = torch.clamp((ymax_inner * H).long(), 0, H)

            for idx in range(batch_size):
                for jdx in range(num_queries):
                    outer_region = feature_map[idx, :, y1_outer[idx, jdx]:y2_outer[idx, jdx],
                                   x1_outer[idx, jdx]:x2_outer[idx, jdx]]
                    outer_mask = create_outer_mask(outer_region,
                                                   (y1_inner[idx, jdx], y2_inner[idx, jdx], x1_inner[idx, jdx],
                                                    x2_inner[idx, jdx]),
                                                   (y1_outer[idx, jdx], y2_outer[idx, jdx], x1_outer[idx, jdx],
                                                    x2_outer[idx, jdx]))
                    ring_region = outer_region * outer_mask
                    pooled_ring_feature = F.adaptive_avg_pool2d(ring_region, (1, 1)).view(-1)
                    ring_features[idx, jdx] += pooled_ring_feature
        return ring_features

    def forward(self, samples: NestedTensor, targets: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        srcs = []
        masks = []

        if self.featdim == 512:
            dim_index = 0
        elif self.featdim == 1024:
            dim_index = 1
        else:
            dim_index = 2
        for l, feat in enumerate(features):
            src, mask = feat.decompose()  # mask与图像预处理相关 统一为256通道数
            resnet_1024_feature = None
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        # TODO 加入encoder预测
        hs, init_reference, inter_references, dn_meta, memory, spatial_shapes, valid_ratios, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs, masks, pos, targets, query_embeds)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = inter_references
            else:
                reference = init_reference[lvl - 1]

            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            expanded_coord = self.expand_boxes(outputs_coord, scale=1.2)
            ring_features = self.extract_ring_feature(outputs_coord, expanded_coord, memory, spatial_shapes,
                                                       valid_ratios)

            output_class = self.class_embed[lvl](hs[lvl], ring_features)

            outputs_classes.append(output_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(outputs_coord, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(outputs_class, dn_meta['dn_num_split'], dim=2)
            outputs_class = out_logits
            outputs_coord = out_bboxes

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'resnet_1024_feat': resnet_1024_feature}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class[:-1], outputs_coord[:-1])
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
            if self.novelty_cls:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, output_class_nc=output_class_nc)
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': [enc_outputs_class], 'pred_boxes': [enc_outputs_coord]}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_class_nc=None):
        xx = [{'pred_logits': a, 'pred_boxes': b}
              for a, b in zip(outputs_class, outputs_coord)]
        return xx


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args, num_classes, matcher, weight_dict, losses, invalid_cls_logits, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        """
           初始化损失计算函数。
           参数:
               num_classes: 对象类别的数量，不包括特殊的无对象类别
               matcher: 能够计算目标和提议之间匹配的模块
               weight_dict: 包含损失名称及其相对权重的字典
               losses: 要应用的所有损失的列表。参见get_loss查看可用的损失列表
               focal_alpha: Focal Loss中的alpha参数
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.nc_epoch = args.nc_epoch
        self.output_dir = args.output_dir
        self.invalid_cls_logits = invalid_cls_logits
        self.unmatched_boxes = args.unmatched_boxes
        self.top_unk = args.top_unk
        self.bbox_thresh = args.bbox_thresh
        self.num_seen_classes = args.PREV_INTRODUCED_CLS + args.CUR_INTRODUCED_CLS
        self.alpha = 0.2
        self.gamma = 2.0

    def loss_NC_labels(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices, log=True):
        """Novelty classification loss
        target labels will contain class as 1
        owod_indices -> indices combining matched indices + psuedo labeled indices
        owod_targets -> targets combining GT targets + psuedo labeled unknown targets
        target_classes_o -> contains all 1's
        """

        assert 'pred_nc_logits' in outputs
        src_logits = outputs['pred_nc_logits']

        idx = self._get_src_permutation_idx(owod_indices)
        target_classes_o = torch.cat(
            [torch.full_like(t["labels"][J], 0) for t, (_, J) in zip(owod_targets, owod_indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]

        losses = {'loss_NC': loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """


        assert 'pred_logits' in outputs
        temp_src_logits = outputs['pred_logits'].clone()
        temp_src_logits[:, :, self.invalid_cls_logits] = -10e10
        src_logits = temp_src_logits

        if self.unmatched_boxes:
            idx = self._get_src_permutation_idx(owod_indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(owod_targets, owod_indices)])
        else:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        temp_pred_logits = outputs['pred_logits'].clone()
        temp_pred_logits[:, :, self.invalid_cls_logits] = -10e10
        pred_logits = temp_pred_logits

        device = pred_logits.device

        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses


    def filter_boxes_inside_conveyor(self, boxes, polygon):
        cx, cy = boxes[:2]
        poly_points = polygon.squeeze(0)
        count = torch.zeros(boxes.shape[0], device=boxes.device)

        n = poly_points.shape[0]
        for i in range(n):
            p1 = poly_points[i]
            p2 = poly_points[(i + 1) % n]

            condition = (p1[1] > cy) != (p2[1] > cy)
            xinters = (cy - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
            condition &= cx < xinters

            count += condition.float()

        inside = (count % 2 == 1)

        return inside

    def loss_boxes(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices):

        assert 'pred_boxes' in outputs

        losses = {}
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        filtered_boxes_list = []
        loss = torch.tensor(0.0, device=src_boxes.device, dtype=src_boxes.dtype, requires_grad=True)

        conveyor_belt_coords = targets[0]['conveyor_points']
        for batch_id, pred_box in zip(idx[0], src_boxes):
            conveyor_belt_coords = targets[batch_id]['conveyor_points']
            inside = self.filter_boxes_inside_conveyor(pred_box, conveyor_belt_coords)
            loss = loss + (1 - inside.float()).sum()

        if num_boxes > 0:
            loss_conveyor = loss / num_boxes
        else:
            loss_conveyor = loss

        losses['loss_conveyor'] = loss_conveyor

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, current_epoch, owod_targets, owod_indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, owod_indices, current_epoch, owod_targets):
        assert 'pred_logits' in outputs

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(
            torch.float32)

        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, owod_indices, current_epoch, owod_targets):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_bce': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, owod_indices, current_epoch, owod_targets):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[...,
                 :-1]
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        pred_score = torch.sigmoid(src_logits).detach()
        weight = 0.1 * pred_score.pow(2.0) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def save_dict(self, di_, filename_):
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)

    def load_dict(self, filename_):
        with open(filename_, 'rb') as f:
            ret_dict = pickle.load(f)
        return ret_dict

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_single_permutation_idx(self, indices, index):
        batch_idx = [torch.full_like(src, i) for i, src in enumerate(indices)][0]
        src_idx = indices[0]
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'NC_labels': self.loss_NC_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'loss_vfl': self.loss_labels_vfl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs)

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device),
                                         torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices

    def forward(self, samples, outputs, targets, epoch):

        if self.nc_epoch > 0:
            loss_epoch = 9
        else:
            loss_epoch = 0
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        indices = self.matcher(outputs_without_aux, targets)

        owod_targets = []
        owod_indices = []

        owod_outputs = outputs_without_aux.copy()
        owod_device = owod_outputs["pred_boxes"].device

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes, epoch, owod_targets, owod_indices, **kwargs))


        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)

                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, epoch, owod_targets,
                                           owod_indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, epoch, owod_targets,
                                           owod_indices, **kwargs)
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# 不加过滤
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = args.num_classes
    print(num_classes)
    if args.dataset == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    prev_intro_cls = args.PREV_INTRODUCED_CLS
    curr_intro_cls = args.CUR_INTRODUCED_CLS
    seen_classes = prev_intro_cls + curr_intro_cls
    invalid_cls_logits = list(
        range(seen_classes, num_classes - 1))  # unknown class indx will not be included in the invalid class range
    print("Invalid class rangw: " + str(invalid_cls_logits))

    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        unmatched_boxes=args.unmatched_boxes,
        novelty_cls=args.NC_branch,
        featdim=args.featdim,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    if args.NC_branch:
        weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_NC': args.nc_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_conveyor'] = 1
    weight_dict['loss_vfl'] = 1
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_dn_': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes', 'cardinality', 'loss_vfl']
    if args.NC_branch:
        losses = ['labels', 'NC_labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]

    criterion = SetCriterion(args, num_classes, matcher, weight_dict, losses, invalid_cls_logits,
                             focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors