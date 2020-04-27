#!/usr/bin/env python
# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Parts are based on https://github.com/multimodallearning/pytorch-mask-rcnn
published under MIT license.
"""
import warnings
warnings.filterwarnings('ignore', '.*From scipy 0.13.0, the output shape of zoom()*')

import numpy as np
import scipy.misc
import scipy.ndimage
import scipy.interpolate
from scipy.ndimage.measurements import label as lb
import torch

import tqdm

from custom_extensions.nms import nms
from custom_extensions.roi_align import roi_align

############################################################
#  Segmentation Processing
############################################################

def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(int(ax))
    return input

def get_one_hot_encoding(y, n_classes):
    """
    transform a numpy label array to a one-hot array of the same shape.
    :param y: array of shape (b, 1, y, x, (z)).
    :param n_classes: int, number of classes to unfold in one-hot encoding.
    :return y_ohe: array of shape (b, n_classes, y, x, (z))
    """

    dim = len(y.shape) - 2
    if dim == 2:
        y_ohe = np.zeros((y.shape[0], n_classes, y.shape[2], y.shape[3])).astype('int32')
    elif dim == 3:
        y_ohe = np.zeros((y.shape[0], n_classes, y.shape[2], y.shape[3], y.shape[4])).astype('int32')
    else:
        raise Exception("invalid dimensions {} encountered".format(y.shape))
    for cl in np.arange(n_classes):
        y_ohe[:, cl][y[:, 0] == cl] = 1
    return y_ohe

def dice_per_batch_inst_and_class(pred, y, n_classes, convert_to_ohe=True, smooth=1e-8):
    '''
    computes dice scores per batch instance and class.
    :param pred: prediction array of shape (b, 1, y, x, (z)) (e.g. softmax prediction with argmax over dim 1)
    :param y: ground truth array of shape (b, 1, y, x, (z)) (contains int [0, ..., n_classes]
    :param n_classes: int
    :return: dice scores of shape (b, c)
    '''
    if convert_to_ohe:
        pred = get_one_hot_encoding(pred, n_classes)
        y = get_one_hot_encoding(y, n_classes)
    axes = tuple(range(2, len(pred.shape)))
    intersect = np.sum(pred*y, axis=axes)
    denominator = np.sum(pred, axis=axes)+np.sum(y, axis=axes)
    dice = (2.0*intersect + smooth) / (denominator + smooth)
    return dice

def dice_per_batch_and_class(pred, targ, n_classes, convert_to_ohe=True, smooth=1e-8):
    '''
    computes dice scores per batch and class.
    :param pred: prediction array of shape (b, 1, y, x, (z)) (e.g. softmax prediction with argmax over dim 1)
    :param targ: ground truth array of shape (b, 1, y, x, (z)) (contains int [0, ..., n_classes])
    :param n_classes: int
    :param smooth: Laplacian smooth, https://en.wikipedia.org/wiki/Additive_smoothing
    :return: dice scores of shape (b, c)
    '''
    if convert_to_ohe:
        pred = get_one_hot_encoding(pred, n_classes)
        targ = get_one_hot_encoding(targ, n_classes)
    axes = (0, *list(range(2, len(pred.shape)))) #(0,2,3(,4))

    intersect = np.sum(pred * targ, axis=axes)

    denominator = np.sum(pred, axis=axes) + np.sum(targ, axis=axes)
    dice = (2.0 * intersect + smooth) / (denominator + smooth)

    assert dice.shape==(n_classes,), "dice shp {}".format(dice.shape)
    return dice


def batch_dice(pred, y, false_positive_weight=1.0, smooth=1e-6):
    '''
    compute soft dice over batch. this is a differentiable score and can be used as a loss function.
    only dice scores of foreground classes are returned, since training typically
    does not benefit from explicit background optimization. Pixels of the entire batch are considered a pseudo-volume to compute dice scores of.
    This way, single patches with missing foreground classes can not produce faulty gradients.
    :param pred: (b, c, y, x, (z)), softmax probabilities (network output).
    :param y: (b, c, y, x, (z)), one hote encoded segmentation mask.
    :param false_positive_weight: float [0,1]. For weighting of imbalanced classes,
    reduces the penalty for false-positive pixels. Can be beneficial sometimes in data with heavy fg/bg imbalances.
    :return: soft dice score (float).This function discards the background score and returns the mena of foreground scores.
    '''

    if len(pred.size()) == 4:
        axes = (0, 2, 3)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean(( (2*intersect + smooth) / (denom + smooth))[1:]) #only fg dice here.

    elif len(pred.size()) == 5:
        axes = (0, 2, 3, 4)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean(( (2*intersect + smooth) / (denom + smooth))[1:]) #only fg dice here.
    else:
        raise ValueError('wrong input dimension in dice loss')


############################################################
#  Bounding Boxes
############################################################

def compute_iou_2D(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2] THIS IS THE GT BOX
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union

    return iou


def compute_iou_3D(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2, z1, z2] (typically gt box)
    boxes: [boxes_count, (y1, x1, y2, x2, z1, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    z1 = np.maximum(box[4], boxes[:, 4])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / union

    return iou



def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)]. / 3D: (z1, z2))
    For better performance, pass the largest set first and the smaller second.
    :return: (#boxes1, #boxes2), ious of each box of 1 machted with each of 2
    """
    # Areas of anchors and GT boxes
    if boxes1.shape[1] == 4:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i] #this is the gt box
            overlaps[:, i] = compute_iou_2D(box2, boxes1, area2[i], area1)
        return overlaps

    else:
        # Areas of anchors and GT boxes
        volume1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 4])
        volume2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 4])
        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(boxes2.shape[0]):
            box2 = boxes2[i]  # this is the gt box
            overlaps[:, i] = compute_iou_3D(box2, boxes1, volume2[i], volume1)
        return overlaps



def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)] / 3D: (z1, z2))
    """
    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)
    result = torch.stack([dy, dx, dh, dw], dim=1)

    if box.shape[1] > 4:
        depth = box[:, 5] - box[:, 4]
        center_z = box[:, 4] + 0.5 * depth
        gt_depth = gt_box[:, 5] - gt_box[:, 4]
        gt_center_z = gt_box[:, 4] + 0.5 * gt_depth
        dz = (gt_center_z - center_z) / depth
        dd = torch.log(gt_depth / depth)
        result = torch.stack([dy, dx, dz, dh, dw, dd], dim=1)

    return result



def unmold_mask_2D(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox
    out_zoom = [y2 - y1, x2 - x1]
    zoom_factor = [i / j for i, j in zip(out_zoom, mask.shape)]

    mask = scipy.ndimage.zoom(mask, zoom_factor, order=1).astype(np.float32)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2]) #only y,x
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


def unmold_mask_2D_torch(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox
    out_zoom = [(y2 - y1).float(), (x2 - x1).float()]
    zoom_factor = [i / j for i, j in zip(out_zoom, mask.shape)]

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torch.nn.functional.interpolate(mask, scale_factor=zoom_factor)
    mask = mask[0][0]
    #mask = scipy.ndimage.zoom(mask.cpu().numpy(), zoom_factor, order=1).astype(np.float32)
    #mask = torch.from_numpy(mask).cuda()
    # Put the mask in the right location.
    full_mask = torch.zeros(image_shape[:2])  # only y,x
    full_mask[y1:y2, x1:x2] = mask
    return full_mask



def unmold_mask_3D(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2, z1, z2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    y1, x1, y2, x2, z1, z2 = bbox
    out_zoom = [y2 - y1, x2 - x1, z2 - z1]
    zoom_factor = [i/j for i,j in zip(out_zoom, mask.shape)]
    mask = scipy.ndimage.zoom(mask, zoom_factor, order=1).astype(np.float32)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:3])
    full_mask[y1:y2, x1:x2, z1:z2] = mask
    return full_mask

def nms_numpy(box_coords, scores, thresh):
    """ non-maximum suppression on 2D or 3D boxes in numpy.
    :param box_coords: [y1,x1,y2,x2 (,z1,z2)] with y1<=y2, x1<=x2, z1<=z2.
    :param scores: ranking scores (higher score == higher rank) of boxes.
    :param thresh: IoU threshold for clustering.
    :return:
    """
    y1 = box_coords[:, 0]
    x1 = box_coords[:, 1]
    y2 = box_coords[:, 2]
    x2 = box_coords[:, 3]
    assert np.all(y1 <= y2) and np.all(x1 <= x2), """"the definition of the coordinates is crucially important here: 
            coordinates of which maxima are taken need to be the lower coordinates"""
    areas = (x2 - x1) * (y2 - y1)

    is_3d = box_coords.shape[1] == 6
    if is_3d: # 3-dim case
        z1 = box_coords[:, 4]
        z2 = box_coords[:, 5]
        assert np.all(z1<=z2), """"the definition of the coordinates is crucially important here: 
           coordinates of which maxima are taken need to be the lower coordinates"""
        areas *= (z2 - z1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:  # order is the sorted index.  maps order to index: order[1] = 24 means (rank1, ix 24)
        i = order[0] # highest scoring element
        yy1 = np.maximum(y1[i], y1[order])  # highest scoring element still in >order<, is compared to itself, that is okay.
        xx1 = np.maximum(x1[i], x1[order])
        yy2 = np.minimum(y2[i], y2[order])
        xx2 = np.minimum(x2[i], x2[order])

        h = np.maximum(0.0, yy2 - yy1)
        w = np.maximum(0.0, xx2 - xx1)
        inter = h * w

        if is_3d:
            zz1 = np.maximum(z1[i], z1[order])
            zz2 = np.minimum(z2[i], z2[order])
            d = np.maximum(0.0, zz2 - zz1)
            inter *= d

        iou = inter / (areas[i] + areas[order] - inter)

        non_matches = np.nonzero(iou <= thresh)[0]  # get all elements that were not matched and discard all others.
        order = order[non_matches]
        keep.append(i)

    return keep



############################################################
#  M-RCNN
############################################################

def refine_proposals(rpn_pred_probs, rpn_pred_deltas, proposal_count, batch_anchors, cf):
    """
    Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement details to anchors.
    :param rpn_pred_probs: (b, n_anchors, 2)
    :param rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
    :return: batch_normalized_props: Proposals in normalized coordinates (b, proposal_count, (y1, x1, y2, x2, (z1), (z2), score))
    :return: batch_out_proposals: Box coords + RPN foreground scores
    for monitoring/plotting (b, proposal_count, (y1, x1, y2, x2, (z1), (z2), score))
    """
    std_dev = torch.from_numpy(cf.rpn_bbox_std_dev[None]).float().cuda()
    norm = torch.from_numpy(cf.scale).float().cuda()
    anchors = batch_anchors.clone()


    batch_scores = rpn_pred_probs[:, :, 1]
    # norm deltas
    batch_deltas = rpn_pred_deltas * std_dev
    batch_normalized_props = []
    batch_out_proposals = []

    # loop over batch dimension.
    for ix in range(batch_scores.shape[0]):

        scores = batch_scores[ix]
        deltas = batch_deltas[ix]

        non_nans = deltas == deltas
        assert torch.all(non_nans), "deltas have nans: {}".format(deltas[~non_nans])

        non_nans = anchors == anchors
        assert torch.all(non_nans), "anchors have nans: {}".format(anchors[~non_nans])

        # improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(cf.pre_nms_limit, anchors.size()[0])
        scores, order = scores.sort(descending=True)
        order = order[:pre_nms_limit]
        scores = scores[:pre_nms_limit]
        deltas = deltas[order, :]

        # apply deltas to anchors to get refined anchors and filter with non-maximum suppression.
        if batch_deltas.shape[-1] == 4:
            boxes = apply_box_deltas_2D(anchors[order, :], deltas)
            non_nans = boxes == boxes
            assert torch.all(non_nans), "unnormalized boxes before clip/after delta apply have nans: {}".format(boxes[~non_nans])
            boxes = clip_boxes_2D(boxes, cf.window)
        else:
            boxes = apply_box_deltas_3D(anchors[order, :], deltas)
            boxes = clip_boxes_3D(boxes, cf.window)

        non_nans = boxes == boxes
        assert torch.all(non_nans), "unnormalized boxes before nms/after clip have nans: {}".format(boxes[~non_nans])
        # boxes are y1,x1,y2,x2, torchvision-nms requires x1,y1,x2,y2, but consistent swap x<->y is irrelevant.
        keep = nms.nms(boxes, scores, cf.rpn_nms_threshold)


        keep = keep[:proposal_count]
        boxes = boxes[keep, :]
        rpn_scores = scores[keep][:, None]

        # pad missing boxes with 0.
        if boxes.shape[0] < proposal_count:
            n_pad_boxes = proposal_count - boxes.shape[0]
            zeros = torch.zeros([n_pad_boxes, boxes.shape[1]]).cuda()
            boxes = torch.cat([boxes, zeros], dim=0)
            zeros = torch.zeros([n_pad_boxes, rpn_scores.shape[1]]).cuda()
            rpn_scores = torch.cat([rpn_scores, zeros], dim=0)

        # concat box and score info for monitoring/plotting.
        batch_out_proposals.append(torch.cat((boxes, rpn_scores), 1).cpu().data.numpy())
        # normalize dimensions to range of 0 to 1.
        non_nans = boxes == boxes
        assert torch.all(non_nans), "unnormalized boxes after nms have nans: {}".format(boxes[~non_nans])
        normalized_boxes = boxes / norm
        where = normalized_boxes <=1
        assert torch.all(where), "normalized box coords >1 found:\n {}\n".format(normalized_boxes[~where])

        # add again batch dimension
        batch_normalized_props.append(torch.cat((normalized_boxes, rpn_scores), 1).unsqueeze(0))

    batch_normalized_props = torch.cat(batch_normalized_props)
    batch_out_proposals = np.array(batch_out_proposals)

    return batch_normalized_props, batch_out_proposals

def pyramid_roi_align(feature_maps, rois, pool_size, pyramid_levels, dim):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.
    :param feature_maps: list of feature maps, each of shape (b, c, y, x , (z))
    :param rois: proposals (normalized coords.) as returned by RPN. contain info about original batch element allocation.
    (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ixs)
    :param pool_size: list of poolsizes in dims: [x, y, (z)]
    :param pyramid_levels: list. [0, 1, 2, ...]
    :return: pooled: pooled feature map rois (n_proposals, c, poolsize_y, poolsize_x, (poolsize_z))

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    boxes = rois[:, :dim*2]
    batch_ixs = rois[:, dim*2]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    if dim == 2:
        y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    else:
        y1, x1, y2, x2, z1, z2 = boxes.chunk(6, dim=1)

    h = y2 - y1
    w = x2 - x1

    # Equation 1 in https://arxiv.org/abs/1612.03144. Account for
    # the fact that our coordinates are normalized here.
    # divide sqrt(h*w) by 1 instead image_area.
    roi_level = (4 + torch.log2(torch.sqrt(h*w))).round().int().clamp(pyramid_levels[0], pyramid_levels[-1])
    # if Pyramid contains additional level P6, adapt the roi_level assignment accordingly.
    if len(pyramid_levels) == 5:
        roi_level[h*w > 0.65] = 5

    # Loop through levels and apply ROI pooling to each.
    pooled = []
    box_to_level = []
    fmap_shapes = [f.shape for f in feature_maps]
    for level_ix, level in enumerate(pyramid_levels):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix, :]
        # re-assign rois to feature map of original batch element.
        ind = batch_ixs[ix].int()

        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()
        if len(pool_size) == 2:
            # remap to feature map coordinate system
            y_exp, x_exp = fmap_shapes[level_ix][2:]  # exp = expansion
            level_boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp], dtype=torch.float32).cuda())
            pooled_features = roi_align.roi_align_2d(feature_maps[level_ix],
                                                     torch.cat((ind.unsqueeze(1).float(), level_boxes), dim=1),
                                                     pool_size)
        else:
            y_exp, x_exp, z_exp = fmap_shapes[level_ix][2:]
            level_boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp, z_exp, z_exp], dtype=torch.float32).cuda())
            pooled_features = roi_align.roi_align_3d(feature_maps[level_ix],
                                                     torch.cat((ind.unsqueeze(1).float(), level_boxes), dim=1),
                                                     pool_size)
        pooled.append(pooled_features)


    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled


def roi_align_3d_numpy(input: np.ndarray, rois, output_size: tuple,
                       spatial_scale: float = 1., sampling_ratio: int = -1) -> np.ndarray:
    """ This fct mainly serves as a verification method for 3D CUDA implementation of RoIAlign, it's highly
        inefficient due to the nested loops.
    :param input:  (ndarray[N, C, H, W, D]): input feature map
    :param rois: list (N,K(n), 6), K(n) = nr of rois in batch-element n, single roi of format (y1,x1,y2,x2,z1,z2)
    :param output_size:
    :param spatial_scale:
    :param sampling_ratio:
    :return: (List[N, K(n), C, output_size[0], output_size[1], output_size[2]])
    """

    out_height, out_width, out_depth = output_size

    coord_grid = tuple([np.linspace(0, input.shape[dim] - 1, num=input.shape[dim]) for dim in range(2, 5)])
    pooled_rois = [[]] * len(rois)
    assert len(rois) == input.shape[0], "batch dim mismatch, rois: {}, input: {}".format(len(rois), input.shape[0])
    print("Numpy 3D RoIAlign progress:", end="\n")
    for b in range(input.shape[0]):
        for roi in tqdm.tqdm(rois[b]):
            y1, x1, y2, x2, z1, z2 = np.array(roi) * spatial_scale
            roi_height = max(float(y2 - y1), 1.)
            roi_width = max(float(x2 - x1), 1.)
            roi_depth = max(float(z2 - z1), 1.)

            if sampling_ratio <= 0:
                sampling_ratio_h = int(np.ceil(roi_height / out_height))
                sampling_ratio_w = int(np.ceil(roi_width / out_width))
                sampling_ratio_d = int(np.ceil(roi_depth / out_depth))
            else:
                sampling_ratio_h = sampling_ratio_w = sampling_ratio_d = sampling_ratio  # == n points per bin

            bin_height = roi_height / out_height
            bin_width = roi_width / out_width
            bin_depth = roi_depth / out_depth

            n_points = sampling_ratio_h * sampling_ratio_w * sampling_ratio_d
            pooled_roi = np.empty((input.shape[1], out_height, out_width, out_depth), dtype="float32")
            for chan in range(input.shape[1]):
                lin_interpolator = scipy.interpolate.RegularGridInterpolator(coord_grid, input[b, chan],
                                                                             method="linear")
                for bin_iy in range(out_height):
                    for bin_ix in range(out_width):
                        for bin_iz in range(out_depth):

                            bin_val = 0.
                            for i in range(sampling_ratio_h):
                                for j in range(sampling_ratio_w):
                                    for k in range(sampling_ratio_d):
                                        loc_ijk = [
                                            y1 + bin_iy * bin_height + (i + 0.5) * (bin_height / sampling_ratio_h),
                                            x1 + bin_ix * bin_width +  (j + 0.5) * (bin_width / sampling_ratio_w),
                                            z1 + bin_iz * bin_depth +  (k + 0.5) * (bin_depth / sampling_ratio_d)]
                                        # print("loc_ijk", loc_ijk)
                                        if not (np.any([c < -1.0 for c in loc_ijk]) or loc_ijk[0] > input.shape[2] or
                                                loc_ijk[1] > input.shape[3] or loc_ijk[2] > input.shape[4]):
                                            for catch_case in range(3):
                                                # catch on-border cases
                                                if int(loc_ijk[catch_case]) == input.shape[catch_case + 2] - 1:
                                                    loc_ijk[catch_case] = input.shape[catch_case + 2] - 1
                                            bin_val += lin_interpolator(loc_ijk)
                            pooled_roi[chan, bin_iy, bin_ix, bin_iz] = bin_val / n_points

            pooled_rois[b].append(pooled_roi)

    return np.array(pooled_rois)

def refine_detections(cf, batch_ixs, rois, deltas, scores, regressions):
    """
    Refine classified proposals (apply deltas to rpn rois), filter overlaps (nms) and return final detections.

    :param rois: (n_proposals, 2 * dim) normalized boxes as proposed by RPN. n_proposals = batch_size * POST_NMS_ROIS
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by mrcnn bbox regressor.
    :param batch_ixs: (n_proposals) batch element assignment info for re-allocation.
    :param scores: (n_proposals, n_classes) probabilities for all classes per roi as predicted by mrcnn classifier.
    :param regressions: (n_proposals, n_classes, regression_features (+1 for uncertainty if predicted) regression vector
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score, *regression vector features))
    """
    # class IDs per ROI. Since scores of all classes are of interest (not just max class), all are kept at this point.
    class_ids = []
    fg_classes = cf.head_classes - 1
    # repeat vectors to fill in predictions for all foreground classes.
    for ii in range(1, fg_classes + 1):
        class_ids += [ii] * rois.shape[0]
    class_ids = torch.from_numpy(np.array(class_ids)).cuda()

    batch_ixs = batch_ixs.repeat(fg_classes)
    rois = rois.repeat(fg_classes, 1)
    deltas = deltas.repeat(fg_classes, 1, 1)
    scores = scores.repeat(fg_classes, 1)
    regressions = regressions.repeat(fg_classes, 1, 1)

    # get class-specific scores and  bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long().cuda()
    # using idx instead of slice [:,] squashes first dimension.
    #len(class_ids)>scores.shape[1] --> probs is broadcasted by expansion from fg_classes-->len(class_ids)
    batch_ixs = batch_ixs[idx]
    deltas_specific = deltas[idx, class_ids]
    class_scores = scores[idx, class_ids]
    regressions = regressions[idx, class_ids]

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(cf.rpn_bbox_std_dev, [1, cf.dim * 2])).float().cuda()
    scale = torch.from_numpy(cf.scale).float().cuda()
    refined_rois = apply_box_deltas_2D(rois, deltas_specific * std_dev) * scale if cf.dim == 2 else \
        apply_box_deltas_3D(rois, deltas_specific * std_dev) * scale

    # round and cast to int since we're dealing with pixels now
    refined_rois = clip_to_window(cf.window, refined_rois)
    refined_rois = torch.round(refined_rois)

    # filter out low confidence boxes
    keep = idx
    keep_bool = (class_scores >= cf.model_min_confidence)
    if not 0 in torch.nonzero(keep_bool).size():

        score_keep = torch.nonzero(keep_bool)[:, 0]
        pre_nms_class_ids = class_ids[score_keep]
        pre_nms_rois = refined_rois[score_keep]
        pre_nms_scores = class_scores[score_keep]
        pre_nms_batch_ixs = batch_ixs[score_keep]

        for j, b in enumerate(unique1d(pre_nms_batch_ixs)):

            bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
            bix_class_ids = pre_nms_class_ids[bixs]
            bix_rois = pre_nms_rois[bixs]
            bix_scores = pre_nms_scores[bixs]

            for i, class_id in enumerate(unique1d(bix_class_ids)):

                ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
                # nms expects boxes sorted by score.
                ix_rois = bix_rois[ixs]
                ix_scores = bix_scores[ixs]
                ix_scores, order = ix_scores.sort(descending=True)
                ix_rois = ix_rois[order, :]

                class_keep = nms.nms(ix_rois, ix_scores, cf.detection_nms_threshold)

                # map indices back.
                class_keep = keep[score_keep[bixs[ixs[order[class_keep]]]]]
                # merge indices over classes for current batch element
                b_keep = class_keep if i == 0 else unique1d(torch.cat((b_keep, class_keep)))

            # only keep top-k boxes of current batch-element
            top_ids = class_scores[b_keep].sort(descending=True)[1][:cf.model_max_instances_per_batch_element]
            b_keep = b_keep[top_ids]

            # merge indices over batch elements.
            batch_keep = b_keep  if j == 0 else unique1d(torch.cat((batch_keep, b_keep)))

        keep = batch_keep

    else:
        keep = torch.tensor([0]).long().cuda()

    # arrange output
    output = [refined_rois[keep], batch_ixs[keep].unsqueeze(1)]
    output += [class_ids[keep].unsqueeze(1).float(), class_scores[keep].unsqueeze(1)]
    output += [regressions[keep]]

    result = torch.cat(output, dim=1)
    # shape: (n_keeps, catted feats), catted feats: [0:dim*2] are box_coords, [dim*2] are batch_ics,
    # [dim*2+1] are class_ids, [dim*2+2] are scores, [dim*2+3:] are regression vector features (incl uncertainty)
    return result


def loss_example_mining(cf, batch_proposals, batch_gt_boxes, batch_gt_masks, batch_roi_scores,
                           batch_gt_class_ids, batch_gt_regressions):
    """
    Subsamples proposals for mrcnn losses and generates targets. Sampling is done per batch element, seems to have positive
    effects on training, as opposed to sampling over entire batch. Negatives are sampled via stochastic hard-example mining
    (SHEM), where a number of negative proposals is drawn from larger pool of highest scoring proposals for stochasticity.
    Scoring is obtained here as the max over all foreground probabilities as returned by mrcnn_classifier (worked better than
    loss-based class-balancing methods like "online hard-example mining" or "focal loss".)

    Classification-regression duality: regressions can be given along with classes (at least fg/bg, only class scores
    are used for ranking).

    :param batch_proposals: (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ixs).
    boxes as proposed by RPN. n_proposals here is determined by batch_size * POST_NMS_ROIS.
    :param mrcnn_class_logits: (n_proposals, n_classes)
    :param batch_gt_boxes: list over batch elements. Each element is a list over the corresponding roi target coordinates.
    :param batch_gt_masks: list over batch elements. Each element is binary mask of shape (n_gt_rois, c, y, x, (z))
    :param batch_gt_class_ids: list over batch elements. Each element is a list over the corresponding roi target labels.
        if no classes predicted (only fg/bg from RPN): expected as pseudo classes [0, 1] for bg, fg.
    :param batch_gt_regressions: list over b elements. Each element is a regression target vector. if None--> pseudo
    :return: sample_indices: (n_sampled_rois) indices of sampled proposals to be used for loss functions.
    :return: target_class_ids: (n_sampled_rois)containing target class labels of sampled proposals.
    :return: target_deltas: (n_sampled_rois, 2 * dim) containing target deltas of sampled proposals for box refinement.
    :return: target_masks: (n_sampled_rois, y, x, (z)) containing target masks of sampled proposals.
    """
    # normalization of target coordinates
    #global sample_regressions
    if cf.dim == 2:
        h, w = cf.patch_size
        scale = torch.from_numpy(np.array([h, w, h, w])).float().cuda()
    else:
        h, w, z = cf.patch_size
        scale = torch.from_numpy(np.array([h, w, h, w, z, z])).float().cuda()

    positive_count = 0
    negative_count = 0
    sample_positive_indices = []
    sample_negative_indices = []
    sample_deltas = []
    sample_masks = []
    sample_class_ids = []
    if batch_gt_regressions is not None:
        sample_regressions = []
    else:
        target_regressions = torch.FloatTensor().cuda()

    std_dev = torch.from_numpy(cf.bbox_std_dev).float().cuda()

    # loop over batch and get positive and negative sample rois.
    for b in range(len(batch_gt_boxes)):

        gt_masks = torch.from_numpy(batch_gt_masks[b]).float().cuda()
        gt_class_ids = torch.from_numpy(batch_gt_class_ids[b]).int().cuda()
        if batch_gt_regressions is not None:
            gt_regressions = torch.from_numpy(batch_gt_regressions[b]).float().cuda()

        #if np.any(batch_gt_class_ids[b] > 0):  # skip roi selection for no gt images.
        if np.any([len(coords)>0 for coords in batch_gt_boxes[b]]):
            gt_boxes = torch.from_numpy(batch_gt_boxes[b]).float().cuda() / scale
        else:
            gt_boxes = torch.FloatTensor().cuda()

        # get proposals and indices of current batch element.
        proposals = batch_proposals[batch_proposals[:, -1] == b][:, :-1]
        batch_element_indices = torch.nonzero(batch_proposals[:, -1] == b).squeeze(1)

        # Compute overlaps matrix [proposals, gt_boxes]
        if not 0 in gt_boxes.size():
            if gt_boxes.shape[1] == 4:
                assert cf.dim == 2, "gt_boxes shape {} doesnt match cf.dim{}".format(gt_boxes.shape, cf.dim)
                overlaps = bbox_overlaps_2D(proposals, gt_boxes)
            else:
                assert cf.dim == 3, "gt_boxes shape {} doesnt match cf.dim{}".format(gt_boxes.shape, cf.dim)
                overlaps = bbox_overlaps_3D(proposals, gt_boxes)

            # Determine positive and negative ROIs
            roi_iou_max = torch.max(overlaps, dim=1)[0]
            # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
            positive_roi_bool = roi_iou_max >= (0.5 if cf.dim == 2 else 0.3)
            # 2. Negative ROIs are those with < 0.1 with every GT box.
            negative_roi_bool = roi_iou_max < (0.1 if cf.dim == 2 else 0.01)
        else:
            positive_roi_bool = torch.FloatTensor().cuda()
            negative_roi_bool = torch.from_numpy(np.array([1]*proposals.shape[0])).cuda()

        # Sample Positive ROIs
        if not 0 in torch.nonzero(positive_roi_bool).size():
            positive_indices = torch.nonzero(positive_roi_bool).squeeze(1)
            positive_samples = int(cf.train_rois_per_image * cf.roi_positive_ratio)
            rand_idx = torch.randperm(positive_indices.size()[0])
            rand_idx = rand_idx[:positive_samples].cuda()
            positive_indices = positive_indices[rand_idx]
            positive_samples = positive_indices.size()[0]
            positive_rois = proposals[positive_indices, :]
            # Assign positive ROIs to GT boxes.
            positive_overlaps = overlaps[positive_indices, :]
            roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
            roi_gt_boxes = gt_boxes[roi_gt_box_assignment, :]
            roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment]
            if batch_gt_regressions is not None:
                roi_gt_regressions = gt_regressions[roi_gt_box_assignment]

            # Compute bbox refinement targets for positive ROIs
            deltas = box_refinement(positive_rois, roi_gt_boxes)
            deltas /= std_dev

            roi_masks = gt_masks[roi_gt_box_assignment]
            assert roi_masks.shape[1] == 1, "gt masks have more than one channel --> is this desired?"
            # Compute mask targets
            boxes = positive_rois
            box_ids = torch.arange(roi_masks.shape[0]).cuda().unsqueeze(1).float()

            if len(cf.mask_shape) == 2:
                y_exp, x_exp = roi_masks.shape[2:]  # exp = expansion
                boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp], dtype=torch.float32).cuda())
                masks = roi_align.roi_align_2d(roi_masks,
                                               torch.cat((box_ids, boxes), dim=1),
                                               cf.mask_shape)
            else:
                y_exp, x_exp, z_exp = roi_masks.shape[2:]  # exp = expansion
                boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp, z_exp, z_exp], dtype=torch.float32).cuda())
                masks = roi_align.roi_align_3d(roi_masks,
                                               torch.cat((box_ids, boxes), dim=1),
                                               cf.mask_shape)

            masks = masks.squeeze(1)
            # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
            # binary cross entropy loss.
            masks = torch.round(masks)

            sample_positive_indices.append(batch_element_indices[positive_indices])
            sample_deltas.append(deltas)
            sample_masks.append(masks)
            sample_class_ids.append(roi_gt_class_ids)
            if batch_gt_regressions is not None:
                sample_regressions.append(roi_gt_regressions)
            positive_count += positive_samples
        else:
            positive_samples = 0

        # Sample negative ROIs. Add enough to maintain positive:negative ratio, but at least 1. Sample via SHEM.
        if not 0 in torch.nonzero(negative_roi_bool).size():
            negative_indices = torch.nonzero(negative_roi_bool).squeeze(1)
            r = 1.0 / cf.roi_positive_ratio
            b_neg_count = np.max((int(r * positive_samples - positive_samples), 1))
            roi_scores_neg = batch_roi_scores[batch_element_indices[negative_indices]]
            raw_sampled_indices = shem(roi_scores_neg, b_neg_count, cf.shem_poolsize)
            sample_negative_indices.append(batch_element_indices[negative_indices[raw_sampled_indices]])
            negative_count  += raw_sampled_indices.size()[0]

    if len(sample_positive_indices) > 0:
        target_deltas = torch.cat(sample_deltas)
        target_masks = torch.cat(sample_masks)
        target_class_ids = torch.cat(sample_class_ids)
        if batch_gt_regressions is not None:
            target_regressions = torch.cat(sample_regressions)

    # Pad target information with zeros for negative ROIs.
    if positive_count > 0 and negative_count > 0:
        sample_indices = torch.cat((torch.cat(sample_positive_indices), torch.cat(sample_negative_indices)), dim=0)
        zeros = torch.zeros(negative_count, cf.dim * 2).cuda()
        target_deltas = torch.cat([target_deltas, zeros], dim=0)
        zeros = torch.zeros(negative_count, *cf.mask_shape).cuda()
        target_masks = torch.cat([target_masks, zeros], dim=0)
        zeros = torch.zeros(negative_count).int().cuda()
        target_class_ids = torch.cat([target_class_ids, zeros], dim=0)
        if batch_gt_regressions is not None:
            # regression targets need to have 0 as background/negative with below practice
            if 'regression_bin' in cf.prediction_tasks:
                zeros = torch.zeros(negative_count, dtype=torch.float).cuda()
            else:
                zeros = torch.zeros(negative_count, cf.regression_n_features, dtype=torch.float).cuda()
            target_regressions = torch.cat([target_regressions, zeros], dim=0)

    elif positive_count > 0:
        sample_indices = torch.cat(sample_positive_indices)
    elif negative_count > 0:
        sample_indices = torch.cat(sample_negative_indices)
        target_deltas = torch.zeros(negative_count, cf.dim * 2).cuda()
        target_masks = torch.zeros(negative_count, *cf.mask_shape).cuda()
        target_class_ids = torch.zeros(negative_count).int().cuda()
        if batch_gt_regressions is not None:
            if 'regression_bin' in cf.prediction_tasks:
                target_regressions = torch.zeros(negative_count, dtype=torch.float).cuda()
            else:
                target_regressions = torch.zeros(negative_count, cf.regression_n_features, dtype=torch.float).cuda()
    else:
        sample_indices = torch.LongTensor().cuda()
        target_class_ids = torch.IntTensor().cuda()
        target_deltas = torch.FloatTensor().cuda()
        target_masks = torch.FloatTensor().cuda()
        target_regressions = torch.FloatTensor().cuda()

    return sample_indices, target_deltas, target_masks, target_class_ids, target_regressions

############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
    return boxes



def generate_anchors_3D(scales_xy, scales_z, ratios, shape, feature_stride_xy, feature_stride_z, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios

    scales_xy, ratios_meshed = np.meshgrid(np.array(scales_xy), np.array(ratios))
    scales_xy = scales_xy.flatten()
    ratios_meshed = ratios_meshed.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales_xy / np.sqrt(ratios_meshed)
    widths = scales_xy * np.sqrt(ratios_meshed)
    depths = np.tile(np.array(scales_z), len(ratios_meshed)//np.array(scales_z)[..., None].shape[0])

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride_xy #translate from fm positions to input coords.
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride_xy
    shifts_z = np.arange(0, shape[2], anchor_stride) * (feature_stride_z)
    shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)

    # Reshape to get a list of (y, x, z) and a list of (h, w, d)
    box_centers = np.stack(
        [box_centers_y, box_centers_x, box_centers_z], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_heights, box_widths, box_depths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (y1, x1, y2, x2, z1, z2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    boxes = np.transpose(np.array([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 2], boxes[:, 5]]), axes=(1, 0))
    return boxes


def generate_pyramid_anchors(logger, cf):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    from configs:
    :param scales: cf.RPN_ANCHOR_SCALES , for conformity with retina nets: scale entries need to be list, e.g. [[4], [8], [16], [32]]
    :param ratios: cf.RPN_ANCHOR_RATIOS , e.g. [0.5, 1, 2]
    :param feature_shapes: cf.BACKBONE_SHAPES , e.g.  [array of shapes per feature map] [80, 40, 20, 10, 5]
    :param feature_strides: cf.BACKBONE_STRIDES , e.g. [2, 4, 8, 16, 32, 64]
    :param anchors_stride: cf.RPN_ANCHOR_STRIDE , e.g. 1
    :return anchors: (N, (y1, x1, y2, x2, (z1), (z2)). All generated anchors in one array. Sorted
    with the same order of the given scales. So, anchors of scale[0] come first, then anchors of scale[1], and so on.
    """
    scales = cf.rpn_anchor_scales
    ratios = cf.rpn_anchor_ratios
    feature_shapes = cf.backbone_shapes
    anchor_stride = cf.rpn_anchor_stride
    pyramid_levels = cf.pyramid_levels
    feature_strides = cf.backbone_strides

    logger.info("anchor scales {} and feature map shapes {}".format(scales, feature_shapes))
    expected_anchors = [np.prod(feature_shapes[level]) * len(ratios) * len(scales['xy'][level]) for level in pyramid_levels]

    anchors = []
    for lix, level in enumerate(pyramid_levels):
        if len(feature_shapes[level]) == 2:
            anchors.append(generate_anchors(scales['xy'][level], ratios, feature_shapes[level],
                                            feature_strides['xy'][level], anchor_stride))
        elif len(feature_shapes[level]) == 3:
            anchors.append(generate_anchors_3D(scales['xy'][level], scales['z'][level], ratios, feature_shapes[level],
                                            feature_strides['xy'][level], feature_strides['z'][level], anchor_stride))
        else:
            raise Exception("invalid feature_shapes[{}] size {}".format(level, feature_shapes[level]))
        logger.info("level {}: expected anchors {}, built anchors {}.".format(level, expected_anchors[lix], anchors[-1].shape))

    out_anchors = np.concatenate(anchors, axis=0)
    logger.info("Total: expected anchors {}, built anchors {}.".format(np.sum(expected_anchors), out_anchors.shape))

    return out_anchors



def apply_box_deltas_2D(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    non_nans = boxes == boxes
    assert torch.all(non_nans), "boxes at beginning of delta apply have nans: {}".format(
        boxes[~non_nans])

    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width

    # clip delta preds in order to avoid infs and later nans after exponentiation.
    height *= torch.exp(torch.clamp(deltas[:, 2], max=6.))
    width *= torch.exp(torch.clamp(deltas[:, 3], max=6.))


    non_nans = width == width
    assert torch.all(non_nans), "inside delta apply, width has nans: {}".format(
        width[~non_nans])

    # 0.*inf results in nan. fix nans to zeros?
    # height[height!=height] = 0.
    # width[width!=width] = 0.


    non_nans = height == height
    assert torch.all(non_nans), "inside delta apply, height has nans directly after setting to zero: {}".format(
        height[~non_nans])

    non_nans = width == width
    assert torch.all(non_nans), "inside delta apply, width has nans directly after setting to zero: {}".format(
        width[~non_nans])

    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)

    non_nans = result == result
    assert torch.all(non_nans), "inside delta apply, result has nans: {}".format(result[~non_nans])

    return result



def apply_box_deltas_3D(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 6] where each row is y1, x1, y2, x2, z1, z2
    deltas: [N, 6] where each row is [dy, dx, dz, log(dh), log(dw), log(dd)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    depth = boxes[:, 5] - boxes[:, 4]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_z = boxes[:, 4] + 0.5 * depth
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    center_z += deltas[:, 2] * depth
    height *= torch.exp(deltas[:, 3])
    width *= torch.exp(deltas[:, 4])
    depth *= torch.exp(deltas[:, 5])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z1 = center_z - 0.5 * depth
    y2 = y1 + height
    x2 = x1 + width
    z2 = z1 + depth
    result = torch.stack([y1, x1, y2, x2, z1, z2], dim=1)
    return result



def clip_boxes_2D(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def clip_boxes_3D(boxes, window):
    """
    boxes: [N, 6] each col is y1, x1, y2, x2, z1, z2
    window: [6] in the form y1, x1, y2, x2, z1, z2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3])),
         boxes[:, 4].clamp(float(window[4]), float(window[5])),
         boxes[:, 5].clamp(float(window[4]), float(window[5]))], 1)
    return boxes

from matplotlib import pyplot as plt


def clip_boxes_numpy(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2 / [N, 6] in 3D.
    window: iamge shape (y, x, (z))
    """
    if boxes.shape[1] == 4:
        boxes = np.concatenate(
            (np.clip(boxes[:, 0], 0, window[0])[:, None],
            np.clip(boxes[:, 1], 0, window[0])[:, None],
            np.clip(boxes[:, 2], 0, window[1])[:, None],
            np.clip(boxes[:, 3], 0, window[1])[:, None]), 1
        )

    else:
        boxes = np.concatenate(
            (np.clip(boxes[:, 0], 0, window[0])[:, None],
             np.clip(boxes[:, 1], 0, window[0])[:, None],
             np.clip(boxes[:, 2], 0, window[1])[:, None],
             np.clip(boxes[:, 3], 0, window[1])[:, None],
             np.clip(boxes[:, 4], 0, window[2])[:, None],
             np.clip(boxes[:, 5], 0, window[2])[:, None]), 1
        )

    return boxes



def bbox_overlaps_2D(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.

    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]

    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    #--> expects x1<x2 & y1<y2
    zeros = torch.zeros(y1.size()[0], requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    assert torch.all(iou<=1), "iou score>1 produced in bbox_overlaps_2D"
    overlaps = iou.view(boxes2_repeat, boxes1_repeat) #--> per gt box: ious of all proposal boxes with that gt box

    return overlaps

def bbox_overlaps_3D(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2, z1, z2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,6)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2, b1_z1, b1_z2 = boxes1.chunk(6, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2, b2_z1, b2_z2 = boxes2.chunk(6, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    z1 = torch.max(b1_z1, b2_z1)[:, 0]
    z2 = torch.min(b1_z2, b2_z2)[:, 0]
    zeros = torch.zeros(y1.size()[0], requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros) * torch.max(z2 - z1, zeros)

    # 3. Compute unions
    b1_volume = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)  * (b1_z2 - b1_z1)
    b2_volume = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)  * (b2_z2 - b2_z1)
    union = b1_volume[:,0] + b2_volume[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)
    return overlaps

def gt_anchor_matching(cf, anchors, gt_boxes, gt_class_ids=None):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2, (z1), (z2))]
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2, (z1), (z2))]
    gt_class_ids (optional): [num_gt_boxes] Integer class IDs for one stage detectors. in RPN case of Mask R-CNN,
    set all positive matches to 1 (foreground)

    Returns:
    anchor_class_matches: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    anchor_delta_targets: [N, (dy, dx, (dz), log(dh), log(dw), (log(dd)))] Anchor bbox deltas.
    """

    anchor_class_matches = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_delta_targets = np.zeros((cf.rpn_train_anchors_per_image, 2*cf.dim))
    anchor_matching_iou = cf.anchor_matching_iou

    if gt_boxes is None:
        anchor_class_matches = np.full(anchor_class_matches.shape, fill_value=-1)
        return anchor_class_matches, anchor_delta_targets

    # for mrcnn: anchor matching is done for RPN loss, so positive labels are all 1 (foreground)
    if gt_class_ids is None:
        gt_class_ids = np.array([1] * len(gt_boxes))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= anchor_matching_iou then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.1 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.1).

    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    if anchors.shape[1] == 4:
        anchor_class_matches[(anchor_iou_max < 0.1)] = -1
    elif anchors.shape[1] == 6:
        anchor_class_matches[(anchor_iou_max < 0.01)] = -1
    else:
        raise ValueError('anchor shape wrong {}'.format(anchors.shape))

    # 2. Set an anchor for each GT box (regardless of IoU value).
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    for ix, ii in enumerate(gt_iou_argmax):
        anchor_class_matches[ii] = gt_class_ids[ix]

    # 3. Set anchors with high overlap as positive.
    above_thresh_ixs = np.argwhere(anchor_iou_max >= anchor_matching_iou)
    anchor_class_matches[above_thresh_ixs] = gt_class_ids[anchor_iou_argmax[above_thresh_ixs]]

    # Subsample to balance positive anchors.
    ids = np.where(anchor_class_matches > 0)[0]
    extra = len(ids) - (cf.rpn_train_anchors_per_image // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        anchor_class_matches[ids] = 0

    # Leave all negative proposals negative for now and sample from them later in online hard example mining.
    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    ids = np.where(anchor_class_matches > 0)[0]
    ix = 0  # index into anchor_delta_targets
    for i, a in zip(ids, anchors[ids]):
        # closest gt box (it might have IoU < anchor_matching_iou)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # convert coordinates to center plus width/height.
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        if cf.dim == 2:
            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]

        else:
            gt_d = gt[5] - gt[4]
            gt_center_z = gt[4] + 0.5 * gt_d
            a_d = a[5] - a[4]
            a_center_z = a[4] + 0.5 * a_d

            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                (gt_center_z - a_center_z) / a_d,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
                np.log(gt_d / a_d)
            ]

        # normalize.
        anchor_delta_targets[ix] /= cf.rpn_bbox_std_dev
        ix += 1

    return anchor_class_matches, anchor_delta_targets



def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2) / 3D: (z1, z2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]  / 3D: (z1, z2)
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    if boxes.shape[1] > 5:
        boxes[:, 4] = boxes[:, 4].clamp(float(window[4]), float(window[5]))
        boxes[:, 5] = boxes[:, 5].clamp(float(window[4]), float(window[5]))

    return boxes

############################################################
#  Connected Componenent Analysis
############################################################

def get_coords(binary_mask, n_components, dim):
    """
    loops over batch to perform connected component analysis on binary input mask. computes box coordinates around
    n_components - biggest components (rois).
    :param binary_mask: (b, y, x, (z)). binary mask for one specific foreground class.
    :param n_components: int. number of components to extract per batch element and class.
    :return: coords (b, n, (y1, x1, y2, x2 (,z1, z2))
    :return: batch_components (b, n, (y1, x1, y2, x2, (z1), (z2))
    """
    assert len(binary_mask.shape)==dim+1
    binary_mask = binary_mask.astype('uint8')
    batch_coords = []
    batch_components = []
    for ix,b in enumerate(binary_mask):
        clusters, n_cands = lb(b)  # performs connected component analysis.
        uniques, counts = np.unique(clusters, return_counts=True)
        keep_uniques = uniques[1:][np.argsort(counts[1:])[::-1]][:n_components] #only keep n_components largest components
        p_components = np.array([(clusters == ii) * 1 for ii in keep_uniques])  # separate clusters and concat
        p_coords = []
        if p_components.shape[0] > 0:
            for roi in p_components:
                mask_ixs = np.argwhere(roi != 0)

                # get coordinates around component.
                roi_coords = [np.min(mask_ixs[:, 0]) - 1, np.min(mask_ixs[:, 1]) - 1, np.max(mask_ixs[:, 0]) + 1,
                               np.max(mask_ixs[:, 1]) + 1]
                if dim == 3:
                    roi_coords += [np.min(mask_ixs[:, 2]), np.max(mask_ixs[:, 2])+1]
                p_coords.append(roi_coords)

            p_coords = np.array(p_coords)

            #clip coords.
            p_coords[p_coords < 0] = 0
            p_coords[:, :4][p_coords[:, :4] > binary_mask.shape[-2]] = binary_mask.shape[-2]
            if dim == 3:
                p_coords[:, 4:][p_coords[:, 4:] > binary_mask.shape[-1]] = binary_mask.shape[-1]

        batch_coords.append(p_coords)
        batch_components.append(p_components)
    return batch_coords, batch_components


# noinspection PyCallingNonCallable
def get_coords_gpu(binary_mask, n_components, dim):
    """
    loops over batch to perform connected component analysis on binary input mask. computes box coordiantes around
    n_components - biggest components (rois).
    :param binary_mask: (b, y, x, (z)). binary mask for one specific foreground class.
    :param n_components: int. number of components to extract per batch element and class.
    :return: coords (b, n, (y1, x1, y2, x2 (,z1, z2))
    :return: batch_components (b, n, (y1, x1, y2, x2, (z1), (z2))
    """
    raise Exception("throws floating point exception")
    assert len(binary_mask.shape)==dim+1
    binary_mask = binary_mask.type(torch.uint8)
    batch_coords = []
    batch_components = []
    for ix,b in enumerate(binary_mask):
        clusters, n_cands = lb(b.cpu().data.numpy())  # peforms connected component analysis.
        clusters = torch.from_numpy(clusters).cuda()
        uniques = torch.unique(clusters)
        counts = torch.stack([(clusters==unique).sum() for unique in uniques])
        keep_uniques = uniques[1:][torch.sort(counts[1:])[1].flip(0)][:n_components] #only keep n_components largest components
        p_components = torch.cat([(clusters == ii).unsqueeze(0) for ii in keep_uniques]).cuda()  # separate clusters and concat
        p_coords = []
        if p_components.shape[0] > 0:
            for roi in p_components:
                mask_ixs = torch.nonzero(roi)

                # get coordinates around component.
                roi_coords = [torch.min(mask_ixs[:, 0]) - 1, torch.min(mask_ixs[:, 1]) - 1,
                              torch.max(mask_ixs[:, 0]) + 1,
                              torch.max(mask_ixs[:, 1]) + 1]
                if dim == 3:
                    roi_coords += [torch.min(mask_ixs[:, 2]), torch.max(mask_ixs[:, 2])+1]
                p_coords.append(roi_coords)

            p_coords = torch.tensor(p_coords)

            #clip coords.
            p_coords[p_coords < 0] = 0
            p_coords[:, :4][p_coords[:, :4] > binary_mask.shape[-2]] = binary_mask.shape[-2]
            if dim == 3:
                p_coords[:, 4:][p_coords[:, 4:] > binary_mask.shape[-1]] = binary_mask.shape[-1]

        batch_coords.append(p_coords)
        batch_components.append(p_components)
    return batch_coords, batch_components


############################################################
#  Pytorch Utility Functions
############################################################

def unique1d(tensor):
    """discard all elements of tensor that occur more than once; make tensor unique.
    :param tensor:
    :return:
    """
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = torch.tensor([True], dtype=torch.bool, requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool.data]


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort(descending=True)[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]



def shem(roi_probs_neg, negative_count, poolsize):
    """
    stochastic hard example mining: from a list of indices (referring to non-matched predictions),
    determine a pool of highest scoring (worst false positives) of size negative_count*poolsize.
    Then, sample n (= negative_count) predictions of this pool as negative examples for loss.
    :param roi_probs_neg: tensor of shape (n_predictions, n_classes).
    :param negative_count: int.
    :param poolsize: int.
    :return: (negative_count).  indices refer to the positions in roi_probs_neg. If pool smaller than expected due to
    limited negative proposals availabel, this function will return sampled indices of number < negative_count without
    throwing an error.
    """
    # sort according to higehst foreground score.
    probs, order = roi_probs_neg[:, 1:].max(1)[0].sort(descending=True)
    select = torch.tensor((poolsize * int(negative_count), order.size()[0])).min().int()

    pool_indices = order[:select]
    rand_idx = torch.randperm(pool_indices.size()[0])
    return pool_indices[rand_idx[:negative_count].cuda()]


############################################################
#  Weight Init
############################################################


def initialize_weights(net):
    """Initialize model weights. Current Default in Pytorch (version 0.4.1) is initialization from a uniform distriubtion.
    Will expectably be changed to kaiming_uniform in future versions.
    """
    init_type = net.cf.weight_init

    for m in [module for module in net.modules() if type(module) in [torch.nn.Conv2d, torch.nn.Conv3d,
                                                                     torch.nn.ConvTranspose2d,
                                                                     torch.nn.ConvTranspose3d,
                                                                     torch.nn.Linear]]:
        if init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        elif init_type == 'xavier_normal':
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        elif init_type == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity=net.cf.relu, a=0)
            if m.bias is not None:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / np.sqrt(fan_out)
                torch.nn.init.uniform_(m.bias, -bound, bound)

        elif init_type == "kaiming_normal":
            torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity=net.cf.relu, a=0)
            if m.bias is not None:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / np.sqrt(fan_out)
                torch.nn.init.normal_(m.bias, -bound, bound)
    net.logger.info("applied {} weight init.".format(init_type))