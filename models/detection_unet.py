import warnings
import os
import shutil
import time

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



import utils.exp_utils as utils
import utils.model_utils as mutils

'''
Use nn.DataParallel to use more than one GPU
'''

def center_crop_2D_image_batched(img, crop_size):
    # from batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
    # dim 0 is batch, dim 1 is channel, dim 2 and 3 are x y
    center = np.array(img.shape[2:]) / 2.
    if not hasattr(crop_size, "__iter__"):
        center_crop = [int(crop_size)] * (len(img.shape) - 2)
    else:
        center_crop = np.array(crop_size)
        assert len(center_crop) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"
    return img[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]

def center_crop_3D_image_batched(img, crop_size):
    # dim 0 is batch, dim 1 is channel, dim 2, 3 and 4 are x y z
    center = np.array(img.shape[2:]) / 2.
    if not hasattr(crop_size, "__iter__"):
        center_crop = np.array([int(crop_size)] * (len(img.shape) - 2))
    else:
        center_crop = np.array(crop_size)
        assert len(center_crop) == (len(
            img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"
    return img[:, :, int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.),
           int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]


def centercrop_vol(tensor, size):
    """:param tensor: tensor whose last two dimensions should be centercropped to size
    :param size: 2- or 3-int tuple of target (height, width(,depth))
    """
    dim = len(size)
    if dim==2:
        center_crop_2D_image_batched(tensor, size)
    elif dim==3:
        center_crop_2D_image_batched(tensor, size)
    else:
        raise Exception("invalid size argument {} encountered in centercrop".format(size))

    """this below worked so fine, when optional z-dim was first spatial dim instead of last
    h_, w_ = size[0], size[1] #target size
    (h,w) = tensor.size()[-2:] #orig size
    dh, dw = h-h_, w-w_ #deltas
    if dim == 3:
        d_ = size[2]
        d  = tensor.size()[-3]
        dd = d-d_
        
    if h_<h:
        tensor = tensor[...,dh//2:-int(math.ceil(dh/2.)),:] #crop height
    elif h_>=h:
        print("no h crop")
        warn.warn("no height crop applied since target dims larger equal orig dims")
    if w_<w:
        tensor = tensor[...,dw//2:-int(math.ceil(dw/2.))]
    elif w_>=w:
        warn.warn("no width crop applied since target dims larger equal orig dims")
    if dim == 3:
        if d_ < d:
            tensor = tensor[..., dd // 2:-int(math.ceil(dd / 2.)),:,:]
        elif d_ >= d:
            warn.warn("no depth crop applied since target dims larger equal orig dims")
    """

    return tensor
    
def dimcalc_conv2D(dims,F=3,s=1,pad="same"):
    r"""
    :param dims: orig width, height as (2,)-np.array
    :param F: quadratic kernel size
    :param s: stride
    :param pad: pad
    """
    if pad=="same":
        pad = (F-1)//2
    h, w = dims[0], dims[1] 
    return np.floor([(h + 2*pad-F)/s+1, (w+ 2*pad-F)/s+1])

def dimcalc_transconv2D(dims,F=2,s=2):
    r"""
    :param dims: orig width, height as (2,)-np.array
    :param F: quadratic kernel size
    :param s: stride
    """    

    h, w = dims[0], dims[1]
    return np.array([(h-1)*s+F, (w-1)*s+F])

def dimcalc_Unet_std(init_dims, F=3, F_pool=2, F_up=2, s=1, s_pool=2, s_up=2, pad=0):
    r"""Calculate theoretic dimensions of feature maps throughout layers of this U-net.
    """
    dims = np.array(init_dims)
    print("init dims: ", dims)
    
    def down(dims):
        for i in range(2):
            dims = dimcalc_conv2D(dims, F=F, s=s, pad=pad)       
        dims = dimcalc_conv2D(dims, F=F_pool, s=s_pool)     
        return dims.astype(int)    
    def up(dims):
        for i in range(2):
            dims = dimcalc_conv2D(dims, F=F, s=s, pad=pad)
        dims = dimcalc_transconv2D(dims, F=F_up,s=s_up)
        return dims.astype(int)
    
    stage = 1
    for i in range(4):
        dims = down(dims)
        print("stage ", stage, ": ", dims)
        stage+=1
    for i in range(4):
        dims = up(dims)
        print("stage ", stage, ": ", dims)
        stage+=1
    for i in range(2):
        dims = dimcalc_conv2D(dims,F=F,s=s, pad=pad).astype(int)
    print("final output size: ", dims)
    return dims

def dimcalc_Unet(init_dims, F=3, F_pool=2, F_up=2, s=1, s_pool=2, s_up=2, pad=0):
    r"""Calculate theoretic dimensions of feature maps throughout layers of this U-net.
    """
    dims = np.array(init_dims)
    print("init dims: ", dims)
    
    def down(dims):
        for i in range(3):
            dims = dimcalc_conv2D(dims, F=F, s=s, pad=pad)       
        dims = dimcalc_conv2D(dims, F=F_pool, s=s_pool)     
        return dims.astype(int)    
    def up(dims):
        dims = dimcalc_transconv2D(dims, F=F_up,s=s_up)
        for i in range(3):
            dims = dimcalc_conv2D(dims, F=F, s=s, pad=pad)
        return dims.astype(int)
    
    stage = 1
    for i in range(6):
        dims = down(dims)
        print("stage ", stage, ": ", dims)
        stage+=1
    for i in range(3):
        dims = dimcalc_conv2D(dims, F=F, s=s, pad=pad)
    for i in range(6):
        dims = up(dims)
        print("stage ", stage, ": ", dims)
        stage+=1
    dims = dims.astype(int)
    print("final output size: ", dims)
    return dims



class horiz_conv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, c_gen, norm, pad=0, relu="relu", bottleneck=True):
        super(horiz_conv, self).__init__()
        #TODO maybe make res-block?
        if bottleneck:
            bottleneck = int(np.round((in_chans+out_chans)*3/8))
            #print("bottleneck:", bottleneck)
        else:
            bottleneck = out_chans
        self.conv = nn.Sequential(
            c_gen(in_chans, bottleneck, kernel_size, pad=pad, norm=norm, relu=relu), #TODO maybe use norm only on last conv?
            c_gen(bottleneck, out_chans, kernel_size, pad=pad, norm=norm, relu=relu), #TODO maybe make bottleneck?
            #c_gen(out_chans, out_chans, kernel_size, pad=pad, norm=norm, relu=relu),
            )
    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, interpol, c_gen, norm, pad=0, relu="relu", stride_ip=2):
        super(up, self).__init__()
        self.dim = c_gen.dim
        self.upsample = interpol(stride_ip, "bilinear") if self.dim==2 else interpol(stride_ip, "trilinear") #TODO check if fits with spatial dims order in data
        self.reduce_chans = c_gen(in_chans, out_chans, ks=1, norm=norm, relu=None)
        self.horiz = horiz_conv(out_chans*2, out_chans, kernel_size, c_gen, norm=norm, pad=pad, relu=relu)

    def forward(self, x, skip_inp):
        #TODO maybe add highway weights in skips?
        x = self.upsample(x)
        x = self.reduce_chans(x)
        #print("shape x, skip", x.shape, skip_inp.shape)
        targ_size = x.size()[-self.dim:] #ft map x,y,z (spatial)
        skip_inp = centercrop_vol(skip_inp, targ_size)
        assert targ_size == skip_inp.size()[-self.dim:], "corresp. skip and forward dimensions don't match"
        x = torch.cat((x,skip_inp),dim=1)
        x = self.horiz(x)
        return x

   
class net(nn.Module):
    r"""U-Net with few more steps than standard.
    
    Dimensions: 
        feature maps have dims ...xhxwxd, d=feature map depth, h, w = orig 
        img height, width. h,w each are downsized by unpadded forward-convs and pooling,
        upsized by upsampling or upconvolution.
        If :math:`F\times F` is the single kernel_size and stride is :math:`s\geq 1`, 
        :math:`k` is the number of kernels in the conv, i.e. the resulting feature map depth,
        (all may differ between operations), then
    
    :Forward Conv: input  :math:`h \times w \times d` is converted to
    .. math:: \left[ (h-F)//s+1 \right] \times \left[ (w-F)//s+1 \right] \times k
    
    :Pooling: input  :math:`h \times w \times d` is converted to
    .. math:: \left[ (h-F)//s+1 \right] \times \left[ (w-F)//s+1 \right] \times d,
    pooling filters have no depths => orig depths preserved.

    :Up-Conv.: input  :math:`h \times w \times d` is converted to
    .. math:: \left[ (h-1)s + F \right] \times \left[ (w-1)s + F \right] \times k
    """


    def down(self, in_chans, out_chans, kernel_size, kernel_size_m, pad=0, relu="relu",maintain_z=False):
        """generate encoder block
        :param in_chans:
        :param out_chans:
        :param kernel_size:
        :param pad:
        :return:
        """
        if maintain_z and self.dim==3:
            stride_pool = (2,2,1)
            if not hasattr(kernel_size_m, "__iter__"):
                kernel_size_m = [kernel_size_m]*self.dim
            kernel_size_m = (*kernel_size_m[:-1], 1)
        else:
            stride_pool = 2
        module = nn.Sequential(
            nn.MaxPool2d(kernel_size_m, stride=stride_pool) if self.dim == 2 else nn.MaxPool3d(
                kernel_size_m, stride=stride_pool),
            #--> needs stride 2 in z in upsampling as well!
            horiz_conv(in_chans, out_chans, kernel_size, self.c_gen, self.norm, pad, relu=relu)
        )
        return module

    def up(self, in_chans, out_chans, kernel_size, pad=0, relu="relu", maintain_z=False):
        """generate decoder block
        :param in_chans:
        :param out_chans:
        :param kernel_size:
        :param pad:
        :param relu:
        :return:
        """
        if maintain_z and self.dim==3:
            stride_ip = (2,2,1)
        else:
            stride_ip = 2

        module = up(in_chans, out_chans, kernel_size, self.Interpolator, self.c_gen, norm=self.norm, pad=pad,
                    relu=relu, stride_ip=stride_ip)

        return module


    def __init__(self, cf, logger):
        super(net, self).__init__()

        self.cf = cf
        self.dim = cf.dim
        self.norm = cf.norm
        self.logger = logger
        backbone = utils.import_module('bbone', cf.backbone_path)
        self.c_gen = backbone.ConvGenerator(cf.dim)
        self.Interpolator = backbone.Interpolate

        #down = DownBlockGen(cf.dim)
        #up = UpBlockGen(cf.dim, backbone.Interpolate)
        down = self.down
        up = self.up

        pad = cf.pad
        if pad=="same":
            pad = (cf.kernel_size-1)//2

        
        self.dims = "not yet recorded"
        self.is_cuda = False
              
        self.init = horiz_conv(len(cf.channels), cf.init_filts, cf.kernel_size, self.c_gen, self.norm, pad=pad,
                               relu=cf.relu)
        
        self.down1 = down(cf.init_filts,    cf.init_filts*2,  cf.kernel_size, cf.kernel_size_m, pad=pad, relu=cf.relu)
        self.down2 = down(cf.init_filts*2,  cf.init_filts*4,  cf.kernel_size, cf.kernel_size_m, pad=pad, relu=cf.relu)
        self.down3 = down(cf.init_filts*4,  cf.init_filts*6,  cf.kernel_size, cf.kernel_size_m, pad=pad, relu=cf.relu)
        self.down4 = down(cf.init_filts*6,  cf.init_filts*8,  cf.kernel_size, cf.kernel_size_m, pad=pad, relu=cf.relu,
                          maintain_z=True)
        self.down5 = down(cf.init_filts*8,  cf.init_filts*12, cf.kernel_size, cf.kernel_size_m, pad=pad, relu=cf.relu,
                          maintain_z=True)
        #self.down6 = down(cf.init_filts*10, cf.init_filts*14, cf.kernel_size, cf.kernel_size_m, pad=pad, relu=cf.relu)
        
        #self.up1 = up(cf.init_filts*14, cf.init_filts*10, cf.kernel_size, pad=pad, relu=cf.relu)
        self.up2 = up(cf.init_filts*12, cf.init_filts*8,  cf.kernel_size, pad=pad, relu=cf.relu, maintain_z=True)
        self.up3 = up(cf.init_filts*8,  cf.init_filts*6,  cf.kernel_size, pad=pad, relu=cf.relu, maintain_z=True)
        self.up4 = up(cf.init_filts*6,  cf.init_filts*4,  cf.kernel_size, pad=pad, relu=cf.relu)
        self.up5 = up(cf.init_filts*4,  cf.init_filts*2,  cf.kernel_size, pad=pad, relu=cf.relu)
        self.up6 = up(cf.init_filts*2,  cf.init_filts,    cf.kernel_size, pad=pad, relu=cf.relu)
        
        self.seg = self.c_gen(cf.init_filts, cf.num_seg_classes, 1, norm=None, relu=None)


        # initialize parameters
        if self.cf.weight_init == "custom":
            logger.info("Tried to use custom weight init which is not defined. Using pytorch default.")
        elif self.cf.weight_init:
            mutils.initialize_weights(self)
        else:
            logger.info("using default pytorch weight init")
        
    
    def forward(self, x):
        r'''Forward application of network-function.
        
        :param x: input to the network, expected as torch.tensor of dims
        .. math:: batch\_size \times channels \times height \times width
        requires_grad should be True for training
        '''
        #self.dims = np.array([x.size()[-self.dim-1:]])
        
        x1 = self.init(x)
        #self.dims = np.vstack((self.dims, x1.size()[-self.dim-1:]))
        
        #---downwards---
        x2 = self.down1(x1)
        #self.dims = np.vstack((self.dims, x2.size()[-self.dim-1:]))
        x3 = self.down2(x2)
        #self.dims = np.vstack((self.dims, x3.size()[-self.dim-1:]))
        x4 = self.down3(x3)
        #self.dims = np.vstack((self.dims, x4.size()[-self.dim-1:]))
        x5 = self.down4(x4)
        #self.dims = np.vstack((self.dims, x5.size()[-self.dim-1:]))
        #x6 = self.down5(x5)
        #self.dims = np.vstack((self.dims, x6.size()[-self.dim-1:]))
        
        #---bottom---
        x = self.down5(x5)
        #self.dims = np.vstack((self.dims, x.size()[-self.dim-1:]))
        
        #---upwards---
        #x = self.up1(x, x6)
        #self.dims = np.vstack((self.dims, x.size()[-self.dim-1:]))
        x = self.up2(x, x5)
        #self.dims = np.vstack((self.dims, x.size()[-self.dim-1:]))
        x = self.up3(x, x4)
        #self.dims = np.vstack((self.dims, x.size()[-self.dim-1:]))
        x = self.up4(x, x3)
        #self.dims = np.vstack((self.dims, x.size()[-self.dim-1:]))
        x = self.up5(x, x2)
        #self.dims = np.vstack((self.dims, x.size()[-self.dim-1:]))

        x = self.up6(x, x1)
        #self.dims = np.vstack((self.dims, x.size()[-self.dim-1:]))

        # ---final---
        x = self.seg(x)
        #self.dims = np.vstack((self.dims, x.size()[-self.dim-1:]))

        seg_logits = x
        out_box_coords, out_scores = [], []
        seg_probs = F.softmax(seg_logits.detach(), dim=1).cpu().data.numpy()
        #seg_probs = F.softmax(seg_logits, dim=1)

        assert seg_logits.shape[1]==self.cf.num_seg_classes
        for cl in range(1, seg_logits.shape[1]):
            hard_mask = np.copy(seg_probs).argmax(1)
            #hard_mask = seg_probs.clone().argmax(1)
            hard_mask[hard_mask != cl] = 0
            hard_mask[hard_mask == cl] = 1
            # perform connected component analysis on argmaxed predictions,
            # draw boxes around components and return coordinates.
            box_coords, rois = mutils.get_coords(hard_mask, self.cf.n_roi_candidates, self.cf.dim)

            # for each object, choose the highest softmax score (in the respective class)
            # of all pixels in the component as object score.
            scores = [[] for b_inst in range(x.shape[0])]  # np.zeros((out_features.shape[0], self.cf.n_roi_candidates))
            for b_inst, brois in enumerate(rois):
                for nix, nroi in enumerate(brois):
                    score_det = np.max if self.cf.score_det == "max" else np.median  # score determination
                    scores[b_inst].append(score_det(seg_probs[b_inst, cl][nroi > 0]))
            out_box_coords.append(box_coords)
            out_scores.append(scores)

        return seg_logits, out_box_coords, out_scores

    # noinspection PyCallingNonCallable
    def train_forward(self, batch, **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :param kwargs:
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes]
                'torch_loss': 1D torch tensor for backprop.
                'class_loss': classification loss for monitoring. here: dummy array, since no classification conducted.
        """

        img = torch.from_numpy(batch["data"]).float().cuda()
        seg = torch.from_numpy(batch["seg"]).long().cuda()
        seg_ohe = torch.from_numpy(mutils.get_one_hot_encoding(batch['seg'], self.cf.num_seg_classes)).float().cuda()

        results_dict = {}
        seg_logits, box_coords, scores = self.forward(img)

        # no extra class loss applied in this model. pass dummy tensor for monitoring.
        results_dict['class_loss'] = np.nan

        results_dict['boxes'] = [[] for _ in range(img.shape[0])]
        for cix in range(len(self.cf.class_dict.keys())):
            for bix in range(img.shape[0]):
                for rix in range(len(scores[cix][bix])):
                    if scores[cix][bix][rix] > self.cf.detection_min_confidence:
                        results_dict['boxes'][bix].append({'box_coords': np.copy(box_coords[cix][bix][rix]),
                                                           'box_score': scores[cix][bix][rix],
                                                           'box_pred_class_id': cix + 1,  # add 0 for background.
                                                           'box_type': 'det',
                                                           })

        for bix in range(img.shape[0]): #bix = batch-element index
            for tix in range(len(batch['bb_target'][bix])): #target index
                gt_box = {'box_coords': batch['bb_target'][bix][tix], 'box_type': 'gt'}
                for name in self.cf.roi_items:
                    gt_box.update({name: batch[name][bix][tix]})
                results_dict['boxes'][bix].append(gt_box)

        # compute segmentation loss as either weighted cross entropy, dice loss, or the sum of both.
        seg_pred = F.softmax(seg_logits, 1)
        loss = torch.tensor([0.], dtype=torch.float, requires_grad=False).cuda()
        if self.cf.seg_loss_mode == 'dice' or self.cf.seg_loss_mode == 'dice_wce':
            loss += 1 - mutils.batch_dice(seg_pred, seg_ohe.float(),
                                         false_positive_weight=float(self.cf.fp_dice_weight))

        if self.cf.seg_loss_mode == 'wce' or self.cf.seg_loss_mode == 'dice_wce':
            loss += F.cross_entropy(seg_logits, seg[:, 0], weight=torch.FloatTensor(self.cf.wce_weights).cuda(),
                                    reduction='mean')

        results_dict['torch_loss'] = loss
        seg_pred = seg_pred.argmax(dim=1).unsqueeze(dim=1).cpu().data.numpy()
        results_dict['seg_preds'] = seg_pred
        if 'dice' in self.cf.metrics:
            results_dict['batch_dices'] = mutils.dice_per_batch_and_class(seg_pred, batch["seg"],
                                                                           self.cf.num_seg_classes, convert_to_ohe=True)
            #print("batch dice scores ", results_dict['batch_dices'] )
        # self.logger.info("loss: {0:.2f}".format(loss.item()))
        return results_dict

    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :param kwargs:
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes]
        """
        img = torch.FloatTensor(batch['data']).cuda()
        seg_logits, box_coords, scores = self.forward(img)

        results_dict = {}
        results_dict['boxes'] = [[] for b_inst in range(img.shape[0])]
        for cix in range(len(box_coords)): #class index
            for bix in range(img.shape[0]): #batch instance
                for rix in range(len(scores[cix][bix])): #range(self.cf.n_roi_candidates):
                    if scores[cix][bix][rix] > self.cf.detection_min_confidence:
                        results_dict['boxes'][bix].append({'box_coords': np.copy(box_coords[cix][bix][rix]),
                                    'box_score': scores[cix][bix][rix],
                                    'box_pred_class_id': cix + 1,
                                    'box_type': 'det'})
        # carry probs instead of preds to use for multi-model voting in predictor
        results_dict['seg_preds'] = F.softmax(seg_logits, dim=1).cpu().data.numpy()


        return results_dict


    def actual_dims(self, print_=True):
        r"""Return dimensions of actually calculated layers at beginning of each block.
        """
        if print_:
            print("dimensions as recorded in forward pass: ")
            for stage in range(len(self.dims)):
                print("Stage ", stage, ": ", self.dims[stage])
        return self.dims
        
    def cuda(self, device=None):
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        try:
            self.loss_f = self.loss_f.cuda()
        except:
            pass
        self.is_cuda = True
        return self._apply(lambda t: t.cuda(device))
    
    def cpu(self):
        r"""Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        self.is_cuda = False
        return self._apply(lambda t: t.cpu()) 




        