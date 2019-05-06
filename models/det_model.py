# coding=utf-8
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .networks import RPN, ROIHead
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cupy as cp
from util import non_maximum_suppression
from .tools import CropResize
import pdb

plt.switch_backend('agg')


# def bbox_iou(box, gt_box):
#     xmin, ymin, xmax, ymax = box
#     gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
#     x_overlap = max(0, min(xmax, gt_xmax) - max(xmin, gt_xmin))
#     y_overlap = max(0, min(ymax, gt_ymax) - max(ymin, gt_ymin))
#     i_area = x_overlap * y_overlap
#     u_area = (xmax - xmin) * (ymax - ymin) + (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) - i_area
#     iou = float(i_area) / float(u_area)
#     return iou


class RPNLoss(nn.Module):
    def __init__(self, img_size=512, stride=16):
        super(RPNLoss, self).__init__()
        feat_size = img_size / stride
        self.feat_size = feat_size
        self.stride = stride
        anw, anh = np.meshgrid([1.0, 0.5, 0.25], [1.0, 0.5, 0.25])
        """
        从第1到第9通道对应的xy比例分别是
        [[1.  , 0.5 , 0.25, 1.  , 0.5 , 0.25, 1.  , 0.5 , 0.25],
         [1.  , 1.  , 1.  , 0.5 , 0.5 , 0.5 , 0.25, 0.25, 0.25]]
        """
        anw = anw.ravel() * img_size
        anh = anh.ravel() * img_size
        anw = anw[..., None, None]
        anh = anh[..., None, None]

        an_xct, an_yct = np.meshgrid(range(feat_size), range(feat_size))
        an_xct *= stride
        an_yct *= stride
        self.anw = np.tile(anw, (1, feat_size, feat_size))
        self.anh = np.tile(anh, (1, feat_size, feat_size))
        self.an_xct = np.tile(an_xct[None, ...], (9, 1, 1))
        self.an_yct = np.tile(an_yct[None, ...], (9, 1, 1))
        self.an_xmin = an_xct - anw/2
        self.an_ymin = an_yct - anh/2
        self.an_xmax = an_xct + anw/2
        self.an_ymax = an_yct + anh/2

        self.pos_th = 0.7
        self.neg_th = 0.3
        self.w_pos = 1.0
        self.w_mask = 1.0

    def iou_gt_anchor(self, gt_box):
        K = gt_box.shape[0]

        gt_xmin = gt_box[:, 0]
        gt_ymin = gt_box[:, 1]
        gt_xmax = gt_box[:, 2]
        gt_ymax = gt_box[:, 3]

        gt_xmin = gt_xmin[..., None, None, None]
        gt_ymin = gt_ymin[..., None, None, None]
        gt_xmax = gt_xmax[..., None, None, None]
        gt_ymax = gt_ymax[..., None, None, None]

        an_xmin = self.an_xmin[None, ...]
        an_ymin = self.an_ymin[None, ...]
        an_xmax = self.an_xmax[None, ...]
        an_ymax = self.an_ymax[None, ...]

        # # big_x = np.zeros((K, 9, feat_size, feat_size, 4))
        # ipdb.set_trace()
        an_xmin = np.broadcast_to(an_xmin, (K, 9, self.feat_size, self.feat_size))
        an_ymin = np.broadcast_to(an_ymin, (K, 9, self.feat_size, self.feat_size))
        an_xmax = np.broadcast_to(an_xmax, (K, 9, self.feat_size, self.feat_size))
        an_ymax = np.broadcast_to(an_ymax, (K, 9, self.feat_size, self.feat_size))

        gt_xmin = np.broadcast_to(gt_xmin, (K, 9, self.feat_size, self.feat_size))
        gt_ymin = np.broadcast_to(gt_ymin, (K, 9, self.feat_size, self.feat_size))
        gt_xmax = np.broadcast_to(gt_xmax, (K, 9, self.feat_size, self.feat_size))
        gt_ymax = np.broadcast_to(gt_ymax, (K, 9, self.feat_size, self.feat_size))

        big_min_x = an_xmin.copy()
        big_min_y = an_ymin.copy()
        small_max_x = an_xmax.copy()
        small_max_y = an_ymax.copy()

        mask = gt_xmin > an_xmin
        big_min_x[mask] = gt_xmin[mask]
        mask = gt_ymin > an_ymin
        big_min_y[mask] = gt_ymin[mask]
        mask = gt_xmax < an_xmax
        small_max_x[mask] = gt_xmax[mask]
        mask = gt_ymax < an_ymax
        small_max_y[mask] = gt_ymax[mask]

        overlap_x = small_max_x - big_min_x
        overlap_x[overlap_x<0] = 0
        overlap_y = small_max_y - big_min_y
        overlap_y[overlap_y<0] = 0
        i_area = overlap_x*overlap_y
        u_area = (an_xmax - an_xmin) * (an_ymax - an_ymin) + (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) - i_area
        iou = i_area / (u_area+1e-8)
        return iou

    def bbox2target(self, gt_box):
        K = gt_box.shape[0]
        iou = self.iou_gt_anchor(gt_box)
        iou = iou.reshape((K, -1))
        # 如果iou最大的anchors的数量超过了128个，随机取128个
        # 否则，如果iou>pos_th的anchors数量够的话，随机取一些凑够128个，不够的话就有多少凑多少
        # 然后随机取一些iou<neg_th的anchors凑够总共256个。（假设iou<neg_th的总是足够多）
        idx_best_ans = iou.argmax(1)  # 最佳anchor
        idx_best_ans = np.unique(idx_best_ans)
        maxiou = iou.max(0)  # 对于每个anchor，考虑和它重叠最大的gtbox和它的iou
        maxiou[idx_best_ans] = -1  # 最佳anchor已经找到了，要避免之后重复找它
        gt_ans = iou.argmax(0)  # 每个anchor对应的gtbox
        gt_best_ans = gt_ans[idx_best_ans]  # 最佳anchor对应的gtbox
        if len(idx_best_ans) < 128:
            idx_pos_ans = np.where(maxiou>self.pos_th)[0]  # positive anchor
            gt_pos_ans = gt_ans[idx_pos_ans]  # positive anchor对应的gtbox
            if len(idx_pos_ans) > 128-len(idx_best_ans):
                _rand_idx = random.sample(range(len(idx_pos_ans)), 128-len(idx_best_ans))
                idx_pos_ans = idx_pos_ans[_rand_idx]
                gt_pos_ans = gt_pos_ans[_rand_idx]
            idx_pos_ans = np.concatenate((idx_best_ans, idx_pos_ans), 0)
            gt_pos_ans = np.concatenate((gt_best_ans, gt_pos_ans), 0)
        else:
            _rand_idx = random.sample(range(len(idx_best_ans)), 128)
            idx_pos_ans = idx_best_ans[_rand_idx]
            gt_pos_ans = gt_best_ans[_rand_idx]
        idx_neg_ans = np.where((maxiou>0)&(maxiou<=self.neg_th))[0]
        idx_neg_ans = np.random.choice(idx_neg_ans, 256-len(idx_pos_ans))
        mask_pos_ans = np.zeros(9*self.feat_size*self.feat_size, dtype=np.bool)
        mask_pos_ans[idx_pos_ans] = 1
        mask_pos_ans = mask_pos_ans.reshape(9, self.feat_size, self.feat_size)
        mask_neg_ans = np.zeros(9*self.feat_size*self.feat_size, dtype=np.bool)
        mask_neg_ans[idx_neg_ans] = 1
        mask_neg_ans = mask_neg_ans.reshape(9, self.feat_size, self.feat_size)

        gt_xmin = gt_box[:, 0]
        gt_ymin = gt_box[:, 1]
        gt_xmax = gt_box[:, 2]
        gt_ymax = gt_box[:, 3]

        gt_xct = ((gt_xmin+gt_xmax)/2)[gt_pos_ans]
        gt_yct = ((gt_ymin+gt_ymax)/2)[gt_pos_ans]
        gtw = (gt_xmax-gt_xmin)[gt_pos_ans]
        gth = (gt_ymax-gt_ymin)[gt_pos_ans]

        an_xct = self.an_xct[mask_pos_ans]
        an_yct = self.an_yct[mask_pos_ans]
        anw = self.anw[mask_pos_ans]
        anh = self.anh[mask_pos_ans]

        # try:
        #     tx = (gt_xct-an_xct)/anw
        # except ValueError:
        #     pdb.set_trace()
        tx = (gt_xct-an_xct)/anw
        ty = (gt_yct-an_yct)/anh

        # assert (gtw<=0).sum()==0
        # assert (gth<=0).sum()==0

        tw = np.log(gtw/anw)
        th = np.log(gth/anh)
        gt_mask = np.ones(mask_pos_ans.shape, dtype=np.long)-2
        gt_mask[mask_pos_ans] = 1
        gt_mask[mask_neg_ans] = 0
        return gt_mask, (tx, ty, tw, th)

    def forward(self, pred, gt_box):
        pred_mask, pred_loc = pred
        gt_mask, gt_loc = self.bbox2target(gt_box.squeeze(0).numpy())
        gt_mask = torch.LongTensor(gt_mask[None, ...]).cuda()
        msk_loss = F.cross_entropy(pred_mask, gt_mask, ignore_index=-1)
        pred_loc = [pp.squeeze(1).contiguous() for pp in pred_loc.split(1, 1)]
        _msk = gt_mask==1
        pred_loc = [pp[_msk] for pp in pred_loc]
        gt_loc = [torch.Tensor(gg).cuda() for gg in gt_loc]
        loc_loss = sum([F.smooth_l1_loss(pred, gt) for pred, gt in zip(pred_loc, gt_loc)])
        loss = self.w_pos*loc_loss + self.w_mask*msk_loss
        return loss


class HeadLoss(nn.Module):
    def __init__(self, img_size=512, stride=16):
        super(HeadLoss, self).__init__()
        self.pos_th = 0.5
        self.neg_th = 0.1
        self.w_pos = 1.0
        self.w_cls = 1.0

    def iou_gt_roi(self, gt_box, roi_box):
        n_gt = gt_box.shape[0]
        n_roi = roi_box.shape[0]

        _gt = [gg[:, 0] for gg in np.split(gt_box, 4, 1)]
        _roi = [rr[:, 0] for rr in np.split(roi_box, 4, 1)]
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = [np.broadcast_to(gg[..., None], (n_gt, n_roi)) for gg in _gt]
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = [np.broadcast_to(rr[None, ...], (n_gt, n_roi)) for rr in _roi]
        big_min_x = roi_xmin.copy()
        big_min_y = roi_ymin.copy()
        small_max_x = roi_xmax.copy()
        small_max_y = roi_ymax.copy()

        mask = gt_xmin > roi_xmin
        big_min_x[mask] = gt_xmin[mask]
        mask = gt_ymin > roi_ymin
        big_min_y[mask] = gt_ymin[mask]
        mask = gt_xmax < roi_xmax
        small_max_x[mask] = gt_xmax[mask]
        mask = gt_ymax < roi_ymax
        small_max_y[mask] = gt_ymax[mask]

        overlap_x = small_max_x - big_min_x
        overlap_x[overlap_x<0] = 0
        overlap_y = small_max_y - big_min_y
        overlap_y[overlap_y<0] = 0
        i_area = overlap_x*overlap_y
        u_area = (roi_xmax - roi_xmin) * (roi_ymax - roi_ymin) + (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin) - i_area
        iou = i_area / (u_area+1e-8)
        return iou

    def bbox2target(self, gt_box, roi_box, labels):
        # gt_box = np.broadcast_to(gt_box, (2, 4))  # 调试用 等会记得删除
        iou = self.iou_gt_roi(gt_box, roi_box)
        maxiou = iou.max(0)  # 对于每个roi，考虑和它重叠最大的gtbox和它的iou
        gt_roi = iou.argmax(0)  # 每个roi对应的gtbox

        idx_pos_roi = np.where(maxiou>self.pos_th)[0]  # positive rois
        gt_pos_roi = gt_roi[idx_pos_roi]  # positive roi对应的gtbox
        if len(idx_pos_roi) > 64:
            _rand_idx = random.sample(range(len(idx_pos_roi)), 64)
            idx_pos_roi = idx_pos_roi[_rand_idx]
            gt_pos_roi = gt_pos_roi[_rand_idx]
        idx_neg_roi = np.where(maxiou<=self.neg_th)[0]  # negative roi
        _rand_idx = random.sample(range(len(idx_neg_roi)), min(128-len(idx_pos_roi), len(idx_neg_roi)))
        idx_neg_roi = idx_neg_roi[_rand_idx]

        gt_xmin = gt_box[:, 0]
        gt_ymin = gt_box[:, 1]
        gt_xmax = gt_box[:, 2]
        gt_ymax = gt_box[:, 3]

        roi_xmin = roi_box[:, 0]
        roi_ymin = roi_box[:, 1]
        roi_xmax = roi_box[:, 2]
        roi_ymax = roi_box[:, 3]

        gt_xct = ((gt_xmin+gt_xmax)/2)[gt_pos_roi]
        gt_yct = ((gt_ymin+gt_ymax)/2)[gt_pos_roi]
        gtw = (gt_xmax-gt_xmin)[gt_pos_roi]
        gth = (gt_ymax-gt_ymin)[gt_pos_roi]

        roi_xct = ((roi_xmin+roi_xmax)/2)[idx_pos_roi]
        roi_yct = ((roi_ymin+roi_ymax)/2)[idx_pos_roi]
        roiw = (roi_xmax-roi_xmin)[idx_pos_roi]
        roih = (roi_ymax-roi_ymin)[idx_pos_roi]

        tx = (gt_xct-roi_xct)/roiw
        ty = (gt_yct-roi_yct)/roih

        tw = np.log(gtw/roiw)
        th = np.log(gth/roih)

        pos_labels = labels[gt_pos_roi]
        return idx_neg_roi, idx_pos_roi, gt_pos_roi, (tx, ty, tw, th), pos_labels

    def forward(self, pred, gt_box, roi_box, labels, dbg=False):
        pred_cls, pred_loc = pred
        idx_neg_roi, idx_pos_roi, gt_pos_roi, gt_loc, pos_labels = \
            self.bbox2target(gt_box.squeeze(0).numpy(), roi_box.numpy(), labels.squeeze(0).numpy())
        gt_cls = torch.LongTensor(len(pred_cls)).fill_(-1)
        gt_cls[idx_pos_roi] = torch.LongTensor(pos_labels)+1
        gt_cls[idx_neg_roi] = 0
        gt_cls = gt_cls.contiguous().cuda()
        cls_loss = F.cross_entropy(pred_cls, gt_cls, ignore_index=-1)

        pred_loc = pred_loc[idx_pos_roi]
        labels = labels.squeeze(0)
        cls_pos_roi = labels[gt_pos_roi]
        cls_pos_roi = cls_pos_roi.cuda()
        if dbg:
            pdb.set_trace()
        cls_pos_roi = cls_pos_roi[..., None, None]
        cls_pos_roi = cls_pos_roi.expand(-1, -1, 4)
        pred_loc = torch.gather(pred_loc, 1, cls_pos_roi)
        pred_loc = [pp.squeeze(1).contiguous() for pp in pred_loc.squeeze(1).split(1, 1)]


        gt_loc = [torch.Tensor(gg).cuda() for gg in gt_loc]
        loc_loss = sum([F.smooth_l1_loss(pred, gt) for pred, gt in zip(pred_loc, gt_loc)])
        loss = cls_loss + loc_loss
        return loss


class DetModel(BaseModel):
    fig, ax = plt.subplots(1)
    plt.ion()


    def __init__(self, opt, n_classes=20):
        BaseModel.initialize(self, opt)
        self.name = opt.model + '_' + opt.base
        self.phase = opt.phase
        self.imageSize = opt.imageSize
        self.n_pre_nms_train = 5000
        self.n_post_nms_train = 600
        self.n_pre_nms_test = 4000
        self.n_post_nms_test = 300
        self.nms_th = 0.7

        self.loss = {}

        self.v_mean = self.Tensor(opt.mean)[None, ..., None, None]
        self.v_std = self.Tensor(opt.std)[None, ..., None, None]
        rpn = RPN(pretrained=opt.isTrain and (not opt.from_scratch),
                                                      base=opt.base)
        rpn = torch.nn.parallel.DataParallel(rpn, device_ids = opt.gpu_ids)
        self.rpn = rpn.cuda()
        head = ROIHead(pretrained=opt.isTrain and (not opt.from_scratch),
                  base=opt.base, n_classes=n_classes)
        head = torch.nn.parallel.DataParallel(head, device_ids = opt.gpu_ids)
        self.head = head.cuda()
        self.roi_pool = CropResize(oh=7, ow=7, scale=16)
        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                 opt.imageSize, opt.imageSize)
        self.criterion_rpn = RPNLoss(img_size=opt.imageSize)
        self.criterion_head = HeadLoss(img_size=opt.imageSize)
        if opt.phase is 'test':
            pass
        else:
            self.optimizer_rpn = torch.optim.Adam(self.rpn.parameters(),
                                                lr=opt.lr)
            self.optimizer_head = torch.optim.Adam(self.head.parameters(),
                                                  lr=opt.lr)

    def fig2data(self):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        self.fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)
        return buf

    def save(self, label):
        self.save_network(self.rpn, self.name+'_rpn', label)
        self.save_network(self.head, self.name+'_head', label)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.rpn, self.name+'_rpn', label)
        self.load_network(self.head, self.name+'_head', label)

    def rpnpred2roi(self):
        if self.phase is 'train':
            n_pre_nms = self.n_pre_nms_train
            n_post_nms = self.n_post_nms_train
        elif self.phase is 'test':
            n_pre_nms = self.n_pre_nms_test
            n_post_nms = self.n_post_nms_test
        pred_mask, pred_loc = self.rpn_prediction
        pred_prob = pred_mask.squeeze(0)[1].detach().cpu().numpy()
        pred_loc = [pp.squeeze(0).squeeze(0).detach().cpu().numpy() for pp in pred_loc.split(1, 1)]
        idx = pred_prob.reshape(-1).argsort()[::-1][:n_pre_nms]
        pred_tx, pred_ty, pred_tw, pred_th = [pp.reshape(-1)[idx] for pp in pred_loc]
        an_xct, an_yct, anw, anh = [aa[idx] for aa in [self.criterion_rpn.an_xct.reshape(-1), self.criterion_rpn.an_yct.reshape(-1),
                                                       self.criterion_rpn.anw.reshape(-1), self.criterion_rpn.anh.reshape(-1)]]
        pred_xct = pred_tx*anw+an_xct
        pred_yct = pred_ty*anh+an_yct
        pred_w = np.exp(pred_tw)*anw
        pred_h = np.exp(pred_th)*anh

        pred_xmin = pred_xct-pred_w/2
        pred_ymin = pred_yct-pred_h/2
        pred_xmax = pred_xct+pred_w/2
        pred_ymax = pred_yct+pred_h/2

        pred_xmin[pred_xmin<0]=0
        pred_ymin[pred_ymin<0]=0
        pred_xmax[pred_xmax>self.imageSize-1]=self.imageSize-1
        pred_ymax[pred_ymax>self.imageSize-1]=self.imageSize-1

        # 因为直接抄的别人的nms，所以顺序也要改成别人用的那种
        pred_box = np.stack((pred_ymin, pred_xmin, pred_ymax, pred_xmax), 1)
        idx = non_maximum_suppression(cp.ascontiguousarray(cp.asarray(pred_box)), self.nms_th)
        idx = idx[:n_post_nms]
        pred_box = pred_box[idx]
        # 然后再改回来
        pred_box = pred_box[:, [1,0,3,2]]
        return pred_box


    def show_tensorboard_eval(self, num_iter):
        for k, v in self.performance.items():
            self.writer.add_scalar(k, v, num_iter)

    def show_tensorboard(self, num_iter, num_show=4):
        for k, v in self.loss.items():
            self.writer.add_scalar(k, v, num_iter)
        pred_box = self.rpnpred2roi()
        pred_box = pred_box[:10]

        plt.cla()
        img = self.input[:num_show]*self.v_std + self.v_mean
        img = img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        img /= 255
        self.ax.imshow(img)
        for pos in pred_box:
            rect = patches.Rectangle(
                (pos[0], pos[1]), pos[2]-pos[0], pos[3]-pos[1], linewidth=2, edgecolor='r',
                facecolor='none')
            self.ax.add_patch(rect)
        img = self.fig2data()
        self.writer.add_image('image', img, num_iter)

    def set_input(self, input, gt_box=None, labels=None):
        self.input.resize_(input.size()).copy_(input.cuda())
        self.gt_box = gt_box
        self.labels = labels


    def forward(self):
        # print("We are Forwarding !!")
        self.rpn_prediction, feat = self.rpn.forward(self.input)
        roi_box = self.rpnpred2roi()
        roi_box = torch.Tensor(roi_box)
        if self.phase == 'train':
            roi_box = torch.cat((self.gt_box.squeeze(0), roi_box), 0)
            roi_box = roi_box.contiguous()
        self.roi_box = roi_box
        feat = self.roi_pool(feat, roi_box.cuda())
        self.head_prediction = self.head(feat)


    def test(self, input, name, WW, HH):
        raise NotImplementedError
        # self.set_input(input)
        # with torch.no_grad():
        #     self.forward()
        # outputs = self.prediction.detach().cpu().numpy()
        #
        # maximum = outputs.max(1).max(1)
        # out_img = outputs / maximum[:, None, None]
        # out_img = (out_img*255).astype(np.uint8)
        # for ii, msk in enumerate(out_img):
        #     msk = Image.fromarray(msk.astype(np.uint8))
        #     msk = msk.resize((WW[ii], HH[ii]))
        #     msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')
        #     np.save('{}/{}'.format(self.opt.results_dir, name[ii]), outputs[ii])


    def backward(self, dbg=False):
        # Combined loss
        loss_var_rpn = self.criterion_rpn(self.rpn_prediction, self.gt_box)
        loss_var_head = self.criterion_head(self.head_prediction, self.gt_box, self.roi_box, self.labels, dbg)
        loss_var = loss_var_rpn + loss_var_head

        loss_var.backward()
        self.loss['rpn'] = loss_var_rpn.item()
        self.loss['head'] = loss_var_head.item()
        # self.loss_head = loss_var_rpn.data[0]


    def optimize_parameters(self, dbg=False):
        self.forward()
        self.optimizer_rpn.zero_grad()
        self.optimizer_head.zero_grad()
        self.backward(dbg)
        self.optimizer_rpn.step()
        self.optimizer_head.step()


    def switch_to_train(self):
        self.rpn.train()
        self.head.train()
        self.phase = 'train'

    def switch_to_eval(self):
        self.rpn.eval()
        self.head.eval()
        self.phase = 'test'

