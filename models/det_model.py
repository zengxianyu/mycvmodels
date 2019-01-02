# coding=utf-8
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
from torch.autograd import Variable
from base_model import _BaseModel
import sys
# from evaluate_sal import fm_and_mae
from tensorboardX import SummaryWriter
from datetime import datetime
from rpn import RPN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pdb

thismodule = sys.modules[__name__]


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
    def __init__(self, img_size=512, feat_size=32, stride=16):
        super(RPNLoss, self).__init__()
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
        gt_best_ans = gt_ans[idx_best_ans]
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
        idx_neg_ans = np.where((maxiou>0)&(maxiou<self.neg_th))[0]
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

        gt_xct = (gt_xmin[gt_pos_ans]+gt_xmax[gt_pos_ans])/2
        gt_yct = (gt_ymin[gt_pos_ans]+gt_ymax[gt_pos_ans])/2
        gtw = gt_xmax[gt_pos_ans]-gt_xmin[gt_pos_ans]
        gth = gt_ymax[gt_pos_ans]-gt_ymin[gt_pos_ans]

        an_xct = self.an_xct[mask_pos_ans]
        an_yct = self.an_yct[mask_pos_ans]
        anw = self.anw[mask_pos_ans]
        anh = self.anh[mask_pos_ans]

        try:
            tx = (gt_xct-an_xct)/anw
        except ValueError:
            pdb.set_trace()
        ty = (gt_yct-an_yct)/anh

        assert (gtw<=0).sum()==0
        assert (gth<=0).sum()==0

        tw = np.log(gtw/anw)
        th = np.log(gth/anh)
        gt_mask = np.ones(mask_pos_ans.shape, dtype=np.long)-2
        gt_mask[mask_pos_ans] = 1
        gt_mask[mask_neg_ans] = 0
        return gt_mask, (tx, ty, tw, th)

    def forward(self, pred, gt_box):
        pred_mask, pred_pos = pred
        gt_mask, gt_pos = self.bbox2target(gt_box.squeeze(0).numpy())
        gt_mask = torch.LongTensor(gt_mask[None, ...]).cuda()
        msk_loss = F.cross_entropy(pred_mask, gt_mask, ignore_index=-1)
        # import matplotlib.pyplot as plt
        # gt_mask[gt_mask!=1] = 0
        # plt.imshow(gt_mask.sum(0))
        # plt.show()
        # pred_tx, pred_ty, pred_tw, pred_th = [pp.squeeze(1) for pp in pred_pos.split(1, 1)]
        pred_pos = [pp.squeeze(1).contiguous() for pp in pred_pos.split(1, 1)]
        _msk = gt_mask==1
        pred_pos = [pp[_msk] for pp in pred_pos]
        gt_pos = [torch.Tensor(gg).cuda() for gg in gt_pos]
        pos_loss = sum([F.smooth_l1_loss(pred, gt) for pred, gt in zip(pred_pos, gt_pos)])
        loss = self.w_pos*pos_loss + self.w_mask*msk_loss
        return loss



# if __name__ == "__main__":
#     sb = RPNLoss()
#     gt_box = np.array([[0., 0., 120., 90.], [120., 120., 240., 240.]])
#     mask_pos_ans, mask_neg_ans, (tx, ty, tw, th) = sb.bbox2target(gt_box)
#     ipdb.set_trace()



class DetModel(_BaseModel):
    fig, ax = plt.subplots(1)
    plt.ion()

    def __init__(self, opt):
        _BaseModel.initialize(self, opt)
        self.name = opt.model + '_' + opt.base
        self.imageSize = opt.imageSize

        self.v_mean = self.Tensor(opt.mean)[None, ..., None, None]
        self.v_std = self.Tensor(opt.std)[None, ..., None, None]
        net = getattr(thismodule, opt.model)(pretrained=opt.isTrain and (not opt.from_scratch),
                                                      base=opt.base)
        net = torch.nn.parallel.DataParallel(net, device_ids = opt.gpu_ids)
        self.net = net.cuda()
        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                 opt.imageSize, opt.imageSize)
        if opt.phase is 'test':
            pass
        else:
            self.criterion = RPNLoss()
            self.optimizer = torch.optim.Adam(self.net.parameters(),
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
        self.save_network(self.net, self.name, label, self.gpu_ids)

    def load(self, label):
        print('loading %s'%label)
        self.load_network(self.net, self.name, label, self.gpu_ids)

    def show_tensorboard_eval(self, num_iter):
        for k, v in self.performance.items():
            self.writer.add_scalar(k, v, num_iter)

    def show_tensorboard(self, num_iter, num_show=4):
        self.writer.add_scalar('loss', self.loss, num_iter)
        num_show = num_show if self.input.size(0) > num_show else self.input.size(0)

        pred_mask, pred_pos = self.prediction
        pred_prob = pred_mask.squeeze(0)[1].detach().cpu().numpy()
        pred_pos = [pp.squeeze(0).squeeze(0).detach().cpu().numpy() for pp in pred_pos.split(1, 1)]
        idx = pred_prob.reshape(-1).argsort()[::-1][:20]
        pred_tx, pred_ty, pred_tw, pred_th = [pp.reshape(-1)[idx] for pp in pred_pos]
        an_xct, an_yct, anw, anh = [aa[idx] for aa in [self.criterion.an_xct.reshape(-1), self.criterion.an_yct.reshape(-1),
                                                       self.criterion.anw.reshape(-1), self.criterion.anh.reshape(-1)]]
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

        plt.cla()
        img = self.input[:num_show]*self.v_std + self.v_mean
        img = img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        img /= 255
        self.ax.imshow(img)
        for i in range(len(pred_xmin)):
            rect = patches.Rectangle(
                (pred_xmin[i], pred_ymin[i]), pred_w[i], pred_h[i], linewidth=2, edgecolor='r',
                facecolor='none')
            self.ax.add_patch(rect)
            # self.ax.annotate(index2name[lbl], (box[1], box[0]), color='w', weight='bold',
            #             fontsize=24, ha='center', va='center')
        img = self.fig2data()
        self.writer.add_image('image', img, num_iter)

    def set_input(self, input, gt_box=None):
        self.input.resize_(input.size()).copy_(input.cuda())
        self.gt_box = gt_box


    def forward(self):
        # print("We are Forwarding !!")
        self.prediction = self.net.forward(self.input)


    def test(self, input, name, WW, HH):
        self.set_input(input)
        with torch.no_grad():
            self.forward()
        outputs = self.prediction.detach().cpu().numpy()

        maximum = outputs.max(1).max(1)
        out_img = outputs / maximum[:, None, None]
        out_img = (out_img*255).astype(np.uint8)
        for ii, msk in enumerate(out_img):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((WW[ii], HH[ii]))
            msk.save('{}/{}.png'.format(self.opt.results_dir, name[ii]), 'PNG')
            np.save('{}/{}'.format(self.opt.results_dir, name[ii]), outputs[ii])


    def backward(self):
        # Combined loss
        self.loss_var = self.criterion(self.prediction, self.gt_box)
        self.loss_var.backward()
        self.loss = self.loss_var.data[0]


    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


    def switch_to_train(self):
        self.net.train()

    def switch_to_eval(self):
        self.net.eval()

