import os
from PIL import Image
import numpy as np
import pdb
import matplotlib.pyplot as plt
import cv2
from multiprocessing import Pool


def eval_one(params):
    results_name, gt_name = params
    if gt_name.split('.')[-1] == 'npy':
        gt = np.load(gt_name)
    elif gt_name.split('.')[-1] == 'png':
        gt = Image.open(gt_name)
        gt = np.array(gt, dtype=np.float) / 1000.0  # to meter
    pre = np.load(results_name)
    pre[pre<=0] = 1e-8
    # pdb.set_trace()
    # pdb.set_trace()
    # pre = Image.open(results_name)
    # pre = np.array(pre, dtype=np.float)
    # gt_small = cv2.resize(gt, (pre.shape[1], pre.shape[0]))
    # pre = piecewise_fit(pre, gt_small)
    pre = cv2.resize(pre, (gt.shape[1], gt.shape[0]))
    valid_mask = (gt>0)
    gt = gt[valid_mask]
    pre = pre[valid_mask]
    count = len(gt)
    rel = (np.abs(pre-gt)/gt).sum()
    # print(rel)
    log_pre = np.log10(pre)
    log_gt = np.log10(gt)
    log_gt = log_gt[~np.isnan(log_pre)]
    count_log = (~np.isnan(log_pre)).sum()
    log_pre = log_pre[~np.isnan(log_pre)]
    log10 = np.abs(log_pre - log_gt).sum()
    # print(log10)
    rms = ((pre-gt)**2).sum()
    # print(rms)
    return (rel, log10, rms, count, count_log)


def rel_log10_rms(results_dir, gt_dir):
    names_re = os.listdir(results_dir)
    if os.path.isdir('{}/{}'.format(results_dir, names_re[0])):
        names_re = ['{}/{}'.format(name, fname)
                 for name in names_re for fname in os.listdir('{}/{}'.format(results_dir, name))]
    names_re = ['.'.join(name.split('.')[:-1]) for name in names_re]

    names_gt = os.listdir(gt_dir)
    if os.path.isdir('{}/{}'.format(gt_dir, names_gt[0])):
        names_gt = ['{}/{}'.format(name, fname)
                    for name in names_gt for fname in os.listdir('{}/{}'.format(gt_dir, name))]
    gt_format = names_gt[0].split('.')[-1]
    names_gt = ['.'.join(name.split('.')[:-1]) for name in names_gt]
    names = list(set(names_gt)&set(names_re))

    results_names = ['{}/{}.npy'.format(results_dir, name) for name in names]
    gt_names = ['{}/{}.{}'.format(gt_dir, name, gt_format) for name in names]
    # sb = eval_one(results_names[:1]+gt_names[:1])
    # pdb.set_trace()
    pool = Pool(4)
    results = pool.map(eval_one, zip(results_names, gt_names))

    rel, log10, rms, count, count_log = list(map(list, zip(*results)))
    count = np.array(count).sum()
    count_log = np.array(count_log).sum()
    rel = np.array(rel).sum() / count
    log10 = np.array(log10).sum() / count_log
    rms = np.sqrt(np.array(rms).sum() / count)
    return rel, log10, rms
    # print(rel)
    # print(log10)
    # print(rms)


if __name__ == "__main__":
    gt_dir = '/home/zeng/data/datasets/depth_dataset/PBR/depth'
    results_dir = '/home/zeng/segggFiles/pbr_grad/results'
    # gt_dir = '/home/zeng/data/datasets/depth_dataset/NYU2/data/depth'
    # results_dir = '/home/zeng/segggFiles/nyu/results'

    rel, log10, rms = rel_log10_rms(results_dir, gt_dir)
    print('rel={}, log10={}, rms={}'.format(rel, log10, rms))


