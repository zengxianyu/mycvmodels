from __future__ import print_function
import numpy as np
import os
import PIL.Image as Image
import pdb
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
eps = np.finfo(float).eps


def eva_one(param):
    input_name, gt_name, mask_name = param
    mask = np.array(Image.open(mask_name))
    mask[mask!=0] = 1
    recon = np.array(Image.open(input_name))
    gt = np.array(Image.open(gt_name))

    mse = ((recon-gt)**2).mean()
    psnr = 10 * np.log10(255.0*255.0 / mse)
    return psnr


def compute_psnr(input_dir, gt_dir, mask_dir):
    filelist_msk = os.listdir(mask_dir)
    msk_format = filelist_msk[0].split('.')[-1]
    filelist_msk = ['.'.join(f.split('.')[:-1]) for f in filelist_msk]

    filelist_gt = os.listdir(gt_dir)
    gt_format = filelist_gt[0].split('.')[-1]
    filelist_gt = ['.'.join(f.split('.')[:-1]) for f in filelist_gt]

    filelist_pred = os.listdir(input_dir)
    pred_format = filelist_pred[0].split('.')[-1]
    filelist_pred = ['.'.join(f.split('.')[:-1]) for f in filelist_pred]

    inputlist = [os.path.join(input_dir, '.'.join([_name, pred_format])) for _name in filelist_pred]
    gtlist = [os.path.join(gt_dir, '.'.join([_name, gt_format])) for _name in filelist_gt]
    masklist = [os.path.join(mask_dir, '.'.join([_name, msk_format])) for _name in filelist_msk]

    pool = Pool(4)
    results = pool.map(eva_one, zip(inputlist, gtlist, masklist))
    return np.array(results).mean()


if __name__ == '__main__':
    # fm, mae, _, _ = fm_and_mae('/home/crow/WSLfiles/WTCW_woSeg_densenet169/results',
    #                      '/home/crow/data/datasets/saliency_Dataset/ECSSD/masks')
    # print(fm)
    # print(mae)
    print_table()
    # draw_curves()



