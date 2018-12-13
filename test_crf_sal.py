import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import PIL.Image as Image
import multiprocessing
from evaluate_sal import fm_and_mae


sal_set = 'OMRON'
img_root = '../data/datasets/saliency_Dataset/%s/images'%sal_set
prob_root = '../data/datasets/saliency_Dataset/results/%s-Sal/Ours-Seg'%sal_set
output_root = '../data/datasets/saliency_Dataset/results/%s-Sal/Ours-Seg-crf'%sal_set

# img_root = '/media/zeng/1656E46156E442DB/Users/Administrator/Desktop/DUT-train/images'
# prob_root = '/media/zeng/1656E46156E442DB/Users/Administrator/Desktop/DUT-AVG'
# output_root = '/media/zeng/1656E46156E442DB/Users/Administrator/Desktop/DUT-AVG-CRF'

if not os.path.exists(output_root):
    os.mkdir(output_root)
# if not os.path.exists(output_root+'_bin'):
#     os.mkdir(output_root+'_bin')

files = os.listdir(prob_root)


# for img_name in files:


def myfunc(img_name):
    img = Image.open(os.path.join(img_root, img_name[:-4]+'.jpg'))
    img = np.array(img, dtype=np.uint8)
    sh = img.shape
    H, W = sh[0], sh[1]
    if len(sh) < 3:
        img = np.stack((img, img, img), 2)
    probs = Image.open(os.path.join(prob_root, img_name[:-4]+'.png'))
    probs = probs.resize((W, H))
    probs = np.array(probs)
    probs = probs.astype(np.float)/255.0
    probs = np.concatenate((1-probs[None, ...], probs[None, ...]), 0)
    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)

    # get unary potentials (neg log probability)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run five inference steps.
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    BMAP = np.argmax(Q, axis=0).reshape((H, W))
    BMAP = (BMAP*255).astype(np.uint8)
    MAP = np.array(Q)[1].reshape((H, W))
    MAP = (MAP*255).astype(np.uint8)
    msk = Image.fromarray(MAP)
    msk.save(os.path.join(output_root, img_name), 'png')
    # msk = Image.fromarray(BMAP)
    # msk.save(os.path.join(output_root+'-bin', img_name), 'png')


if __name__ == '__main__':
    # myfunc(files[0])
    print('start crf')
    pool = multiprocessing.Pool(processes=8)
    pool.map(myfunc, files)
    pool.close()
    pool.join()
    print('done')
    fm, mae, _, _ = fm_and_mae(output_root, '../data/datasets/saliency_Dataset/%s/masks'%sal_set)
    print(fm)
    print(mae)
