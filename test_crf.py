import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import PIL.Image as Image
import multiprocessing
from datasets.voc import palette
import pdb
from evaluate import evaluate


img_root = '../data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/JPEGImages'
# img_root = '../data/datasets/ILSVRC14VOC/images'
map_root = '../WSLfiles/WTCW_woSal_densenet169/train_results_prob'
output_root = '../WSLfiles/WTCW_woSal_densenet169/train_crf'

if not os.path.exists(output_root):
    os.mkdir(output_root)

files = os.listdir(map_root)
# for img_name in files:


def myfunc(img_name):
    if not img_name.endswith('.npz.npy'):
        return
    img = Image.open(os.path.join(img_root, img_name[:-8]+'.jpg')).convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img, dtype=np.uint8)
    probs = np.load(os.path.join(map_root, img_name))

    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 21)

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
    MAP = np.argmax(Q, axis=0).reshape((256, 256))
    # bb = np.array(Q)[0].reshape(img.shape[:2])
    msk = Image.fromarray(MAP.astype(np.uint8))

    msk = msk.convert('P')
    msk.putpalette(palette)
    msk.save(os.path.join(output_root, img_name[:-8] + '.png'), 'png')


if __name__ == '__main__':
    # myfunc(files[0])
    print('start crf')
    pool = multiprocessing.Pool(processes=4)
    pool.map(myfunc, files)
    pool.close()
    pool.join()
    print('done')
    evaluate(output_root, '../data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClassAug', 21)
    # evaluate(output_root, '../data/datasets/segmentation_Dataset/VOCdevkit/VOC2012/SegmentationClass', 21)
