import os
import pdb
from PIL import Image
from tqdm import tqdm


input_dir = '/home/crow/data/datasets/depth_dataset/PBR/images-mlt'
output_dir = '/home/crow/data/datasets/depth_dataset/PBR/images-jpg'


folders = os.listdir(input_dir)

for folder in tqdm(folders):
    if not os.path.exists('{}/{}'.format(output_dir, folder)):
        os.mkdir('{}/{}'.format(output_dir, folder))
    names = os.listdir('{}/{}'.format(input_dir, folder))
    for name in names:
        img = Image.open('{}/{}/{}'.format(input_dir, folder, name))
        img.save('{}/{}/{}'.format(output_dir, folder, name[:-4]+'.jpg'))
