import cv2
import numpy as np
import random
from PIL import Image
import pdb


def rectangle_mask(image_height=256, image_width=256, min_hole_size=10, max_hole_size=128):
    mask = np.zeros((image_height, image_width))
    hole_size = random.randint(min_hole_size, max_hole_size)
    x = random.randint(0, image_width-hole_size-1)
    y = random.randint(0, image_height-hole_size-1)
    mask[x:x+hole_size, y:y+hole_size] = 1
    return mask


def stroke_mask(image_height=256, image_width=256, max_vertex=5, max_mask=5):
    max_angle = np.pi
    max_length = min(image_height, image_width)*0.5
    max_brush_width = max(1, int(min(image_height, image_width)*0.2))
    min_brush_width = max(1, int(min(image_height, image_width)*0.05))

    mask = np.zeros((image_height, image_width))
    for k in range(random.randint(1, max_mask)):
        num_vertex = random.randint(1, max_vertex)
        start_x = random.randint(0, image_width-1)
        start_y = random.randint(0, image_height-1)
        for i in range(num_vertex):
            angle = random.uniform(0, max_angle)
            if i % 2 == 0:
                angle = 2*np.pi - angle
            length = random.uniform(0, max_length)
            brush_width = random.randint(min_brush_width, max_brush_width)
            end_x = min(int(start_x + length * np.cos(angle)), image_width)
            end_y = min(int(start_y + length * np.sin(angle)), image_height)
            mask = cv2.line(mask, (start_x, start_y), (end_x, end_y), color=1, thickness=brush_width)
            start_x, start_y = end_x, end_y
            mask = cv2.circle(mask, (start_x, start_y), int(brush_width/2), 1)
        if random.randint(0, 1):
            mask = mask[:, ::-1].copy()
        if random.randint(0, 1):
            mask = mask[::-1, :].copy()
    return mask


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    input_dir = '/home/zeng/data/datasets/imagenet10val'
    output_dir = '/home/zeng/data/datasets/imagenet10mask'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    names = os.listdir(input_dir)
    for name in tqdm(names):
        img = Image.open('{}/{}'.format(input_dir, name))
        w, h = img.size
        max_hole = min(w/2, h/2)
        mask = stroke_mask(h, w) if random.randint(0, 1) else rectangle_mask(h, w, max_hole_size=max_hole, min_hole_size=max_hole/2)
        mask = (mask*255).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask.save('%s/%s.png'%(output_dir, '.'.join(name.split('.')[:-1])))
    # output_dir = '/Users/zeng/gitmodel/mask'
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # for i in range(500):
    #     mask = stroke_mask(128, 128) if random.randint(0, 1) else rectangle_mask(128, 128, max_hole_size=64, min_hole_size=32)
    #     mask = (mask*255).astype(np.uint8)
    #     mask = Image.fromarray(mask)
    #     mask.save('%s/%s.png'%(output_dir, i))

