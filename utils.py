import glob
import os
import os.path as path

import numpy as np
import torchvision
from PIL import Image
from scipy.misc import imsave


def rename_images(root):
    files = sorted(glob.glob(f'{root}/*.png') + glob.glob(f'{root}/*.jpg'))
    for i, file in enumerate(files, 0):
        ext = path.splitext(file)[1]
        os.rename(file, f'{root}/1{i:06d}{ext}')

    files = sorted(glob.glob(f'{root}/*.png') + glob.glob(f'{root}/*.jpg'))
    for i, file in enumerate(files, 0):
        ext = path.splitext(file)[1]
        os.rename(file, f'{root}/{i:06d}{ext}')


def to_rgb(root, rgb_root):
    files = sorted(glob.glob(f'{root}/*.png') + glob.glob(f'{root}/*.jpg'))
    for i, file in enumerate(files, 0):
        img = Image.open(file)
        img.load()
        if img.mode == 'P':
            img = img.convert('RGBA')
        if img.mode == 'RGBA':
            rgb = Image.new('RGB', img.size, (255, 255, 255))
            rgb.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        else:
            rgb = img
        rgb.save(f'{rgb_root}/{i:06d}.jpg', 'JPEG')


def resize_images(root, resize_root, size):
    files = sorted(glob.glob(f'{root}/*.jpg'))
    for file in files:
        img = Image.open(file)
        img = torchvision.transforms.Resize(size, interpolation=Image.BICUBIC)(img)
        filename = path.basename(file)
        img.save(f'{resize_root}/{filename}')


def save_images(tensor, path):
    n_samples = tensor.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows > 0:
        rows -= 1

    nh, nw = rows, n_samples // rows

    tensor = tensor.transpose(0, 2, 3, 1)
    h, w = tensor[0].shape[:2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(tensor):
        j = n // nw
        i = n % nw
        img[j * h: (j + 1) * h, i * w: (i + 1) * w] = x

    imsave(path, img)
