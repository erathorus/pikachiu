import numpy as np
from scipy.misc import imsave


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
