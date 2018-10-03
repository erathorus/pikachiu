import numpy as np
import torchvision
from PIL import Image


def load(path, batch_size, n_files=3478):
    epoch_count = [1]

    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        files = list(range(n_files))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = Image.open(f'{path}/{i:06d}.jpg')
            image = torchvision.transforms.ColorJitter(hue=0.3)(image)
            image = torchvision.transforms.RandomAffine(20, scale=(0.8, 1.2), fillcolor=(255, 255, 255))(image)
            image = torchvision.transforms.RandomHorizontalFlip()(image)
            image = torchvision.transforms.RandomVerticalFlip(0.2)(image)
            image = np.array(image)
            image = image.transpose((2, 0, 1))
            images[n % batch_size] = image
            if n > 0 and n % batch_size == 0:
                yield images

    return get_epoch
