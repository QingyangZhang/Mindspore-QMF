import os.path
import random

from PIL import Image
from PIL import ImageFile
import sys
import os
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

ImageFile.LOAD_TRUNCATED_IMAGES = True

import copy
import numpy as np

from mindspore.dataset.vision import ToPIL


def has_file_allowed_extension(fname, extensions):
    """Checks if a file is an allowed extension.

    Args:
        fname (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return fname.lower().endswith(extensions)


def make_dataset(directory, class_to_index, extensions=None):
    instances = []
    directory = os.path.expanduser(directory)
    for target_class in sorted(class_to_index.keys()):
        class_index = class_to_index[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir)):
            for fname in sorted(fnames):
                if extensions is None or has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)
    return instances


def random_noise(nc, width, height):
    '''Generator a random noise image from tensor.

    If nc is 1, the Grayscale image will be created.
    If nc is 3, the RGB image will be generated.

    Args:
        nc (int): (1 or 3) number of channels.
        width (int): width of output_bert image.
        height (int): height of output_bert image.
    Returns:
        PIL Image.
    '''
    img = P.rand(height, width, nc)
    img = img.asnumpy().astype(np.uint8)
    img = ToPIL()(img)
    return img

def ood_image(nc, width, height, label):
    '''Generator a random noise image from tensor.

    If nc is 1, the Grayscale image will be created.
    If nc is 3, the RGB image will be generated.

    Args:
        nc (int): (1 or 3) number of channels.
        width (int): width of output_bert image.
        height (int): height of output_bert image.
    Returns:
        PIL Image.
    '''
    img = P.ones(nc, width, height)*label/10
    img = ToPIL()(img)
    return img


class oodConcDataset:

    def __init__(self, cfg, data_dir=None, transform=None, labeled=True, rgb_threshold=0.2, depth_threshold=0.2):
        self.cfg = cfg
        self.transform = transform
        self.data_dir = data_dir
        self.labeled = labeled
        self.rgb_threshold = rgb_threshold
        self.depth_threshold = depth_threshold

        self.classes, self.class_to_idx = find_classes(self.data_dir)
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.imgs = make_dataset(self.data_dir, self.class_to_idx, 'png')
        # FIXME: remove this. just for debugging
        self.imgs = self.imgs[:64]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # print(index, end=', ')
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc_1 = Image.open(img_path).convert('RGB')
        AB_conc_2 = random_noise(3, AB_conc_1.size[1], AB_conc_1.size[0])
        #AB_conc_2 = ood_image(3, AB_conc_1.size[1], AB_conc_1.size[0], label)
        #AB_conc_2 = Image.open(ood_img_path).convert('RGB')

        #print(AB_conc_1.size)
        #print(AB_conc_2.size)
        #exit()


        # split RGB and Depth as A and B
        w, h = AB_conc_1.size
        w2 = int(w / 2)
        if w2 > self.cfg.FINE_SIZE:
            A_1 = AB_conc_1.crop((0, 0, w2, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
            B_1 = AB_conc_1.crop((w2, 0, w, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
            A_2 = AB_conc_2.crop((0, 0, w2, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
            B_2 = AB_conc_2.crop((w2, 0, w, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
        else:
            A_1 = AB_conc_1.crop((0, 0, w2, h))
            B_1 = AB_conc_1.crop((w2, 0, w, h))
            A_2 = AB_conc_2.crop((0, 0, w2, h))
            B_2 = AB_conc_2.crop((w2, 0, w, h))

        if self.labeled:
            pa = np.random.uniform()>self.rgb_threshold
            pb = np.random.uniform()>self.depth_threshold
            sample = {'A': A_1 if pa else A_2, 'B': B_1 if pb else B_2, 'img_name': img_name, 'label': label, 'ID_1': 1 if pa else 0, 'ID_2': 1 if pb else 0, 'idx': index}
        else:
            sample = {'A': A_1, 'B': B_1, 'img_name': img_name_1}

        if self.transform:
            sample['A'] = self.transform(sample['A'])
            sample['B'] = self.transform(sample['B'])

        return sample['A'], sample['B'], sample['label']


def find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
