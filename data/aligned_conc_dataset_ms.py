import os.path
import random

from PIL import Image
from PIL import ImageFile
import sys
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter

ImageFile.LOAD_TRUNCATED_IMAGES = True

import copy
from data.ood_conc_dataset_ms import make_dataset


class AlignedConcDataset:

    def __init__(self, cfg, data_dir=None, transform=None, labeled=True):
        self.cfg = cfg
        self.transform = transform
        self.data_dir = data_dir
        self.labeled = labeled

        self.classes, self.class_to_idx = find_classes(self.data_dir)
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.imgs = make_dataset(self.data_dir, self.class_to_idx, 'png')
        # FIXME: remove this. just for debugging
        self.imgs = self.imgs[:64]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        if w2 > self.cfg.FINE_SIZE:
            A = AB_conc.crop((0, 0, w2, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
            B = AB_conc.crop((w2, 0, w, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
        else:
            A = AB_conc.crop((0, 0, w2, h))
            B = AB_conc.crop((w2, 0, w, h))

        if self.labeled:
            sample = {'A': A, 'B': B, 'img_name': img_name, 'label': label}
        else:
            sample = {'A': A, 'B': B, 'img_name': img_name}

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
