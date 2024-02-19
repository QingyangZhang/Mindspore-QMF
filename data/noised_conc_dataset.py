import os.path
import random

import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from torchvision.datasets.folder import make_dataset
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

from torchvision.transforms import functional as F
import copy
import numpy as np

from torchvision.transforms import ToPILImage

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def random_noise(nc, width, height, index):
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
    #seed_torch(index)
    #seed = torch.initial_seed()
    st = torch.random.get_rng_state()
    torch.manual_seed(index)
    img = torch.rand(nc, width, height)
    img = ToPILImage()(img)
    torch.random.set_rng_state(st)

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
    img = torch.ones(nc, width, height)*label/10
    img = ToPILImage()(img)
    return img

class noisedConcDataset:

    def __init__(self, cfg, data_dir=None, transform=None, labeled=True, rgb_threshold=0.5, depth_threshold=0.5):
        self.cfg = cfg
        self.rgb_transform = transform
        self.depth_transform = transform
        self.normal_transform = transform
        self.data_dir = data_dir
        self.labeled = labeled
        self.rgb_threshold = rgb_threshold
        self.depth_threshold = depth_threshold

        print(self.rgb_threshold)
        print(self.depth_threshold)

        self.classes, self.class_to_idx = find_classes(self.data_dir)
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.imgs = make_dataset(self.data_dir, self.class_to_idx, 'png')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc_1 = Image.open(img_path).convert('RGB')
        #AB_conc_2 = random_noise(3, AB_conc_1.size[1], AB_conc_1.size[0], index)
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
            #A_2 = AB_conc_2.crop((0, 0, w2, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
            #B_2 = AB_conc_2.crop((w2, 0, w, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)            
        else:
            A_1 = AB_conc_1.crop((0, 0, w2, h))
            B_1 = AB_conc_1.crop((w2, 0, w, h))
            #A_2 = AB_conc_2.crop((0, 0, w2, h))
            #B_2 = AB_conc_2.crop((w2, 0, w, h))    


        if self.labeled:
            #if not (pa and pb):
            #    print(index%10)
            #    print(self.rgb_threshold*10-0.1)
            #sample = {'A': A_1 if pa else A_2, 'B': B_1 if pb else B_2, 'img_name': img_name, 'label': label, 'ID_1': 1 if pa else 0, 'ID_2': 1 if pb else 0}
            sample = {'A': A_1, 'B': B_1, 'img_name': img_name, 'label': label}
        else:
            sample = {'A': A_1, 'B': B_1, 'img_name': img_name}

        if self.rgb_transform:
            pa = ((index%10) < (self.rgb_threshold*10 - 0.1))
            if pa:
                sample['A'] = self.rgb_transform(sample['A'])
            else:
                sample['A'] = self.normal_transform(sample['A'])
        if self.depth_transform:
            pb = ((index%10) < (self.depth_threshold*10 - 0.1))
            if pb:
                sample['B'] = self.depth_transform(sample['B'])
            else:
                sample['B'] = self.normal_transform(sample['B'])

        return sample


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        if self.padding > 0:
            A = F.pad(A, self.padding)
            B = F.pad(B, self.padding)

        # pad the width if needed
        if self.pad_if_needed and A.size[0] < self.size[1]:
            A = F.pad(A, (int((1 + self.size[1] - A.size[0]) / 2), 0))
            B = F.pad(B, (int((1 + self.size[1] - B.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and A.size[1] < self.size[0]:
            A = F.pad(A, (0, int((1 + self.size[0] - A.size[1]) / 2)))
            B = F.pad(B, (0, int((1 + self.size[0] - B.size[1]) / 2)))

        i, j, h, w = self.get_params(A, self.size)
        sample['A'] = F.crop(A, i, j, h, w)
        sample['B'] = F.crop(B, i, j, h, w)

        # _i, _j, _h, _w = self.get_params(A, self.size)
        # sample['A'] = F.crop(A, i, j, h, w)
        # sample['B'] = F.crop(B, _i, _j, _h, _w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.center_crop(A, self.size)
        sample['B'] = F.center_crop(B, self.size)
        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        if random.random() > 0.5:
            A = F.hflip(A)
            B = F.hflip(B)

        sample['A'] = A
        sample['B'] = B

        return sample


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


class Resize(transforms.Resize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        h = self.size[0]
        w = self.size[1]

        sample['A'] = F.resize(A, (h, w))
        sample['B'] = F.resize(B, (h, w))

        return sample


class ToTensor(object):
    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        # if isinstance(sample, dict):
        #     for key, value in sample:
        #         _list = sample[key]
        #         sample[key] = [F.to_tensor(item) for item in _list]

        sample['A'] = F.to_tensor(A)
        sample['B'] = F.to_tensor(B)

        return sample


class Normalize(transforms.Normalize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.normalize(A, self.mean, self.std)
        sample['B'] = F.normalize(B, self.mean, self.std)

        return sample


class Lambda(transforms.Lambda):

    def __call__(self, sample):
        return self.lambd(sample)
