import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random


class LHID:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self):
        print("=> evaluating LHID set...")
        train_dataset = LHIDDataset(dir=os.path.join(self.config.data.data_dir, 'TrainingSet'), val=False, transforms=self.transforms)
        val_dataset = LHIDDataset(dir=os.path.join(self.config.data.data_dir, 'TestingSet', 'TestB'), val=True, transforms=self.transforms)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class LHIDDataset(torch.utils.data.Dataset):
    def __init__(self, dir, val,  transforms):
        super().__init__()

        LHID_dir = dir
        input_names, gt_names, diff_names = [], [], []

        Diff_output = 'Diffusion-branch/results/images/LHID/Merge'
        LHID_inputs = os.path.join(LHID_dir, 'Haze')

        images = [f for f in sorted(listdir(Diff_output)) if isfile(os.path.join(Diff_output, f))]
        # assert len(images) == 8250
        input_names += [os.path.join(LHID_inputs, i) for i in images]
        gt_names += [os.path.join(os.path.join(LHID_dir, 'GT'), i.split('_')[0] + '.jpg') for i in images]
        diff_names += [os.path.join(Diff_output, i) for i in images]
        print(len(input_names))

        x = list(enumerate(input_names))
        random.shuffle(x)
        indices, input_names = zip(*x)
        gt_names = [gt_names[idx] for idx in indices]
        diff_names = [diff_names[idx] for idx in indices]
        self.dir = None

        self.input_names = input_names
        self.gt_names = gt_names
        self.diff_names = diff_names
        self.transforms = transforms

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        diff_name = self.diff_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        diff_img = PIL.Image.open(os.path.join(self.dir, diff_name)) if self.dir else PIL.Image.open(diff_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        # Resizing images to multiples of 16 for whole-image restoration
        wd_new, ht_new = input_img.size
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
        diff_img = diff_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

        return torch.cat([self.transforms(input_img), self.transforms(gt_img), self.transforms(diff_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
