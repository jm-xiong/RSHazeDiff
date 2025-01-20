import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torch import nn

import models
import datasets
import torchvision.utils as utils
from models import GlobalNet_Fusion, GlobalBranch
import glob
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure
import cv2
from skimage import img_as_float32
import skimage
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import deltaE_ciede2000 as compare_ciede

import pdb
from math import exp
from torch.autograd import Variable


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def ssim(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return [ssim_map.mean()]
    else:
        return [ssim_map.mean(1).mean(1).mean(1)]

def calc_psnr(im1, im2):
    #tensor转numpy
    im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    ans = [compare_psnr(im1_y, im2_y)]
    return ans

def calc_ssim(im1, im2):
    im1 = im1[0].view(im1.shape[2],im1.shape[3],3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2],im2.shape[3],3).detach().cpu().numpy()

    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    ans = [compare_ssim(im1_y, im2_y)]
    return ans

def calc_ciede2000(im1, im2):
    #tensor转numpy
    im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

    #numpy数据类型转换为float32
    im1 =img_as_float32(im1)
    im2 = img_as_float32(im2)
    # im1 = im1.astype(np.float32)
    # im2 = im2.astype(np.float32)

    in_lab = cv2.cvtColor(im1, cv2.COLOR_RGB2Lab)
    gt_lab = cv2.cvtColor(im2, cv2.COLOR_RGB2Lab)
    ans = [np.average(compare_ciede(gt_lab, in_lab))]
    return ans


def to_ssim_skimage(pred_image, gt):
    pred_image_list = torch.split(pred_image, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_image_list_np = [pred_image_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(pred_image_list))]
    ssim_list = [measure.compare_ssim(pred_image_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(pred_image_list))]

    return ssim_list


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Testing Global Branch Models')
    parser.add_argument("--config", default='DHID.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--test_set", type=str, default='results/images/',
                        help="restoration test set options: ['DHID','ERICE']")
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=42, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders()

    # create model
    print("=> creating global model with wrapper...")
    model = GlobalNet_Fusion()
    Branch = GlobalBranch(model, args, config)
    if os.path.isfile(args.resume):
        Branch.load_ddm_ckpt(args.resume)
        model.eval()
    else:
        print('Pre-trained global model path is missing!')

    psnr_list, ssim_list, ciede_list = [], [], []
    for i, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            start_time = time.time()
            input = x[:, :3, :, :].to(device)
            gt = x[:, 3:6, :, :].to(device)
            diff_img = x[:, 6:, :, :].to(device)

            pred_image = model(input, diff_img)
            torch.cuda.synchronize()
            inference_time = time.time() - start_time
            print(inference_time)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(ssim(pred_image, gt))

        # --- Calculate the average ciede2000 --- #
        ciede_list.extend(calc_ciede2000(pred_image, gt))

        pred_image_images = torch.split(pred_image, 1, dim=0)
        batch_num = len(pred_image_images)

        for ind in range(batch_num):
            utils.save_image(pred_image_images[ind],
                                 '{}{}/{}.jpg'.format(args.image_folder, config.data.dataset, y[ind]))

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    avr_ciede = sum(ciede_list) / len(ciede_list)
    print('Current Metrics: \nPSNR: {:.3f}, \nSSIM: {:.5f}, \nCIEDE2000: {:.5f}'.format(avr_psnr, avr_ssim, avr_ciede))

if __name__ == '__main__':
    main()
