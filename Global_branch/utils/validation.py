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



def validation(net, val_data_loader, device, config, save_tag=False):

    psnr_list = []
    ssim_list = []
    ciede_list = []

    for i, (x, y) in enumerate(val_data_loader):

        with torch.no_grad():
            input = x[:, :3, :, :].to(device)
            gt = x[:, 3:6, :, :].to(device)
            diff_img = x[:, 6:, :, :].to(device)
            pred_image = net(input, diff_img)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(ssim(pred_image, gt))

        # --- Calculate the average ciede2000 --- #
        ciede_list.extend(calc_ciede2000(pred_image, gt))

        # --- Save image --- #
        if save_tag:
            # print()
            save_image(pred_image, y, config)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    avr_ciede = sum(ciede_list) / len(ciede_list)
    return avr_psnr, avr_ssim, avr_ciede


def save_image(pred_image, image_name, config):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)
    
    for ind in range(batch_num):
        # print(image_name[ind])
        utils.save_image(pred_image_images[ind], './results/{}/{}.jpg'.format(config.data.dataset, image_name[ind]))
        # utils.save_image(pred_image_images[ind], './Ablation/fusion_results/{}/fusion3/{}.jpg'.format(config.data.dataset, image_name[ind]))


def print_log(epoch, num_epochs, step, snapshot_freq, val_psnr, val_ssim, val_ciede, config):
    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Val_PSNR: {:.2f}, Val_SSIM: {:.4f}, Val_CIEDE:{:.4f}'.
          format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                 epoch, num_epochs,
                 step, snapshot_freq,
                 val_psnr, val_ssim, val_ciede))

    log_path = './training_log/{}_log.txt'.format(config.data.dataset)

    # --- Write the training log --- #
    with open(log_path, 'a') as f:
        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Val_PSNR: {:.2f}, Val_SSIM: {:.4f}, Val_CIEDE:{:.4f}'.
              format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                     epoch, num_epochs,
                     step, snapshot_freq,
                     val_psnr, val_ssim, val_ciede), file=f)



def adjust_learning_rate(optimizer, epoch,  lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 100

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            #print(file_)
            #print(result)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

# initial_epoch is keep training start epoch(train.py print is epoch+1)