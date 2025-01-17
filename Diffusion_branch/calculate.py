from skimage import io
import os
import cv2
import numpy as np
from skimage import img_as_float32
import pyiqa

from utils.metrics import calculate_psnr, calculate_ssim
from skimage.color import deltaE_ciede2000

# Sample script to calculate PSNR and SSIM metrics from saved images in two directories
# using calculate_psnr and calculate_ssim functions from: https://github.com/JingyunLiang/SwinIR

lpips_metric = pyiqa.create_metric('lpips').cuda()
fid_metric = pyiqa.create_metric('fid')

gt_path = '/home/xjm/Project/jmxiong/data/Remote sense/ERICE/Test/GT'
results_path = 'results/images/ERICE'

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))
assert len(imgsName) == len(gtsName)

cumulative_psnr, cumulative_ssim, cumulative_ciede, cumulative_lpips = 0, 0, 0, 0

for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
    cur_psnr = calculate_psnr(res, gt)
    cur_ssim = calculate_ssim(res, gt)

    im1 = img_as_float32(io.imread(os.path.join(gt_path, gtsName[i])))
    im2 = img_as_float32(io.imread(os.path.join(results_path, imgsName[i])))
    im1_lab = cv2.cvtColor(im1, cv2.COLOR_RGB2Lab)
    im2_lab = cv2.cvtColor(im2, cv2.COLOR_RGB2Lab)
    cur_ciede = np.average(deltaE_ciede2000(im1_lab, im2_lab))

    cur_lpips = lpips_metric(os.path.join(results_path, imgsName[i]), os.path.join(gt_path, gtsName[i]))

    print('PSNR is %.2f, SSIM is %.3f, CIEDE is %.3f and LPIPS is %.3f' % (cur_psnr, cur_ssim, cur_ciede, cur_lpips))

    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
    cumulative_ciede += cur_ciede
    cumulative_lpips += cur_lpips

fid_score = fid_metric(results_path, gt_path)

print('Testing set, PSNR is %.2f, SSIM is %.3f and CIEDE is %.3f, LPIPS is %.3f, and FID is %.2f' % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName), cumulative_ciede / len(imgsName), cumulative_lpips/ len(imgsName),fid_score))
print(results_path)
