import numpy as np
import torch
import torch.nn as nn
from osgeo import gdal

import utils
import torchvision
import os
import time


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)



def save_tif(dehaze, reference_file, filename):
    dehaze = dehaze.numpy().squeeze(0)
    dehaze = (dehaze / 255) * 65535
    dehaze = dehaze.astype(np.uint16)

    bands, height, width = dehaze.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, width, height, bands,
                            gdal.GDT_UInt16)
    ds = gdal.Open(reference_file)
    dataset.SetGeoTransform(ds.GetGeoTransform())
    dataset.SetProjection(ds.GetProjection())
    if bands == 1:
        dataset.GetRasterBand(1).WriteArray(dehaze)
    else:
        for i in range(bands):
            dataset.GetRasterBand(i + 1).WriteArray(dehaze[i])


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, r=None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset)

        avg_time = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                start_time = time.time()
                print("{} starting processing from image {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),y))
                x = x.flatten(start_dim=0, end_dim=1).float() if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device).float()
                x_output = self.diffusive_restoration(x_cond, r=r)
                torch.cuda.synchronize()
                x_output = inverse_data_transform(x_output)
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}" + '.jpg'))
                avg_time += time.time() - start_time
                print(time.time() - start_time)
            print('average time:', avg_time/len(val_loader))

    def diffusive_restoration(self, x_cond, gt=None, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, gt, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
