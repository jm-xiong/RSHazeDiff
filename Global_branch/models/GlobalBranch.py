from datetime import datetime
import time
import os
import torch
import torch.backends.cudnn as cudnn
import utils
from utils.validation import validation
from utils.losses import VGGLoss

def charbonnier_loss(restored, target):
    eps = 1e-3
    diff = restored - target
    loss = torch.mean(torch.sqrt((diff * diff) + (eps*eps)))
    return loss


class GlobalBranch(object):
    def __init__(self, model, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = model
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        self.validation = validation
        self.l1_loss = torch.nn.L1Loss()
        self.perception_loss = VGGLoss()


    def load_ddm_ckpt(self, load_path):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, dataset):
        cudnn.benchmark = True
        train_loader, val_loader = dataset.get_loaders()
        epoch_loss = 0
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        old_val_psnr, old_val_ssim, old_val_ciede = self.validation(self.model, val_loader, self.device,
                                                        self.config, save_tag=False)
        print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('***** Epoch: ', epoch, '*****')
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):

                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                input = x[:, :3, :, :]
                gt = x[:, 3:6, :, :]
                diff = x[:, 6:, :, :]
                pre = self.model(input, diff)

                loss = self.l1_loss(pre, gt) + self.perception_loss(pre, gt)

                epoch_loss += loss.item()

                if self.step % 100 == 0:
                    self.model.eval()
                    print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}'.
                                    format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                        epoch, self.config.training.n_epochs,
                                        self.step, self.config.training.snapshot_freq,
                                           (epoch_loss/self.step)))
                    
                    log_path = './log/DHID.txt'
                    # --- Write the training log --- #
                    with open(log_path, 'a') as f:
                        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}, Val_PSNR: {:.2f}, Val_SSIM: {:.4f}, Val_CIEDE:{:.4f}'.
                            format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                   epoch, self.config.training.n_epochs,
                                   self.step, self.config.training.snapshot_freq,
                                   (epoch_loss / self.step), val_psnr, val_ssim, val_ciede))


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                data_start = time.time()

            self.model.eval()
            val_psnr, val_ssim, val_ciede = self.validation(self.model, val_loader, self.device,
                                                                self.config, save_tag=True)
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}, Val_PSNR: {:.2f}, Val_SSIM: {:.4f}, Val_CIEDE:{:.4f}'.
                    format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                           epoch, self.config.training.n_epochs,
                           self.step, self.config.training.snapshot_freq,
                           (epoch_loss / self.step), val_psnr, val_ssim, val_ciede))

            log_path = './log/DHID.txt'
            # --- Write the training log --- #
            with open(log_path, 'a') as f:
                    print(
                        '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}, Val_PSNR: {:.2f}, Val_SSIM: {:.4f}, Val_CIEDE:{:.4f}'.
                        format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                               epoch, self.config.training.n_epochs,
                               self.step, self.config.training.snapshot_freq,
                               (epoch_loss / self.step), val_psnr, val_ssim, val_ciede), file=f)


            utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'step': self.step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'params': self.args,
                    'config': self.config
                }, filename=os.path.join('./checkpoints', self.config.data.dataset,
                                         'epoch_{}'.format(epoch)))


