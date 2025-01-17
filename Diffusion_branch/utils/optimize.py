import torch
import torch.optim as optim

from warmup_scheduler import GradualWarmupScheduler


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
        warmup_epochs = 1
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.n_epochs - warmup_epochs,
                                                                      eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
        scheduler.step()
    elif config.optim.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
        warmup_epochs = 1
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                      T_max=config.training.n_epochs - warmup_epochs,
                                                                      eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
    elif config.optim.optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
        warmup_epochs = 1
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                      T_max=config.training.n_epochs - warmup_epochs,
                                                                      eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.optim.step_size,
    #                                       gamma=config.optim.gamma, last_epoch=-1)

    return optimizer, scheduler
