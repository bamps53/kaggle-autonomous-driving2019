import math

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts


def get_scheduler(optimizer, config):
    if config.scheduler.name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    elif config.scheduler.name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, config.scheduler.params.t_max, eta_min=1e-6,
                                      last_epoch=-1)
    elif config.scheduler.name == 'cosine_warmup':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler.params.t_max, eta_min=1e-6,
                                                last_epoch=-1)
    else:
        scheduler = None
    return scheduler
