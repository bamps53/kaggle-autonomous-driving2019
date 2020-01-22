import math

import torch
from torch.optim.optimizer import Optimizer


def get_optimizer(model, config):
    if config.optimizer.params_type == 'encoder_decoder' and config.model.arch == 'CenterNetFPN':
        print('info: set each lr for encoder and decoder.')
        encoder_parms = []
        decoder_parms = []
        for n, p in model.named_parameters():
            if 'fpn.net' in n:
                encoder_parms.append(p)
            else:
                decoder_parms.append(p)
        params = [
            {'params': encoder_parms, 'lr': config.optimizer.params.encoder_lr},
            {'params': decoder_parms, 'lr': config.optimizer.params.decoder_lr}
        ]                
    elif config.optimizer.params_type == 'weight_decay':
        no_decay = ['mean', 'std', 'bias'] + ['.bn%d.' % i for i in range(100)]
        params = [{'params': [], 'weight_decay': config.optimizer.params.weight_decay},
            {'params': [], 'weight_decay': 0.0}]
        for n, p in model.named_parameters():
            if not any(nd in n for nd in no_decay):
                # print("Decay: %s" % n)
                params[0]['params'].append(p)
            else:
                # print("No Decay: %s" % n)
                params[1]['params'].append(p)

    elif config.optimizer.params_type == 'both':
        print('info: set each lr for encoder and decoder and weight decay.')
        no_decay = ['mean', 'std', 'bias'] + ['.bn%d.' % i for i in range(100)]
        encoder_parms = []
        encoder_parms_wd = []
        decoder_parms = []
        decoder_parms_wd = []
        for n, p in model.named_parameters():
            if 'fpn.net' in n:
                if not any(nd in n for nd in no_decay):
                    encoder_parms_wd.append(p)
                else:
                    encoder_parms.append(p)
            else:
                if not any(nd in n for nd in no_decay):
                    decoder_parms_wd.append(p)
                else:
                    decoder_parms.append(p)
        params = [
            {'params': encoder_parms, 'lr': config.optimizer.params.encoder_lr},
            {'params': encoder_parms_wd, 'lr': config.optimizer.params.encoder_lr, 'weight_decay': config.optimizer.params.weight_decay},
            {'params': decoder_parms, 'lr': config.optimizer.params.decoder_lr},
            {'params': decoder_parms_wd, 'lr': config.optimizer.params.decoder_lr, 'weight_decay': config.optimizer.params.weight_decay}
        ]        



    if config.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(params, config.optimizer.params.encoder_lr)
    elif config.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(params, config.optimizer.params.encoder_lr)
    return optimizer



