import pretrainedmodels
import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
from config.base import load_config
from .centernet import CenterNetFPN
from catalyst.dl.utils import load_checkpoint

class MultiModels:
    def __init__(self, models, tta=True):
        self.models = models
        self.tta = True

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
            if self.tta:
                x = x.flip(-1)
                y_hat = m(x).flip(-1)
                y_hat[:,[2,3,4],:,:] = -y_hat[:,[2,3,4],:,:] # multiply -1 for ['x', 'pitch_sin', 'roll']
                res.append(y_hat)
        res = torch.stack(res)
        return torch.mean(res, dim=0)

def load_model(config_path, checkpoint_path, fold=0):
    config = load_config(config_path)
    if not 'fold' in config.work_dir:
        config.work_dir = config.work_dir + '_fold{}'.format(fold)

    model = CenterNetFPN(
        slug=config.model.encoder,
        num_classes=len(config.data.features),
        )

    model.to(config.device)
    model.eval()

    checkpoint = load_checkpoint(checkpoint_path)
    print('load model from {}'.format(checkpoint_path))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model