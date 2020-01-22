import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    # dataset
    c.data = edict()
    c.data.name = 'DefaultDataset'
    c.data.num_classes = 3
    c.data.features = ['pitch_cos', 'pitch_sin', 'z']
    c.data.sample_submission_path = '../input/pku-autonomous-driving/sample_submission.csv'
    c.data.test_dir = '../input/pku-autonomous-driving/test_images'
    c.data.train_df_path = '../input/pku-autonomous-driving/train.csv'
    c.data.train_dir = '../input/pku-autonomous-driving/train_images'
    c.data.params = edict()
    c.data.height = 2710
    c.data.width = 3384
    c.data.model_scale = 4
    c.data.pseudo_path = False

    # model
    c.model = edict()
    c.model.arch = 'CenterNetFPN'
    c.model.encoder = 'r34'
    c.model.params = edict()
    c.model.params.upconv = False
    c.model.checkpoint_path = None

    # train
    c.train = edict()
    c.train.batch_size = 16
    c.train.num_epochs = 50
    c.train.main_metric = 'loss'
    c.train.minimize_metric = True
    c.train.mixup = False
    c.train.early_stop_patience = 0
    c.train.accumulation_size = 0
    c.train.horizontal_flip = False
    c.train.fp16 = False

    # test
    c.test = edict()
    c.test.batch_size = 16
    c.test.tta = False

    # optimizer
    c.optimizer = edict()
    c.optimizer.name = 'Adam'
    c.optimizer.params_type = 'weight_decay'
    c.optimizer.params = edict()
    c.optimizer.params.encoder_lr = 1.0e-4
    c.optimizer.params.decoder_lr = 1.0e-4
    c.optimizer.params.weight_decay = 1.0e-4

    # scheduler
    c.scheduler = edict()
    c.scheduler.name = 'plateau'
    c.scheduler.params = edict()

    # transforms
    c.transforms = edict()
    c.transforms.params = edict()

    c.transforms.train = edict()
    c.transforms.train.mean = [0.485, 0.456, 0.406]
    c.transforms.train.std = [0.229, 0.224, 0.225]
    c.transforms.train.Contrast = False
    c.transforms.train.Noise = False
    c.transforms.train.Blur = False
    c.transforms.train.Distort = False
    c.transforms.train.ShiftScaleRotate = False

    c.transforms.test = edict()
    c.transforms.test.mean = [0.485, 0.456, 0.406]
    c.transforms.test.std = [0.229, 0.224, 0.225]
    c.transforms.test.Contrast = False
    c.transforms.test.Noise = False
    c.transforms.test.Blur = False
    c.transforms.test.Distort = False
    c.transforms.test.ShiftScaleRotate = False

    # losses
    c.loss = edict()
    c.loss.name = 'Center'
    c.loss.params = edict()
    c.loss.params.focal = False
    c.loss.params.reduce = 'sum'
    c.loss.return_callback = False

    c.device = 'cuda'
    c.num_workers = 16
    c.work_dir = './work_dir'
    c.checkpoint_path = None
    c.debug = False

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid, Loader=yaml.SafeLoader))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config


def save_config(config, file_name):
    with open(file_name, "w") as wf:
        yaml.dump(config, wf)
