from transforms import get_transforms
from schedulers import get_scheduler
from losses import get_criterion_and_callback
from optimizers import get_optimizer
from datasets import make_loader
from config.base import load_config, save_config
from models import CenterNetFPN
import segmentation_models_pytorch as smp
from catalyst.utils import get_device
from catalyst.dl.callbacks import DiceCallback, IouCallback, CheckpointCallback, MixupCallback, \
    EarlyStoppingCallback, OptimizerCallback, CriterionCallback, CriterionAggregatorCallback
from catalyst.dl import SupervisedRunner
import argparse
import os
import warnings
from catalyst.dl.utils import load_checkpoint

warnings.filterwarnings("ignore")


def run(config_file, device_id, idx_fold):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    print('info: use gpu No.{}'.format(device_id))

    config = load_config(config_file)

    #for n-folds loop
    if config.data.params.idx_fold == -1:
        config.data.params.idx_fold = idx_fold
        config.work_dir = config.work_dir + '_fold{}'.format(idx_fold)
    elif config.data.params.idx_fold == 0:
        original_fold = int(config.work_dir.split('_fold')[1])
        if original_fold == idx_fold:
            raise Exception('if you specify fold 0, you should use train.py or resume from fold 1.')
        config.data.params.idx_fold = idx_fold
        config.work_dir = config.work_dir.split('_fold')[0] + '_fold{}'.format(idx_fold)
    else:
        raise Exception('you should use train.py if idx_fold is specified.')
    print('info: training for fold {}'.format(idx_fold))

    if not os.path.exists(config.work_dir):
        os.makedirs(config.work_dir, exist_ok=True)

    all_transforms = {}
    all_transforms['train'] = get_transforms(config.transforms.train)
    all_transforms['valid'] = get_transforms(config.transforms.test)

    dataloaders = {
        phase: make_loader(
            df_path=config.data.train_df_path,
            data_dir=config.data.train_dir,
            features=config.data.features,
            phase=phase,
            img_size=(config.data.height, config.data.width),
            batch_size=config.train.batch_size,
            num_workers=config.num_workers,
            idx_fold=config.data.params.idx_fold,
            transforms=all_transforms[phase],
            horizontal_flip=config.train.horizontal_flip,
            model_scale=config.data.model_scale,
            debug=config.debug,
            pseudo_path=config.data.pseudo_path,
        )
        for phase in ['train', 'valid']
    }

    # create segmentation model with pre trained encoder
    num_features = len(config.data.features)
    print('info: num_features =',num_features)
    model = CenterNetFPN(
        slug=config.model.encoder,
        num_classes=num_features,
        )

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # model runner
    runner = SupervisedRunner(model=model, device=get_device())

    # train setting
    criterion, callbacks = get_criterion_and_callback(config)
    
    if config.train.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            patience=config.train.early_stop_patience))

    if config.train.accumulation_size > 0:
        accumulation_steps = config.train.accumulation_size // config.train.batch_size
        callbacks.extend(
            [OptimizerCallback(accumulation_steps=accumulation_steps)]
        )

    # to resume from check points if exists
    if os.path.exists(config.work_dir + '/checkpoints/last_full.pth'):
        callbacks.append(CheckpointCallback(
            resume=config.work_dir + '/checkpoints/last_full.pth'))

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataloaders,
        logdir=config.work_dir,
        num_epochs=config.train.num_epochs,
        main_metric=config.train.main_metric,
        minimize_metric=config.train.minimize_metric,
        callbacks=callbacks,
        verbose=True,
        fp16=config.train.fp16,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    parser.add_argument('--device_id', '-d', default='0', type=str)
    parser.add_argument('--num_folds', '-n', default=5, type=int)
    parser.add_argument('--start_fold', '-s', default=0, type=int)
    parser.add_argument('--end_fold', '-e', default=4, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    print('train model for {} folds.'.format(args.num_folds))
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    for idx_fold in range(args.start_fold, args.end_fold+1):
        run(args.config_file, args.device_id, idx_fold)


if __name__ == '__main__':
    main()