import cv2
from sklearn.metrics import accuracy_score, f1_score
from transforms import get_transforms
from datasets import make_loader
from losses import depth_transform
from config.base import load_config
from utils import coords2str, str2coords
from utils.postprocess import extract_coords
from utils.metrics import calc_map_score
from utils.functions import predict_batch
from models import CenterNetFPN, load_model
import segmentation_models_pytorch as smp
from catalyst.dl.utils import load_checkpoint
import argparse
import os
import warnings
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pickle

warnings.filterwarnings("ignore")


def run(config_file, fold=0, device_id=0):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    config = load_config(config_file)
    if not '_fold' in config.work_dir:
        config.work_dir = config.work_dir + '_fold{}'.format(fold)

    testloader = make_loader(
        data_dir=config.data.test_dir,
        df_path=config.data.sample_submission_path,
        features=config.data.features,
        phase='test',
        img_size=(config.data.height, config.data.width),
        batch_size=config.test.batch_size,
        num_workers=config.num_workers,
        transforms=get_transforms(config.transforms.test),
    )

    checkpoint_path = config.work_dir + '/checkpoints/best.pth'
    model = load_model(config_file, checkpoint_path, fold)

    predictions = []
    z_pos = config.data.z_pos[0]
    with torch.no_grad():
        for i, (batch_fnames, batch_images) in enumerate(tqdm(testloader)):
            batch_images = batch_images.to(config.device)
            batch_preds = model(batch_images.to(config.device))
            batch_preds[:,0] = torch.sigmoid(batch_preds[:,0])
            batch_preds[:,z_pos] = depth_transform(batch_preds[:,z_pos])
            batch_preds = batch_preds.data.cpu().numpy()

            for preds in batch_preds:
                coords = extract_coords(
                    preds,
                    features=config.data.features,
                    img_size=(config.data.height, config.data.width),
                    confidence_threshold=config.test.confidence_threshold,
                    distance_threshold=config.test.distance_threshold,
                    )
                s = coords2str(coords)
                predictions.append(s)

    #---------------------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------------------------------------------------
    test = pd.read_csv(config.data.sample_submission_path)
    test['PredictionString'] = predictions
    out_path = config.work_dir + 'submission.csv'
    postprocess(out_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file',
                        default=None, type=str)
    parser.add_argument('--device_id', '-d', default='0', type=str)
    parser.add_argument('--fold', '-f', default=0, type=int)
    return parser.parse_args()


def main():
    print('predict model.')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    run(args.config_file, args.fold, args.device_id)


if __name__ == '__main__':
    main()
