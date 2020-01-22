import cv2
from sklearn.metrics import accuracy_score, f1_score
from transforms import get_transforms
from datasets import make_loader
from losses import depth_transform
from config.base import load_config
from utils import coords2str, str2coords, dict_to_json
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

    validloader = make_loader(
        data_dir=config.data.train_dir,
        df_path=config.data.train_df_path,
        features=config.data.features,
        phase='valid',
        img_size=(config.data.height, config.data.width),
        batch_size=config.test.batch_size,
        num_workers=config.num_workers,
        idx_fold=fold,
        transforms=get_transforms(config.transforms.test),
        model_scale=config.data.model_scale,
        return_fnames=True,
    )


    # load model
    checkpoint_path = config.work_dir + '/checkpoints/best.pth'
    model = load_model(config_file, checkpoint_path)

    folds = pd.read_csv('data/folds.csv')

    predictions = []
    targets = []
    image_ids = []
    z_pos = config.data.z_pos[0]
    with torch.no_grad():
        for i, (batch_images, batch_mask_regr, batch_image_ids) in enumerate(tqdm(validloader)):
            batch_preds = model(batch_images.to(config.device))
            batch_preds[:,0] = torch.sigmoid(batch_preds[:,0])
            batch_preds[:,z_pos] = depth_transform(batch_preds[:,z_pos])

            batch_preds = batch_preds.data.cpu().numpy()
            batch_mask_regr = batch_mask_regr.data.cpu().numpy()
            image_ids.extend(batch_image_ids)

            for preds, mask_regr, image_id in zip(batch_preds, batch_mask_regr, batch_image_ids):
                coords = extract_coords(
                    preds,
                    features=config.data.features,
                    img_size=(config.data.height, config.data.width),
                    confidence_threshold=config.test.confidence_threshold,
                    distance_threshold=config.test.distance_threshold,
                    )
                predictions.append(coords)

                s = folds.loc[folds.ImageId == image_id.split('.jpg')[0], 'PredictionString'].values[0]
                true_coords = str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z'])
                targets.append(true_coords)

    with open(config.work_dir + '/predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    with open(config.work_dir + '/targets.pkl', 'wb') as f:
        pickle.dump(targets, f)

    rows = []
    for p, i in zip(predictions, image_ids):
        rows.append({'ImageId':i, 'PredictionString':coords2str(p)})
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(config.work_dir +'/val_pred.csv', index=False)

    all_result, result = calc_map_score(targets, predictions)
    result['confidence_threshold']=config.test.confidence_threshold
    result['distance_threshold']=config.test.distance_threshold

    dict_to_json(all_result, config.work_dir + '/all_result_th{}.json'.format(config.test.distance_threshold ))
    dict_to_json(result, config.work_dir + '/result_th{}.json'.format(config.test.distance_threshold ))

    for k in sorted(result.keys()):
        print(k,result[k])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    parser.add_argument('--device_id', '-d', default='0', type=str)
    parser.add_argument('--fold', '-f', default=0, type=int)
    return parser.parse_args()


def main():
    print('validate model.')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    run(args.config_file, args.fold, args.device_id)


if __name__ == '__main__':
    main()