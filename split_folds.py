import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.model_selection import KFold

from config.base import load_config
from utils.postprocess import rcz2xyz
from utils import str2coords, coords2str, CAMERA_MATRIX

def add_img_coords(df, image_size, model_scale=4):
    xs = df['x']
    ys = df['y']
    zs = df['z']
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(CAMERA_MATRIX, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    df['r_img'] = img_ys - 1355 # because we croped lower half image
    df['c_img'] = img_xs
    df['r_heatmap'] = (df['r_img'] * image_size[0] / 1355 / model_scale).astype(int)
    df['c_heatmap'] = (df['c_img'] * image_size[1] / 3384 / model_scale).astype(int) 
    df['r_offset'] = df['r_img'] * image_size[0] / 1355 / model_scale - df['r_heatmap']
    df['c_offset'] = df['c_img'] * image_size[1] / 3384 / model_scale - df['c_heatmap']
    return df

def expand_df(df):
    rows = []
    for _, row in df.iterrows():
        image_id = row.ImageId
        s = row.PredictionString
        fold = row.fold
        coords = str2coords(s)
        for coord in coords:
            coord['image_id'] = image_id
            coord['fold'] = fold
            coord['pitch_cos'] = np.cos(coord['pitch'])
            coord['pitch_sin'] = np.sin(coord['pitch'])
        rows.extend(coords)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    print('split train dataset')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))

    config = load_config(args.config_file)

    if not os.path.exists('data/folds.csv'):
        df = pd.read_csv('../input/pku-autonomous-driving/train.csv')
        kf = KFold(n_splits=config.data.num_folds, shuffle=True, random_state=0)
        for i, (_, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = i
        df.to_csv('data/folds.csv',index=False)
    else:
        df = pd.read_csv('data/folds.csv')

    expanded_df = expand_df(df)
    expanded_df = add_img_coords(expanded_df, image_size=(config.data.height, config.data.width))
    expanded_df.to_csv('data/img{}_df.csv'.format(config.data.height), index=False)
    print('saved dataframe to','data/img{}_df.csv'.format(config.data.height))


if __name__ == '__main__':
    main()
