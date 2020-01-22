import os
import cv2
import jpeg4py as jpeg
import numpy as np
import pandas as pd
import torch
import pickle
import random
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transforms import hflip


def convert_image_id(image_id):
    zeros = '0' * (6 - len(str(image_id)))
    return zeros + str(image_id)


class TrainDataset(Dataset):
    def __init__(self, df, data_dir, features, transforms, horizontal_flip, 
                img_size, model_scale, return_fnames=False):
        self.df = df
        self.data_dir = data_dir
        self.features = features
        self.num_features = len(self.features)
        self.df['z'] = self.df['z'] / 100.0
        self.df['image_id'] = self.df['image_id'].map(convert_image_id)
        self.transforms = transforms
        self.img_size = img_size
        self.model_scale = model_scale
        self.fnames = self.df.image_id.unique().tolist()
        self.return_fnames = return_fnames
        self.flip = horizontal_flip

    def __getitem__(self, idx):
        if self.flip:
            flip = np.random.randint(10) == 1
        else:
            flip = False

        image_id = self.fnames[idx]
        image_path = self.data_dir + '/{}.jpg'.format(image_id)
        img = jpeg.JPEG(image_path).decode()
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        if flip:
            img = hflip(img, p=1)

        annotations = self.df[self.df.image_id == image_id]
        # create label
        mask_size = (self.img_size[0]//self.model_scale, self.img_size[1]//self.model_scale)
        mask = np.zeros(mask_size, dtype='float32')
        regr = np.zeros([mask_size[0], mask_size[1], self.num_features], dtype='float32')
        for _, ann in annotations.iterrows():
            r = int(ann['r_heatmap'])
            c = int(ann['c_heatmap'])
            if flip:
                ann['pitch_sin'] *= -1.0
                ann['roll'] *= -1.0
                ann['x'] *= -1.0
            features = np.array([ann[col] for col in self.features]).astype(float)
            features = features.reshape(1,1,len(features))

            if 0 <= r < mask_size[0] and 0 <= c < mask_size[1]:
                kernel_size=3
                lower_r = max(r-1,0)
                upper_r = min(r+2,mask.shape[0])
                left_c = max(c-1,0)
                right_c = min(c+2,mask.shape[1])
                kernel = np.float32([[0.5, 0.75, 0.5], [0.75, 1.0, 0.75], [0.5, 0.75, 0.5]])
                kernel = kernel[lower_r-(r-1):kernel_size-((r+2)-upper_r),left_c-(c-1):kernel_size-((c+2)-right_c)]
                mask[lower_r:upper_r, left_c:right_c] = kernel
                regr[lower_r:upper_r, left_c:right_c] = features

        mask = mask.reshape(mask.shape[0],mask.shape[1],1)
        mask_regr = np.concatenate([mask, regr],axis=2)
        if flip:
            mask_regr = np.array(mask_regr[:,::-1])

        augmented = self.transforms(image=img, mask=mask_regr)
        img = augmented['image']
        mask_regr = augmented['mask']

        mask_regr = mask_regr[0].permute(2, 0, 1)

        if self.return_fnames:
            return img, mask_regr, image_id
        else:
            return img, mask_regr

    def __len__(self):
        return len(self.fnames)

class TestDataset(Dataset):
    def __init__(self, df, data_dir, transforms, img_size):
        self.fnames = df['ImageId'].unique().tolist()
        self.data_dir = data_dir
        self.num_samples = len(self.fnames)
        self.transforms = transforms
        self.img_size = img_size

    def __getitem__(self, idx):
        image_id = self.fnames[idx]

        image_id = image_id + '.jpg'
        image_path = os.path.join(self.data_dir, image_id)
        img = jpeg.JPEG(image_path).decode()[1355:]
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        images = self.transforms(image=img)["image"]
        return image_id, images

    def __len__(self):
        return self.num_samples


def make_loader(
        df_path,
        data_dir,
        features,
        phase,
        img_size=(256, 640),
        batch_size=8,
        num_workers=2,
        idx_fold=None,
        transforms=None,
        horizontal_flip=False,
        model_scale=4,
        pseudo_label_path=None,
        return_fnames=False,
        debug=False,
        pseudo_path=None,
):
    if debug:
        num_rows = 100
    else:
        num_rows = None

    df = pd.read_csv(df_path, nrows=num_rows)

    if phase == 'test':
        image_dataset = TestDataset(df, data_dir, transforms, img_size)
        is_shuffle = False

    else:  # train or valid
        if phase == "train":
            df = df[df['fold'] != idx_fold]
            if pseudo_path:
                pseudo = pd.read_csv(pseudo_path)
                pseudo = pseudo[pseudo['PredictionString'].notnull()]
                df = pd.concat([df, pseudo], axis=0)
            is_shuffle = True
        else:
            df = df[df['fold'] == idx_fold]
            is_shuffle = False

        image_dataset = TrainDataset(
            df=df, 
            data_dir=data_dir, 
            features=features,
            transforms=transforms, 
            horizontal_flip=horizontal_flip, 
            img_size=img_size, 
            model_scale=model_scale, 
            return_fnames=return_fnames, 
            )


    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=is_shuffle,
    )
