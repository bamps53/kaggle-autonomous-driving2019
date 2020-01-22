import pandas as pd
from .utils import str2coords, coords2str, CAMERA_MATRIX, CAMERA_MATRIX_inv
import swifter
import os
import numpy as np
import jpeg4py as jpeg


def read_mask(image_id, mode):
    image_id = image_id + '.jpg'
    filter_path = os.path.join(
        '../input/pku-autonomous-driving/{}_masks/'.format(mode), image_id)
    if os.path.exists(filter_path):
        return jpeg.JPEG(str(filter_path)).decode()[:, :, 0]
        # return cv2.imread(str(filter_path))[:,:,0]
    else:
        return np.zeros((2710, 3384), dtype='uint8')


def filter_masked_pred(row):
    new_coords = []
    image_id = row.ImageId
    s = row.PredictionString
    if len(s) == 0:
        return row
    else:
        image_filter = read_mask(image_id, mode='test')
        coords = str2coords(
            s, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence'])
        xs, ys = get_img_coords(
            s, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence'])

        for x, y, coord in zip(xs, ys, coords):
            x, y = int(x), int(y)
            if image_filter[y, x] < 200:
                new_coords.append(coord)
        row.PredictionString = coords2str(
            new_coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence'])
        return row


def postprocess(sub_path):
    sub = pd.read_csv(sub_path)
    sub['PredictionString'].fillna('', inplace=True)
    print('before filtering:', sub.PredictionString.map(
        lambda x: x.count(' ')//7+1).sum())
    sub = sub.swifter.apply(filter_masked_pred, axis=1)
    print('after filtering:', sub.PredictionString.map(
        lambda x: x.count(' ')//7+1).sum())
    out_path = sub_path.split('.csv')[0] + '_filtered.csv'
    sub.to_csv(out_path, index=False)


def clear_duplicates(coords, DISTANCE_THRESH_CLEAR=2):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]


def rcz2xyz(r, c, z, before_img_size=(2710, 3384), after_img_size=(256, 640), model_scale=4):
    remove_height = 1355
    residual_height = 1355

    before_height, before_width = before_img_size
    after_height, after_width = after_img_size

    r = r * model_scale
    r = r / (after_height / residual_height)
    r = r + remove_height

    c = c * model_scale
    c = c / (after_width / before_width)

    hwz = np.array([c * z, r * z, z]).reshape(3, 1)
    x, y, z = list(np.dot(CAMERA_MATRIX_inv, hwz).squeeze())
    return x, y, z


def extract_coords(
    prediction,
    features,
    confidence_threshold=0.1,
    distance_threshold=2,
    img_size=(256, 640),
    model_scale=4,
):
    heatmap = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(heatmap > confidence_threshold)
    coords = []
    for r, c in points:
        regr_dict = dict(zip(features, regr_output[:, r, c]))
        regr_dict = _regr_back(regr_dict)
        regr_dict['confidence'] = heatmap[r, c]
        regr_dict['x'], regr_dict['y'], regr_dict['z'] = \
            rcz2xyz(
                r, c, regr_dict['z'], after_img_size=img_size, model_scale=model_scale)
        regr_dict['roll'] = -3.090270  # train set median
        regr_dict['yaw'] = 0.155064  # train set median
        if regr_dict['y'] > 0 and regr_dict['z'] > 0:
            coords.append(regr_dict)

    coords = clear_duplicates(coords, distance_threshold)
    return coords


def _regr_back(regr_dict):
    regr_dict['z'] = regr_dict['z'] * 100
    pitch_sin = regr_dict['pitch_sin'] / \
        np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / \
        np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict
