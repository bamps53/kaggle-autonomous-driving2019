import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from math import sin, cos

from utils.postprocess import clear_duplicates, extract_coords
from utils import CAMERA_MATRIX, get_img_coords


def normalize(img):
    img = img - img.min()
    img = img/img.max()
    return img


def show_image(img):
    img = normalize(img)
    plt.imshow(np.transpose(img, [1, 2, 0]))
    plt.show()


def show_image_with_str(image_id, pred_str, mode='train', names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    if mode == 'train':
        root = Path('../input/pku-autonomous-driving/train_images/')
    else:
        root = Path('../input/pku-autonomous-driving/test_images/')
    if '.jpg' not in image_id:
        image_id = image_id + '.jpg'
    img = cv2.imread(str(root / (image_id)))

    plt.imshow(img)
    plt.scatter(*get_img_coords(pred_str, names=names), color='yellow', s=50)
    plt.show()


# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
    return image


def visualize(image_id, coords, mode='train'):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    if mode == 'train':
        root = Path('../input/pku-autonomous-driving/train_images/')
    else:
        root = Path('../input/pku-autonomous-driving/test_images/')
    if '.jpg' not in image_id:
        image_id = image_id + '.jpg'
    img = cv2.imread(str(root / (image_id)))

    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(CAMERA_MATRIX, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    return img


def check_validloader_with_model(validloader, features, model, num_images=10, show_mask=False):
    num_repeat = int(num_images/validloader.batch_size)+1
    vl = iter(validloader)
    for i in range(num_repeat):
        batch_images, batch_mask_regr, batch_image_ids = next(vl)
        batch_preds = model(batch_images.to(config.device))
        batch_preds[:, 0] = torch.sigmoid(batch_preds[:, 0])
        batch_images = batch_images.data.cpu().numpy()
        batch_preds = batch_preds.data.cpu().numpy()
        batch_mask_regr = batch_mask_regr.data.cpu().numpy()

        for img, mask, pred, image_id in zip(batch_images, batch_mask_regr, batch_preds, batch_image_ids):
            mask_coords = extract_coords(
                mask, features=features, img_size=(img.shape[1], img.shape[2]))
            pred_coords = extract_coords(
                pred, features=features, img_size=(img.shape[1], img.shape[2]))
            plt.imshow(visualize(image_id, mask_coords))
            plt.show()
            plt.imshow(visualize(image_id, pred_coords))
            plt.show()
            if show_mask:
                plt.imshow(pred[0])
                plt.show()


def check_validloader(validloader, features, num_images=10, show_mask=False):
    num_repeat = int(num_images/validloader.batch_size)+1
    vl = iter(validloader)
    for i in range(num_repeat):
        batch_images, batch_mask_regr, batch_image_ids = next(vl)
        batch_images = batch_images.data.cpu().numpy()
        batch_mask_regr = batch_mask_regr.data.cpu().numpy()

        for img, mask_regr, image_id in zip(batch_images, batch_mask_regr, batch_image_ids):
            mask_coords = extract_coords(
                mask_regr, features=features, img_size=(img.shape[1], img.shape[2]))
            img_with_mask = visualize(image_id, mask_coords)
            # print(img_with_mask.shape)
            show_image(img)
            print('original image and annotation')
            plt.imshow(img_with_mask)
            plt.show()
            if show_mask:
                plt.imshow(pred[0])
                plt.show()
