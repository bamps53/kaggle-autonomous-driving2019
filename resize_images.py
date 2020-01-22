import os
import cv2
import tqdm
os.makedirs('../input/pku-autonomous-driving/resized_images/', exist_ok=True)


def exist_write(image_path, img):
    if not os.path.exists(image_path):
        cv2.imwrite(image_path, img)


for image_path in tqdm.tqdm(os.listdir('../input/pku-autonomous-driving/train_images')):

    img = cv2.imread(
        '../input/pku-autonomous-driving/train_images/' + image_path)
    img = img[1355:]
    img = cv2.resize(img, (2720, 1088))
    exist_write(
        '../input/pku-autonomous-driving/resized_images/' + image_path, img)

for image_path in tqdm.tqdm(os.listdir('../input/pku-autonomous-driving/test_images')):
    img = cv2.imread(
        '../input/pku-autonomous-driving/test_images/' + image_path)
    img = img[1355:]
    img = cv2.resize(img, (2720, 1088))
    exist_write(
        '../input/pku-autonomous-driving/resized_images/' + image_path, img)
