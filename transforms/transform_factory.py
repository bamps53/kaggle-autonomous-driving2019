from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomCrop,
    Resize,
    Cutout,
    Normalize,
    Compose,
    GaussNoise,
    IAAAdditiveGaussianNoise,
    RandomContrast,
    RandomGamma,
    RandomRotate90,
    RandomSizedCrop,
    RandomBrightness,
    ShiftScaleRotate,
    MotionBlur,
    MedianBlur,
    Blur,
    OpticalDistortion,
    GridDistortion,
    IAAPiecewiseAffine,
    OneOf)
from albumentations.pytorch import ToTensor

HEIGHT, WIDTH = 2710 // 2, 3384

def get_transforms(phase_config):
    list_transforms = []
    if phase_config.Noise:
        list_transforms.append(
            OneOf([
                GaussNoise(),
                IAAAdditiveGaussianNoise(),
            ], p=0.5),
        )
    if phase_config.Contrast:
        list_transforms.append(
            OneOf([
                RandomContrast(0.5),
                RandomGamma(),
                RandomBrightness(),
            ], p=0.5),
        )
    if phase_config.Blur:
        list_transforms.append(
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.5)
        )
    if phase_config.Distort:
        list_transforms.append(
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.5)
        )
    list_transforms.extend(
        [
            Normalize(mean=phase_config.mean, std=phase_config.std, p=1),
            ToTensor(),
        ]
    )

    return Compose(list_transforms)


def hflip(img, p=1):
    return HorizontalFlip(p=p)(image=img)['image']
