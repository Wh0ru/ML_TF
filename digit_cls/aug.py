import imgaug as ia
from imgaug import augmenters as iaa

def Aug(img, seed=0.5):
    sometimes = lambda aug: iaa.Sometimes(seed, aug)
    seq = iaa.Sequential([
        sometimes(
            iaa.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                       rotate=(-20, 20),
                       shear=(-10, 10)))
    ])
    img = seq.augment_image(img)
    return img
