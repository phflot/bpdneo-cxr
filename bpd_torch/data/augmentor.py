import torch
import torchvision.transforms as transforms
import random
from torch.utils.data._utils.collate import default_collate
import numpy as np


def mixup_collate_depr(batch, *, alpha: float, prob: float):
    if random.random() >= prob:
        return default_collate(batch)
    imgs, lbls = zip(*batch)
    imgs = torch.stack(imgs)
    lbls = torch.stack(lbls)
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0))
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, (lbls, lbls[idx], lam)


def mixup_collate(batch, *, alpha: float, prob: float):
    imgs, lbls = zip(*batch)
    imgs = torch.stack(imgs)
    lbls = torch.stack(lbls)

    if random.random() < prob:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(imgs.size(0))
        mixed_imgs = lam * imgs + (1 - lam) * imgs[idx]

        return mixed_imgs, (lbls, lbls[idx], lam)
    else:
        return imgs, (lbls, lbls, 1.0)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mixup_collate(batch, *, alpha: float, prob: float):
    """
    Applies CutMix augmentation to a batch of images and labels.
    Returns data in a unified format compatible with the training loop.
    """
    imgs, lbls = zip(*batch)
    imgs = torch.stack(imgs)
    lbls = torch.stack(lbls)

    if random.random() < prob:
        # Sample the mixing ratio 'lam' from a Beta distribution.
        # Note: Standard CutMix typically uses alpha = 1.0. See note below.
        lam_beta = np.random.beta(alpha, alpha)

        # Get a shuffled index to select images to cut from
        idx = torch.randperm(imgs.size(0))

        # Generate the bounding box for the patch
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam_beta)

        # Create a copy of the original images to modify
        mixed_imgs = imgs.clone()

        # Paste the patch from a shuffled image onto the original image
        mixed_imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[idx][:, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to match the true ratio of the patch area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

        # Return the new images and the mixed labels in the unified format
        return mixed_imgs, (lbls, lbls[idx], lam)
    else:
        # If not applying CutMix, return the original batch in the same format
        return imgs, (lbls, lbls, 1.0)


class RandomMasking:
    def __init__(self, mask_prob=0.25, p=0.1):
        self.mask_prob = mask_prob
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        mask = torch.rand_like(img) > self.mask_prob
        return img * mask


def get_augmentations(size):
    return transforms.Compose([
    transforms.RandomRotation(degrees=5, interpolation=transforms.functional.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size, scale=(0.95, 1.05)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.05, 2))], p=0.5),
    RandomMasking(mask_prob=0.25, p=0.1),
    ])
