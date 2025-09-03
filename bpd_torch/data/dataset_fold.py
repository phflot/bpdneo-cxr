import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchxrayvision as xrv


class BPDDataset_5fold(Dataset):
    """
    A PyTorch Dataset for single-channel (grayscale) X-ray images, designed
    to work with a 5-fold cross-validation scheme.

    It is initialized with a master DataFrame and a list of patient IDs for the
    specific fold (train or test).

    - In 'train' mode, it includes all images for the selected patients.
    - In 'test' mode, it includes only one image per patient to ensure a
      patient-centric evaluation.
    """

    def __init__(self, master_df: pd.DataFrame, patient_ids: np.ndarray,
                 stage: str = "train", augmentor=None):
        """
        Args:
            master_df (pd.DataFrame): The full DataFrame containing columns
                                      ['patient_id', 'image_path', 'label'].
            patient_ids (np.ndarray): An array of patient IDs to include in this
                                      dataset instance.
            stage (str, optional): Either "train" or "test". Defaults to "train".
            augmentor (callable, optional): Augmentation pipeline. Defaults to None.
        """
        super().__init__()
        self.augmentor = augmentor
        self.stage = stage

        # Filter the master dataframe to only the patient IDs for this fold
        fold_df = master_df[master_df['patient_id'].isin(patient_ids)]

        if self.stage == 'train':
            # For training, use all available images for the selected patients
            self.data = fold_df # .reset_index(drop=True)
        else:  # stage == 'test'
            # For testing, use only the first image for each patient
            # This ensures a patient-level evaluation
            self.data = fold_df.drop_duplicates(subset=['patient_id']).reset_index(drop=True)
            # self.data = fold_df # .reset_index(drop=True)


        # Pre-defined transform for the XRV models (single-channel)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((512, 512), antialias=True),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = torch.tensor(row['label'], dtype=torch.float32)

        # Load as single-channel grayscale, normalize for XRV
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        image = xrv.datasets.normalize(image, 255)  # Specific normalization for XRV

        # Apply transforms
        image = self.transform(image)

        if self.stage == 'train' and self.augmentor is not None:
            image = self.augmentor(image)

        return image.to(torch.float32), label


class BPDDatasetRGB_5fold(Dataset):
    """
    A PyTorch Dataset for three-channel (RGB) X-ray images, designed
    to work with a 5-fold cross-validation scheme.

    It is initialized with a master DataFrame and a list of patient IDs for the
    specific fold (train or test).

    - In 'train' mode, it includes all images for the selected patients.
    - In 'test' mode, it includes only one image per patient to ensure a
      patient-centric evaluation.
    """

    def __init__(self, master_df: pd.DataFrame, patient_ids: np.ndarray,
                 stage: str = "train", augmentor=None):
        """
        Args:
            master_df (pd.DataFrame): The full DataFrame containing columns
                                      ['patient_id', 'image_path', 'label'].
            patient_ids (np.ndarray): An array of patient IDs to include in this
                                      dataset instance.
            stage (str, optional): Either "train" or "test". Defaults to "train".
            augmentor (callable, optional): Augmentation pipeline. Defaults to None.
        """
        super().__init__()
        self.augmentor = augmentor
        self.stage = stage

        # Filter the master dataframe to only the patient IDs for this fold
        fold_df = master_df[master_df['patient_id'].isin(patient_ids)]

        if self.stage == 'train':
            # For training, use all available images for the selected patients
            self.data = fold_df.reset_index(drop=True)
        else:  # stage == 'test'
            # For testing, use only the first image for each patient
            # This ensures a patient-level evaluation
            self.data = fold_df.drop_duplicates(subset=['patient_id']).reset_index(drop=True)

        # Pre-defined transform for the torchvision models (3-channel)
        self.transform = T.Compose([
            T.Resize((512, 512), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = torch.tensor(row['label'], dtype=torch.float32)

        # Load and convert to 3-channel RGB
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        image = self.transform(image)

        if self.stage == 'train' and self.augmentor is not None:
            image = self.augmentor(image)

        return image, label
