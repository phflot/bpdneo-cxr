from torch.utils.data import Dataset
from torchxrayvision.datasets import normalize
from collections import defaultdict
import random
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
from os.path import join

import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import torchvision.transforms as T
from pathlib import Path


class BPDDatasetOld(Dataset):
    def __init__(self, image_path, stage="train", augmentor=None):
        self.image_path = image_path
        self.data = []
        self.augmentor = augmentor
        image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join('bpd_im_2', f))]

        # Group images by patient number
        patient_images = defaultdict(list)
        for image_file in image_files:
            patient_number = image_file.split('_')[0]  # Extract patient number
            patient_images[patient_number].append(image_file)

        # Group images by patient number
        patient_images = defaultdict(list)
        for image_file in image_files:
            patient_number = image_file.split('_')[0]  # Extract patient number
            patient_images[patient_number].append(image_file)

        # Split patients into training and testing sets
        patients = list(patient_images.keys())
        random.shuffle(patients)
        num_train_patients = int(0.8 * len(patients))  # 80% for training, 20% for testing
        train_patients = patients[:num_train_patients]
        test_patients = patients[num_train_patients:]

        # Assign images to training and testing sets, ensuring all images of a patient are in the same set
        # As there can multiple images for the same patient in dataset it they are grouped by such that they belong either to test or train data
        train_set = []
        test_set = []
        for patient_number, images in patient_images.items():
            if patient_number in train_patients:
                train_set.extend(images)
            else:
                test_set.extend(images)

        training_df = pd.DataFrame(train_set, columns=['filename'])
        testing_df = pd.DataFrame(test_set, columns=['filename'])
        training_df['label'] = training_df['filename'].apply(
            lambda x: 'BPD' if 'mild' in x or 'mod' in x or 'sev' in x else 'NoBPD')
        testing_df['label'] = testing_df['filename'].apply(
            lambda x: 'BPD' if 'mild' in x or 'mod' in x or 'sev' in x else 'NoBPD')

        if stage == "train":
            self.data = training_df
        else:
            self.data = testing_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        image_filename = self.data['filename'][idx]
        image = imread(os.path.join(self.image_path, image_filename))
        image = normalize(image)
        label = self.data['label'][idx]

        if self.augmentor is not None:
            image = self.augmentor(image)

        sample["idx"] = idx
        sample["lab"] = label
        sample["img"] = image

        return sample


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
])

def load_and_preprocess_image(filename):
    img_path = os.path.join(filename)
    image = Image.open(img_path).convert('L')
    image = np.array(image)
    image = xrv.datasets.normalize(image, 255)
    image = transform(image)
    image = image
    return image


class BPDTestTrainSplitter:
    def __init__(self, input_path):
        input_file = join(input_path, "labels.xlsx")
        df = pd.read_excel(input_file, sheet_name=None)
        df = df["Tabelle1"]

        image_path = join(input_path, "01_manual_preprocessing_rot_crop")
        images = glob(os.path.join(image_path, "*.png"))

        def generate_label(filename):
            base_name = os.path.basename(filename)
            patient_id = base_name.split('_')[0]
            label = df['bpd'].loc[df['patient_id'] == patient_id].to_numpy()[0]
            label_map = {
                "no BPD": 0,
                "mild": 0,
                "moderate": 1,
                "severe": 1,
                "keine Angabe": -1,
            }
            return patient_id, label_map[label]

        labels = []
        for filename in images:
            patient_id, label = generate_label(filename)
            labels.append(label)
        labels = np.array(labels)
        valid_idx = labels != -1
        self.labels = labels[valid_idx]
        self.images = np.array(images)[valid_idx]

        self.image_path = image_path

        patient_id = []
        for img in self.images:
            img = os.path.basename(img)
            patient_id.append(img.split('_')[0])

        patient_df = pd.DataFrame({
            'patient_id': patient_id,
            'image_path': self.images,
            'label': self.labels
        })

        bpd_patients = patient_df[patient_df['label'] == 1]['patient_id'].unique().tolist()
        no_bpd_patients = patient_df[patient_df['label'] == 0]['patient_id'].unique().tolist()

        test_size_bpd = int(0.15 * len(bpd_patients))
        test_size_no_bpd = int(0.15 * len(no_bpd_patients))

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        test_bpd_patients = random.sample(bpd_patients, test_size_bpd)
        test_no_bpd_patients = random.sample(no_bpd_patients, test_size_no_bpd)

        test_patients = test_bpd_patients + test_no_bpd_patients

        test_set_df = patient_df[patient_df['patient_id'].isin(test_patients)]
        train_set_df = patient_df[~patient_df['patient_id'].isin(test_patients)]

        while len(test_set_df[test_set_df['label'] == 1]) != len(test_set_df[test_set_df['label'] == 0]):
            if len(test_set_df[test_set_df['label'] == 1]) > len(test_set_df[test_set_df['label'] == 0]):
                more_no_bpd_patient = random.sample(set(no_bpd_patients) - set(test_no_bpd_patients), 1)[0]
                test_no_bpd_patients.append(more_no_bpd_patient)
            else:
                more_bpd_patient = random.sample(set(bpd_patients) - set(test_bpd_patients), 1)[0]
                test_bpd_patients.append(more_bpd_patient)

            test_patients = test_bpd_patients + test_no_bpd_patients
            test_set_df = patient_df[patient_df['patient_id'].isin(test_patients)]
            train_set_df = patient_df[~patient_df['patient_id'].isin(test_patients)]
        test_set_df.to_csv(join(input_path, "test_set.csv"))
        train_set_df.to_csv(join(input_path, "train_set.csv"))


class BPDDataset(Dataset):
    def __init__(self, input_path, stage="train", augmentor=None):

        self.input_path = input_path
        self.augmentor = augmentor

        if stage == "train":
            print("loading train set")
            self.data = pd.read_csv(join(input_path, "train_set.csv"))
        elif stage == "test":
            print("loading test set")
            self.data = pd.read_csv(join(input_path, "test_set.csv"))
        else:
            raise ValueError("stage must be 'train' or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = load_and_preprocess_image(self.data.iloc[idx]["image_path"])
        label = self.data.iloc[idx]["label"]

        if self.augmentor is not None:
            image = self.augmentor(image)

        return image.to(torch.float32),  torch.tensor(label, dtype=torch.float32)


class BPDDatasetRGB(torch.utils.data.Dataset):
    def __init__(self, root: str, stage: str, augmentor=None):
        csv = Path(root) / f"{stage}_set.csv"
        self.df = pd.read_csv(csv)
        self.augmentor = augmentor or (lambda x: x)
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.image_path).convert("RGB")
        img = self.transform(img)
        img = self.augmentor(img)
        return img, torch.tensor(row.label, dtype=torch.float32)


if __name__ == "__main__":
    image_path = "/local/landmark_project/bpd"

    BPDTestTrainSplitter(image_path)

    dataset = BPDDataset(image_path)
    img, label = dataset[0]

    print(len(dataset))
