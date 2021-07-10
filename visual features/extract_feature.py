import os

import csv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans
from PIL import Image
import numpy as np

from config import get_config
from data import data_pipe
from models.iresnetse import IrResNetSe
from models.iresnet import iresnet100, iresnet100_nohead
from models.iresnet_2branches import iresnet100_2branches


class TestDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
        self.length = len(self.imgs)
        self.transform = trans.Compose([
            trans.Resize((112, 112)),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.imgs[index]
        img = Image.open(image_path)
        img = self.transform(img)
        return img, image_path


def test_dataloader(video_folder):
    imgs = []
    for root, _, files in os.walk(video_folder):
        for file in files:
            imgs.append(os.path.join(root, file))

    ds = TestDataset(imgs)
    loader = DataLoader(ds, batch_size=64,
                        shuffle=False, pin_memory=True,
                        num_workers=1, drop_last=False)
    return loader



class ValDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
        self.length = len(self.imgs)
        self.transform = trans.Compose([
            trans.Resize((112, 112)),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        img = self.transform(img)
        return img, label


def val_data_loader(data_path, annot_path):
    imgs = []
    labels = []
    with open(annot_path) as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            img_path = line[0]
            label = int(line[1])
            if label != -1:
                if os.path.exists(os.path.join(data_path, img_path)):
                    imgs.append(os.path.join(data_path, img_path))
                    labels.append(label)

    ds = ValDataset(imgs, labels)
    print('val data loader generated')
    loader = DataLoader(ds, batch_size=64,
                        shuffle=False, pin_memory=True,
                        num_workers=4, drop_last=False)
    return loader


class ExtractFeature():
    def __init__(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = '/data/jinyue/logs/v2.3.3_IResnet100-dropout0.6_focal-pytorch_sgd0.001_basic+color_nosample-\
modeleval/save/model_2021-06-15-20-50_loss-0.010_acc-0.630_f1-0.402_step-21760.pth'
        self.data_path = '/data/jinyue/dataset/opensource/AffWild2/cropped_aligned'
        self.expr_train_annot_path = '/data/jinyue/dataset/opensource/AffWild2/Annotations/expression_annots_train.csv'
        self.expr_valid_annot_path = '/data/jinyue/dataset/opensource/AffWild2/Annotations/expression_annots_valid.csv'

        self.output_dir = '/data/jinyue/extracted_features/v2.3.3'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.train_labels_npy = os.path.join(self.output_dir, 'train_labels.npy')
        self.train_features_npy = os.path.join(self.output_dir, 'train_features.npy')
        self.valid_labels_npy = os.path.join(self.output_dir, 'valid_labels.npy')
        self.valid_features_npy = os.path.join(self.output_dir, 'valid_features.npy')

        self.model = iresnet100_nohead().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.transform = trans.Compose([
            trans.Resize((112, 112)),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def inference_single_image(self, image_path):
        img = Image.open(image_path)
        img = self.transform(img)
        output = self.model(img.unsqueeze(0).to(self.device))
        print(output.shape)
        return output

    def inference_dataloader_to_npy(self):
        # extract train data features
        train_features = []
        train_labels = []
        data_loader = val_data_loader(self.data_path, self.expr_train_annot_path)
        for imgs, labels in tqdm(iter(data_loader)):
            imgs = imgs.to(self.device)
            outputs = self.model(imgs).cpu().detach().numpy()
            for item in labels:
                train_labels.append(item)
            for item in outputs:
                train_features.append(item)
        np.save(self.train_features_npy, train_features)
        np.save(self.train_labels_npy, train_labels)

        # extract valid data features
        valid_features = []
        valid_labels = []
        valid_data_loader = val_data_loader(self.data_path, self.expr_valid_annot_path)
        for imgs, labels in tqdm(iter(valid_data_loader)):
            imgs = imgs.to(self.device)
            outputs = self.model(imgs).cpu().detach().numpy()
            for item in labels:
                valid_labels.append(item)
            for item in outputs:
                valid_features.append(item)
        np.save(self.valid_features_npy, valid_features)
        np.save(self.valid_labels_npy, valid_labels)


if __name__ == '__main__':
    inference_module = ExtractFeature()

    test_dataset = '/data/jinyue/dataset/opensource/AffWild2/unannotated_expr_cropped_aligned'
    inference_module.inference_dataloader_to_npy()
