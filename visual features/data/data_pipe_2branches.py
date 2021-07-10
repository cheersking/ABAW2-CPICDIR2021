from config import get_config
from torchvision import transforms as trans
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
import torchvision
from PIL import Image, ImageFile
from typing import Callable
import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import csv
import os
import sys
sys.path.append('..')


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))
                            ) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(
            self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[:][1]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class TrainDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
        self.length = len(self.imgs)
        self.transform = trans.Compose([
            # trans.Resize((112, 112)),
            trans.RandomResizedCrop((112, 112), scale=(0.6, 1.0)),
            trans.ColorJitter(brightness=0.3, contrast=0.3,
                              saturation=0.3, hue=0.3),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # self.transform = trans.Compose([
        #     trans.Resize((112, 112)),
        #     ImageNetPolicy(),
        #     trans.ToTensor(),
        #     trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.imgs[index]  # original code:index+1
        img = Image.open(img_path)
        # original code:index+1
        label = torch.tensor(self.labels[index], dtype=torch.long)
        img = self.transform(img)
        return img, label

    def get_labels(self):
        return self.labels


def train_data_loader(conf):
    img_folder = conf.train_data_path
    # pseudo_img_folder = conf.pseudo_train_data_path
    annot_path = conf.expr_train_annot_path
    imgs = []
    labels = []
    with open(annot_path) as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            img_path = line[0]
            label = int(line[1])
            if label != -1:
                if os.path.exists(os.path.join(img_folder, img_path)):
                    imgs.append(os.path.join(img_folder, img_path))
                    labels.append(label)
                # if os.path.exists(os.path.join(pseudo_img_folder, img_path)):
                #     imgs.append(os.path.join(pseudo_img_folder, img_path))
                #     labels.append(label)

    ds = TrainDataset(imgs, labels)
    print('train data loader generated')
    if conf.sample:
        print('do sampling')
        loader = DataLoader(ds, batch_size=conf.batch_size,
                            shuffle=False, pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers, drop_last=True,
                            sampler=ImbalancedDatasetSampler(ds))
    else:
        loader = DataLoader(ds, batch_size=conf.batch_size,
                            shuffle=True, pin_memory=conf.pin_memory,
                            num_workers=conf.num_workers, drop_last=True)
    return loader


class ValDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
        self.length = len(self.imgs) - 1
        self.transform = trans.Compose([
            trans.Resize((112, 112)),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.imgs[index+1]
        img = Image.open(img_path)
        label = torch.tensor(self.labels[index+1], dtype=torch.long)
        img = self.transform(img)
        return img, label


def val_data_loader(conf):
    img_folder = conf.train_data_path
    annot_path = conf.expr_valid_annot_path
    imgs = []
    labels = []
    with open(annot_path) as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            img_path = line[0]
            label = int(line[1])
            if label != -1:
                if os.path.exists(os.path.join(img_folder, img_path)):
                    imgs.append(os.path.join(img_folder, img_path))
                    labels.append(label)

    ds = ValDataset(imgs, labels)
    print('val data loader generated')
    loader = DataLoader(ds, batch_size=conf.batch_size,
                        shuffle=False, pin_memory=conf.pin_memory,
                        num_workers=conf.num_workers, drop_last=False)
    return loader


AU_list = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]


class AUDataset(Dataset):
    def __init__(self, image_dir, txt_file, label_width=12, transform=None, pred_txt_file=None,flag='train'):
        self._image_dir = image_dir
        self._flag = flag
        self.aus = [1, 2, 4, 6, 7, 10,  12,  15, 23, 24, 25, 26]
        # self.aus = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 43]
        self.au_names = [f'AU{au:0>2}' for au in self.aus]

        # self._samples = pd.read_csv(txt_file).loc[:, ['filename'] + [f'AU{au:0>2}' for au in self.aus]]
        self._label_width = label_width
        self._transform = transform
        self._bp4d_path = '/mnt/data3/jinyue/dataset/opensource/BP4D-cropped-aligned'
        self._affwild_path = '/mnt/data3/jinyue/dataset/opensource/AffWild2/cropped_aligned'
        if flag=='train':
            # self._train_samples = np.load('/data/tianqing/dataset/au_label_merged.npy',allow_pickle=True)
            # self._train_paths = np.load('/data/tianqing/dataset/au_path_merged.npy',allow_pickle=True)
            # self._valid_samples = np.load('/home/tianqing/Face/au_valid_list(without-1).npy',allow_pickle=True)
            # self._valid_samples[:,0] = '/data/jinyue/dataset/opensource/AffWild2/cropped_aligned/'+self._valid_samples[:,0]

            # self._samples = np.concatenate((self._train_samples,self._valid_samples[:,1:]),axis=0)
            # self._paths = np.concatenate((self._train_paths,self._valid_samples[:,0]),axis=0)
            # self._sampled_id = list(range(self._samples.shape[0]))
            self._samples = np.load('/mnt/data3/jinyue/dataset/opensource/AffWild2/au_label_train_V2_new.npy',allow_pickle=True)
            self._paths = np.load('/mnt/data3/jinyue/dataset/opensource/AffWild2/au_path_train_V2_new.npy',allow_pickle=True)
            self. _sampled_id = list(range(self._samples.shape[0]))
        else:
            # self._samples = np.load('/home/tianqing/Face/au_valid_list(without-1).npy',allow_pickle=True)
            # self. _sampled_id = list(range(self._samples.shape[0]))
            self._samples = np.load('/mnt/data3/jinyue/dataset/opensource/AffWild2/au_label_valid_V2_new.npy',allow_pickle=True)
            self._paths = np.load('/mnt/data3/jinyue/dataset/opensource/AffWild2/au_path_valid_V2_new.npy',allow_pickle=True)
            self. _sampled_id = list(range(self._samples.shape[0]))
        # self._check_dataset()
        # self._pred_samples = pd.read_csv(txt_file).loc[:, ['filename'] + [f'AU{au:0>2}' for au in self.aus]] if pred_txt_file else None

    def __len__(self):
        # return self._samples.shape[0]
        return len(self. _sampled_id)

    # def __getitem__(self, index):
    #     # row = self._samples.loc[index]
    #     # fname = row['filename']
    #     if self._flag=='train':
    #         fname = self._paths[index]
    #         labels = self._samples[index].astype(int)
    #         ##############modify文件夹名称不一样
    #         with open(fname, 'rb') as f:
    #             img = Image.open(f).convert('RGB')#(112,112,3)

    #     if self._flag!='train':
    #         row = self._samples[self._sampled_id[index]]
    #         fname = row[0]
    #         labels = row[1:].astype(int)
    #         with open(os.path.join(self._image_dir, fname), 'rb') as f:
    #             img = Image.open(f).convert('RGB')#(112,112,3)
    #     if self._transform:
    #         img = self._transform(img)
    #     # if self._pred_samples is not None:
    #     #     prediction = self._pred_samples.loc[index, 'AU01':]
    #     #     labels = np.where(np.logical_and(labels == 999, prediction > 0.7), 1, labels)
    #     labels = torch.tensor(labels, dtype=torch.int32)
    #     return img, labels
    def __getitem__(self, index):
        # row = self._samples.loc[index]
        # fname = row['filename']
        # if self._flag=='train':
        fname = self._paths[index]
        if 'AffWild2' in fname:
            fname = fname.replace('/data/jinyue/dataset/opensource/AffWild2/cropped_aligned', self._affwild_path)
        elif 'BP4D' in fname:
            fname = fname.replace('/data/tianqing/BP4D-cropped-aligned', self._bp4d_path)
        labels = self._samples[index].astype(int)
        ##############modify文件夹名称不一样
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')#(112,112,3)

        # if self._flag!='train':
        #     row = self._samples[self._sampled_id[index]]
        #     fname = row[0]
        #     labels = row[1:].astype(int)
        #     with open(os.path.join(self._image_dir, fname), 'rb') as f:
        #         img = Image.open(f).convert('RGB')#(112,112,3)
        if self._transform:
            img = self._transform(img)
        # if self._pred_samples is not None:
        #     prediction = self._pred_samples.loc[index, 'AU01':]
        #     labels = np.where(np.logical_and(labels == 999, prediction > 0.7), 1, labels)
        labels = torch.tensor(labels, dtype=torch.int32)
        return img, labels


def au_dataloader(image_dir, txt_file, batch_size, is_training=True, pred_txt_file=None, flag='train'):
    if is_training:
        transform = trans.Compose([
            # trans.Resize((112, 112)),
            trans.RandomResizedCrop((112, 112), scale=(0.6, 1.0)),
            trans.ColorJitter(brightness=0.3, contrast=0.3,
                                  saturation=0.3, hue=0.3),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        transform = trans.Compose([trans.Resize(112),
                                   trans.ToTensor(),
                                   trans.Normalize(
                                       [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ])
    return DataLoader(AUDataset(image_dir, txt_file, transform=transform, pred_txt_file=pred_txt_file, flag=flag),
                      batch_size=batch_size,
                      shuffle=is_training,
                      pin_memory=True,
                      num_workers=16,
                      drop_last=is_training)


def autolabel(rects, ax, bar_label):
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)


def plot_n_samples_each_cate(labels, figname):
    fig, ax = plt.subplots(figsize=(10, 5))
    pos_samples = np.sum(labels, axis=0)
    bar_plot = plt.bar(np.arange(len(AU_list)), pos_samples)
    autolabel(bar_plot, ax, [str(x) for x in pos_samples])
    plt.xticks(np.arange(len(AU_list)), AU_list)
    plt.ylabel("Number of Samples")
    plt.savefig(figname)
    # plt.savefig("AddWild2_AU_valid.png")
    # plt.show()


def merge_two_datasets():
    print("AMFED")
    txt_file = '/data/tianqing/dataset/AU_datasets/amfed_train.txt'
    samples = pd.read_csv(
        txt_file).loc[:, ['filename'] + [f'AU{au:0>2}' for au in AU_list]]
    # drop_total =np.load('/home/tianqing/Face/au_train_total_drop_list.npy')
    # samples.dropsdrop_total,axis=0,inplace=True)
    samples = samples.reset_index(drop=True).values
    # print(samples.values)
    labels = samples[:, 1:]
    print(999 in labels)
    labels[labels > 2] = 0
    print(999 in labels)
    print(labels.shape)
    print(labels[0])
    np.savetxt("AMFED.csv", labels, delimiter=",")
    plot_n_samples_each_cate(labels, figname='AMFED.png')

    print("EmotioNet_Asian")
    txt_file = '/data/tianqing/dataset/AU_datasets/EmotioNet_Asian.txt'
    samples = pd.read_csv(
        txt_file).loc[:, ['filename'] + [f'AU{au:0>2}' for au in AU_list]]
    # drop_total =np.load('/home/tianqing/Face/au_train_total_drop_list.npy')
    # samples.dropsdrop_total,axis=0,inplace=True)
    samples = samples.reset_index(drop=True).values
    # print(samples.values)
    labels = samples[:, 1:]
    print(999 in labels)
    labels[labels > 2] = 0
    print(999 in labels)
    print(labels.shape)
    print(labels[1])
    np.savetxt("EmotioNet_Asian.csv", labels, delimiter=",")
    plot_n_samples_each_cate(labels, figname='EmotioNet_Asian.png')

    print("emotionet")
    txt_file = '/data/tianqing/dataset/AU_datasets/emotionet_train.txt'
    samples = pd.read_csv(
        txt_file).loc[:, ['filename'] + [f'AU{au:0>2}' for au in AU_list]]
    # drop_total =np.load('/home/tianqing/Face/au_train_total_drop_list.npy')
    # samples.dropsdrop_total,axis=0,inplace=True)
    samples = samples.reset_index(drop=True).values
    # print(samples.values)
    labels = samples[:, 1:]
    print(999 in labels)
    labels[labels > 2] = 0
    print(999 in labels)
    print(labels.shape)
    print(labels[1])
    np.savetxt("emotionet.csv", labels, delimiter=",")
    plot_n_samples_each_cate(labels, figname='emotionet.png')

    print("mmi_apex")
    txt_file = '/data/tianqing/dataset/AU_datasets/mmi_apex_train.txt'
    samples = pd.read_csv(
        txt_file).loc[:, ['filename'] + [f'AU{au:0>2}' for au in AU_list]]
    # drop_total =np.load('/home/tianqing/Face/au_train_total_drop_list.npy')
    # samples.dropsdrop_total,axis=0,inplace=True)
    samples = samples.reset_index(drop=True).values
    # print(samples.values)
    labels = samples[:, 1:]
    print(999 in labels)
    labels[labels == 999] = 0
    print(999 in labels)
    print(1 in labels)
    print(labels.shape)
    print(labels[1])
    np.savetxt("mmi_apex.csv", labels, delimiter=",")
    plot_n_samples_each_cate(labels, figname='mmi_apex_train.png')


def IRLbl(labels):
    # imbalance ratio per label
    # Args:
    #	 labels is a 2d numpy array, each row is one instance, each column is one class; the array contains (0, 1) only
    N, C = labels.shape
    pos_nums_per_label = np.sum(labels, axis=0)
    max_pos_nums = np.max(pos_nums_per_label)
    return max_pos_nums/pos_nums_per_label


def MeanIR(labels):
    IRLbl_VALUE = IRLbl(labels)
    return np.mean(IRLbl_VALUE)


def ML_ROS(all_labels, Preset_MeanIR_value=None, sample_size=32):
    N, C = all_labels.shape
    MeanIR_value = MeanIR(
        all_labels) if Preset_MeanIR_value is None else Preset_MeanIR_value
    IRLbl_value = IRLbl(all_labels)
    indices_per_class = {}
    minority_classes = []
    for i in range(C):
        ids = all_labels[:, i] == 1
        indices_per_class[i] = [ii for ii, x in enumerate(ids) if x]
        if IRLbl_value[i] > MeanIR_value:
            minority_classes.append(i)
    new_all_labels = all_labels
    oversampled_ids = []
    for i in minority_classes:
        while True:
            pick_id = list(np.random.choice(indices_per_class[i], sample_size))
            indices_per_class[i].extend(pick_id)
            # recalculate the IRLbl_value
            new_all_labels = np.concatenate(
                [new_all_labels, all_labels[pick_id]], axis=0)
            oversampled_ids.extend(pick_id)
            if IRLbl(new_all_labels)[i] <= MeanIR_value:
                break
            print("oversample length:{}".format(
                len(oversampled_ids)), end='\r')

    oversampled_ids = np.array(oversampled_ids)
    return new_all_labels


class UnionDataset(Dataset):
    def __init__(self, imgs, expr_labels, au_labels, expr_masks, au_masks):
        self.imgs = imgs
        self.expr_labels = expr_labels
        self.au_labels = au_labels
        self.expr_masks = expr_masks
        self.au_masks = au_masks
        self.length = len(self.imgs)
        self.transform = trans.Compose([
            # trans.Resize((112, 112)),
            trans.RandomResizedCrop((112, 112), scale=(0.6, 1.0)),
            trans.ColorJitter(brightness=0.3, contrast=0.3,
                              saturation=0.3, hue=0.3),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        img = self.transform(img)
        expr_label = torch.LongTensor(self.expr_labels[index])
        au_label = torch.LongTensor(np.ndarray(self.au_labels[index]))
        expr_mask = torch.LongTensor(self.expr_masks[index])
        au_mask = torch.LongTensor(self.au_masks[index])
        return img, expr_label, au_label, expr_mask, au_mask

    def get_labels(self):
        return self.expr_labels


def union_data_loader(conf):
    # au_mask: 0, not labeled; 1, labeled
    # expr_mask
    img_folder = conf.train_data_path
    expr_annot_path = conf.expr_train_annot_path
    au_annot_path = '/home/tianqing/Face/au_train_list(without-1).npy'
    au_annots_samples = np.load(au_annot_path, allow_pickle=True)

    # transform au annot into dict
    au_samples_dict = {}
    for sample in au_annots_samples:
        au_samples_dict[sample[0]] = sample[1:]

    imgs = []
    expr_labels = []
    au_labels = []
    expr_masks = []
    au_masks = []

    with open(expr_annot_path) as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            img_path = line[0]
            expr_label = int(line[1])
            if os.path.exists(os.path.join(img_folder, img_path)):
                imgs.append(os.path.join(img_folder, img_path))
                if expr_label == -1:
                    expr_labels.append(0)
                    expr_masks.append(0)
                else:
                    expr_labels.append(expr_label)
                    expr_masks.append(1)
                if img_path in au_samples_dict:
                    labels = au_samples_dict[img_path]
                    if -1 in labels:
                        au_labels.append([0] * 12)
                        au_masks.append(0)
                    else:
                        au_labels.append(labels)
                        au_masks.append(1)
                    del au_samples_dict[img_path]
                else:
                    au_labels.append([0] * 12)
                    au_masks.append(0)

    for img_path in au_samples_dict:
        labels = au_samples_dict[img_path]
        if -1 in labels:
            au_labels.append([0] * 12)
            au_masks.append(0)
        else:
            au_labels.append(labels)
            au_masks.append(1)
        imgs.append(os.path.join(img_folder, img_path))
        expr_labels.append(0)
        expr_masks.append(0)

    ds = UnionDataset(imgs, expr_labels, au_labels, expr_masks, au_masks)
    print('train data loader generated')
    loader = DataLoader(ds, batch_size=conf.batch_size,
                        shuffle=True, pin_memory=conf.pin_memory,
                        num_workers=conf.num_workers, drop_last=True)
    return loader


if __name__ == '__main__':
    conf = get_config()
    loader = union_data_loader(conf)
    for imgs, expr_labels, au_labels, expr_masks, au_masks in iter(loader):
        print(imgs)
        print(expr_labels)
        print(au_labels)
        print(expr_masks)
        print(au_masks)

        # print(Counter(labels.numpy().tolist()))
