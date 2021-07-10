from core.lib.helper_new import get_im_dict, get_label_landmark, train_test_split
from torch.utils.data import Dataset
from core.audio.clip_transforms import *
from core.audio.utils import *
import subprocess
import math
import torchaudio
import pickle
from tqdm import tqdm
from PIL import Image
import random
import json
from matplotlib import pyplot as plt
import torchvision
import torch.utils.data
from typing import Callable
import csv
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans
from PIL import Image, ImageFile
from torchvision.transforms import autoaugment
from torchvision.transforms.transforms import ColorJitter, RandomResizedCrop, Resize
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import pandas as pd
# from audio.video import Video


AU_list = [1, 2, 4, 6, 7, 10,  12,  15, 23, 24, 25, 26]


class Aff2ComAUDataset(Dataset):
    def __init__(self, transform=None, flag='train', withBP4D=False, is_random=False, seq_num=30):
        super(Aff2ComAUDataset, self).__init__()
        self._flag = flag
        self.aus = [1, 2, 4, 6, 7, 10,  12,  15, 23, 24, 25, 26]
        self.au_names = [f'AU{au:0>2}' for au in self.aus]
        self._transform = transform
        self.im_dic, self.im_dic_vaild = get_im_dict()
        self.train_person, self.test_person = list(
            self.im_dic.keys()), list(self.im_dic_vaild.keys())
        # self.person_batch = person_batch
        self.seq_num = seq_num

        # audio params
        self.window_size = 10e-3
        self.window_stride = 5e-3
        self.sample_rate = 44100
        num_fft = 2 ** math.ceil(math.log2(self.window_size *
                                 self.sample_rate))
        window_fn = torch.hann_window

        # audio transforms
        self.audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=64,
                                                                    n_fft=num_fft,
                                                                    win_length=int(
                                                                        self.window_size * self.sample_rate),
                                                                    hop_length=int(self.window_stride
                                                                                   * self.sample_rate),
                                                                    window_fn=window_fn)

        self.audio_spec_transform = ComposeWithInvert(
            [AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])

        # Genrate training img dataset
        if flag == 'train':
            print("Generate traning dataset!")
            if not withBP4D:
                print("Without BP4D!")
                drop_list = []
                for i in self.train_person:
                    if i[0] == 'M' or i[0] == 'F':
                        drop_list.append(i)
                for i in drop_list:
                    self.train_person.remove(i)
            else:
                print("With BP4D!")
                # the json file is generated in ./lib/helper_new:get_label_landmark
            with open("./data/train_labels.json", 'r', encoding='UTF-8') as f:
                load_dict = json.load(f)
            self._labels = load_dict
            # self._train_list =  np.load('./data/train_select_list_30.npy',allow_pickle=True)

            # is_random = True means Randomly select video frames
            if is_random:
                print("Randomly select video frames")
                img_paths = []
                for p in self.train_person:
                    random.shuffle(self.im_dic[p])
                    p_frames, len_p = self.im_dic[p], len(self.im_dic[p])
                    for iii in range(int(len_p/seq_num)+1):
                        if (len_p % seq_num != 0) and (iii == int(len_p/seq_num)):
                            mod_ = len_p % seq_num
                            if p[0] == "F" or p[0] == "M":
                                selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(
                                    kk, '04d')+'.jpg' for kk in sorted(p_frames[-1 * seq_num:])]
                            else:
                                # the path need to modify
                                selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/' + p + '/' + format(
                                    kk, '05d') + '.jpg' for kk in sorted(p_frames[-1 * seq_num:])]
                            # if len(selected_p[-1 * mod_:]==100):
                            if len(selected_p) == seq_num:
                                img_paths.append(selected_p)
                        elif iii == int(len_p/seq_num):
                            continue
                        else:
                            if p[0] == "F" or p[0] == "M":
                                selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(
                                    kk, '04d')+'.jpg' for kk in sorted(p_frames[iii*seq_num:(iii+1)*seq_num])]
                            else:
                                selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(
                                    kk, '05d')+'.jpg' for kk in sorted(p_frames[iii*seq_num:(iii+1)*seq_num])]
                            if len(selected_p) == seq_num:
                                img_paths.append(selected_p)

            else:
                img_paths = []
                for p in self.train_person:
                    # print('test: ', p, ' ', count, 'of', len(train_person))
                    p_frames, len_p = self.im_dic[p], len(self.im_dic[p])
                    # iii = 0
                    iii = random.randint(0, 5)
                    while iii+seq_num < len_p:
                        if p[0] == "F" or p[0] == "M":
                            selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(
                                kk, '04d')+'.jpg' for kk in p_frames[iii:(iii+seq_num)]]
                        else:
                            selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(
                                kk, '05d')+'.jpg' for kk in p_frames[iii: iii+seq_num]]
                        if len(selected_p) == seq_num:
                            img_paths.append(selected_p)
                        iii += 10
                    # # for iii in range(int(len_p/seq_num)+1):
                    #     if (len_p%seq_num != 0) and (iii==int(len_p/seq_num)):
                    #         mod_ = len_p % seq_num
                    #         #the path need to modify
                    #         if p[0]=="F" or p[0]=="M":
                    #             selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(kk,'04d')+'.jpg' for kk in sorted(p_frames[-1 * seq_num:])]
                    #         else:
                    #             selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+ p + '/' + format(kk,'05d') + '.jpg' for kk in p_frames[-1 * seq_num:]]
                    #         # if len(selected_p[-1 * mod_:]==100):
                    #         # img_paths.append(selected_p[-1 * mod_:])
                    #         if len(selected_p)==seq_num:
                    #             img_paths.append(selected_p)
                    #     elif iii==int(len_p/seq_num):
                    #         continue
                    #     else:
                    #         if p[0]=="F" or p[0]=="M":
                    #             selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(kk,'04d')+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                    #         else:
                    #             selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(kk,'05d')+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                    #         if len(selected_p)==seq_num:
                    #             img_paths.append(selected_p)
                        # img_paths.append(np.array(selected_p))
            # you can save the _train_list and load it next time
            # np.save('./data/train_select_list_50.npy',np.array(img_paths))
            # self._train_list =  np.load('./data/train_select_list_50.npy',allow_pickle=True)
            self._train_list = np.array(img_paths)

        # Genrate testing img dataset
        else:
            with open("./data/valid_labels.json", 'r', encoding='UTF-8') as f:
                load_dict = json.load(f)
            self._labels_valid = load_dict
            img_paths = []
            for p in self.test_person:
                p_frames, len_p = self.im_dic_vaild[p], len(
                    self.im_dic_vaild[p])
                for iii in range(int(len_p/seq_num)+1):
                    if (len_p % seq_num != 0) and (iii == int(len_p/seq_num)):
                        mod_ = len_p % seq_num
                        temp = p_frames[-1 * seq_num:]
                        # temp.sort()
                        selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/' +
                                      p + '/' + format(kk, '05d') + '.jpg' for kk in temp]
                        if len(selected_p) == seq_num:
                            img_paths.append(selected_p)
                    elif iii == int(len_p/seq_num):
                        continue
                    else:
                        # selected_p = [p+'_'+str(kk)+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                        selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(
                            kk, '05d')+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                        if len(selected_p) == seq_num:
                            img_paths.append(np.array(selected_p))
            # you can save the _train_list and load it next time
            self._valid_list = np.array(img_paths)

    def _check_dataset(self):
        print('pass')

    def __len__(self):
        if self._flag == 'train':
            return self._train_list.shape[0]
        else:
            return self._valid_list.shape[0]

    def imdata(self, im_paths, is_training=True):
        # print(im_paths)
        data_ = torch.zeros(len(im_paths), 3, 112, 112)
        labels_ = torch.zeros(len(im_paths), 12)
        for i in range(len(im_paths)):
            im_path = im_paths[i]
            # print(im_path)
            with Image.open(im_path) as img:
                img = img.convert('RGB')
            img = self._transform(img)
            data_[i, :, :, :] = img
            if is_training:
                labels_[i, :] = torch.tensor(self._labels[im_path])
            else:
                labels_[i, :] = torch.tensor(
                    self._labels_valid[im_path.split('/')[-2]+'/'+im_path.split('/')[-1]])
        return data_, labels_

    def audiodata(self, selected_p):
        data_ = torch.zeros(selected_p.shape[0], 2, 64, 8)
        frame_list = []
        for i in range(selected_p.shape[0]):
            frame_list.append(int(selected_p[i].split('/')[-1][:-4]))

        start_frame = frame_list[0]
        end_frame = frame_list[-1]
        subject = selected_p[0].split('/')[-2]  # video name
        audio_file = '/data/tian/Data/Aff-Wild2-audio/audio/' + subject + '.wav'
        if not os.path.exists(audio_file):
            # print('file not exist:',audio_file)
            audio_file = audio_file.split('_')[0]+'.wav'

        # self.sample_rate = 44100
        audio, sample_rate = torchaudio.load(audio_file)
        #[2,44100]
        audio = audio[:, int(self.sample_rate*start_frame/30):int(self.sample_rate*(end_frame+2)/30)]
        audio_features = self.audio_transform(audio).detach()  # [2,64,201]
        if self.audio_spec_transform is not None:
            audio_features = self.audio_spec_transform(audio_features)

        
        #2,64,8
        for i in range(selected_p.shape[0]):
            if int((frame_list[i]-frame_list[0])*200/30)+8 > audio_features.shape[2]-1:
                data_[i, :, :, :] = audio_features[:,:,-8:]
            else:
                data_[i, :, :, :] = audio_features[:,:,
                                                int((frame_list[i]-frame_list[0])*200/30): int((frame_list[i]-frame_list[0])*200/30)+8]
            if data_[i, :, :, :].shape[2]<8:
                data_[i, :, :, :] = audio_features[:,:,-8:]

        #(30,2,64,8)
        return data_

    def __getitem__(self, index):
        # row = self._samples.loc[index]
        # fname = row['filename']
        data = {'Index': index}
        if self._flag == 'train':
            selected_p = self._train_list[index]
            im_data, im_labels = self.imdata(selected_p)
            # return im_data,im_labels
        else:
            selected_p = self._valid_list[index]
            im_data, im_labels = self.imdata(selected_p, is_training=False)
        data['img'] = im_data  # (30,3,112,112)
        data['label'] = im_labels #(30,12)
        data['audio'] = self.audiodata(selected_p) #(30,2,64,8)
        # print(data['audio'].shape)

        # start_frame = int(selected_p[0].split('/')[-1][:-4])
        # end_frame = int(selected_p[-1].split('/')[-1][:-4])
        # subject = selected_p[0].split('/')[-2]

        return data


class AUDataset(Dataset):
    def __init__(self, image_dir=None, txt_file=None, transform=None, flag='train', withBP4D=False, is_random=False, seq_num=30):
        self._image_dir = image_dir
        self._flag = flag
        self.aus = [1, 2, 4, 6, 7, 10,  12,  15, 23, 24, 25, 26]
        self.au_names = [f'AU{au:0>2}' for au in self.aus]
        self._transform = transform
        self.im_dic, self.im_dic_vaild = get_im_dict()
        self.train_person, self.test_person = list(
            self.im_dic.keys()), list(self.im_dic_vaild.keys())
        # self.person_batch = person_batch
        self.seq_num = seq_num

        # Genrate training dataset
        if flag == 'train':
            print("Generate traning dataset!")
            if not withBP4D:
                print("Without BP4D!")
                drop_list = []
                for i in self.train_person:
                    if i[0] == 'M' or i[0] == 'F':
                        drop_list.append(i)
                for i in drop_list:
                    self.train_person.remove(i)
            else:
                print("With BP4D!")
                # the json file is generated in ./lib/helper_new:get_label_landmark
            with open("./data/train_labels.json", 'r', encoding='UTF-8') as f:
                load_dict = json.load(f)
            self._labels = load_dict
            # self._train_list =  np.load('./data/train_select_list_30.npy',allow_pickle=True)

            # is_random = True means Randomly select video frames
            if is_random:
                print("Randomly select video frames")
                img_paths = []
                for p in self.train_person:
                    random.shuffle(self.im_dic[p])
                    p_frames, len_p = self.im_dic[p], len(self.im_dic[p])
                    for iii in range(int(len_p/seq_num)+1):
                        if (len_p % seq_num != 0) and (iii == int(len_p/seq_num)):
                            mod_ = len_p % seq_num
                            if p[0] == "F" or p[0] == "M":
                                selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(
                                    kk, '04d')+'.jpg' for kk in sorted(p_frames[-1 * seq_num:])]
                            else:
                                # the path need to modify
                                selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/' + p + '/' + format(
                                    kk, '05d') + '.jpg' for kk in sorted(p_frames[-1 * seq_num:])]
                            # if len(selected_p[-1 * mod_:]==100):
                            if len(selected_p) == seq_num:
                                img_paths.append(selected_p)
                        elif iii == int(len_p/seq_num):
                            continue
                        else:
                            if p[0] == "F" or p[0] == "M":
                                selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(
                                    kk, '04d')+'.jpg' for kk in sorted(p_frames[iii*seq_num:(iii+1)*seq_num])]
                            else:
                                selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(
                                    kk, '05d')+'.jpg' for kk in sorted(p_frames[iii*seq_num:(iii+1)*seq_num])]
                            if len(selected_p) == seq_num:
                                img_paths.append(selected_p)

            else:
                img_paths = []
                for p in self.train_person:
                    # print('test: ', p, ' ', count, 'of', len(train_person))
                    p_frames, len_p = self.im_dic[p], len(self.im_dic[p])
                    # iii = 0
                    iii = random.randint(0, 5)
                    while iii+seq_num < len_p:
                        if p[0] == "F" or p[0] == "M":
                            selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(
                                kk, '04d')+'.jpg' for kk in p_frames[iii:(iii+seq_num)]]
                        else:
                            selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(
                                kk, '05d')+'.jpg' for kk in p_frames[iii: iii+seq_num]]
                        if len(selected_p) == seq_num:
                            img_paths.append(selected_p)
                        iii += 5
                    # # for iii in range(int(len_p/seq_num)+1):
                    #     if (len_p%seq_num != 0) and (iii==int(len_p/seq_num)):
                    #         mod_ = len_p % seq_num
                    #         #the path need to modify
                    #         if p[0]=="F" or p[0]=="M":
                    #             selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(kk,'04d')+'.jpg' for kk in sorted(p_frames[-1 * seq_num:])]
                    #         else:
                    #             selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+ p + '/' + format(kk,'05d') + '.jpg' for kk in p_frames[-1 * seq_num:]]
                    #         # if len(selected_p[-1 * mod_:]==100):
                    #         # img_paths.append(selected_p[-1 * mod_:])
                    #         if len(selected_p)==seq_num:
                    #             img_paths.append(selected_p)
                    #     elif iii==int(len_p/seq_num):
                    #         continue
                    #     else:
                    #         if p[0]=="F" or p[0]=="M":
                    #             selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(kk,'04d')+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                    #         else:
                    #             selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(kk,'05d')+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                    #         if len(selected_p)==seq_num:
                    #             img_paths.append(selected_p)
                        # img_paths.append(np.array(selected_p))
            # you can save the _train_list and load it next time
            # np.save('./data/train_select_list_50.npy',np.array(img_paths))
            # self._train_list =  np.load('./data/train_select_list_50.npy',allow_pickle=True)
            self._train_list = np.array(img_paths)

        # Genrate testing dataset
        else:
            with open("./data/valid_labels.json", 'r', encoding='UTF-8') as f:
                load_dict = json.load(f)
            self._labels_valid = load_dict
            img_paths = []
            for p in self.test_person:
                p_frames, len_p = self.im_dic_vaild[p], len(
                    self.im_dic_vaild[p])
                for iii in range(int(len_p/seq_num)+1):
                    if (len_p % seq_num != 0) and (iii == int(len_p/seq_num)):
                        mod_ = len_p % seq_num
                        temp = p_frames[-1 * seq_num:]
                        # temp.sort()
                        selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/' +
                                      p + '/' + format(kk, '05d') + '.jpg' for kk in temp]
                        if len(selected_p) == seq_num:
                            img_paths.append(selected_p)
                    elif iii == int(len_p/seq_num):
                        continue
                    else:
                        # selected_p = [p+'_'+str(kk)+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                        selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(
                            kk, '05d')+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                        if len(selected_p) == seq_num:
                            img_paths.append(np.array(selected_p))
            # you can save the _train_list and load it next time
            self._valid_list = np.array(img_paths)

    def _check_dataset(self):
        print('pass')

    def __len__(self):
        if self._flag == 'train':
            return self._train_list.shape[0]
        else:
            return self._valid_list.shape[0]

    def imdata(self, im_paths, is_training=True):
        # print(im_paths)
        data_ = torch.zeros(len(im_paths), 3, 112, 112)
        labels_ = torch.zeros(len(im_paths), 12)
        for i in range(len(im_paths)):
            im_path = im_paths[i]
            # print(im_path)
            with Image.open(im_path) as img:
                img = img.convert('RGB')
            img = self._transform(img)
            data_[i, :, :, :] = img
            if is_training:
                labels_[i, :] = torch.tensor(self._labels[im_path])
            else:
                labels_[i, :] = torch.tensor(
                    self._labels_valid[im_path.split('/')[-2]+'/'+im_path.split('/')[-1]])
        return data_, labels_

    def __getitem__(self, index):
        # row = self._samples.loc[index]
        # fname = row['filename']
        if self._flag == 'train':
            selected_p = self._train_list[index]
            im_data, im_labels = self.imdata(selected_p)
            return im_data, im_labels
        if self._flag != 'train':
            selected_p = self._valid_list[index]
            im_data, im_labels = self.imdata(selected_p, is_training=False)
            return im_data, im_labels


def get_dataloader(image_dir=None, txt_file=None, batch_size=3, is_training=True, flag='train', withBP4D=False, is_random=False, seq_num=30):
    if is_training:
        transform = trans.Compose([
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
    return DataLoader(Aff2ComAUDataset(image_dir=None, txt_file=None, transform=transform, flag=flag, withBP4D=withBP4D, is_random=is_random, seq_num=seq_num),
                      batch_size=batch_size,
                      shuffle=is_training,
                      pin_memory=True,
                      num_workers=16,
                      drop_last=is_training)


def test_trainset():
    # image_dir = '/data/jinyue/dataset/opensource/AffWild2/cropped_aligned'
    # txt_file ='/data/jinyue/dataset/opensource/AffWild2/Annotations/au_annots_train.csv'
    batch_size = 3
    transform = trans.Compose([
        # trans.Resize((112, 112)),
        trans.RandomResizedCrop((112, 112), scale=(0.6, 1.0)),
        trans.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.3),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    emtiondata = Aff2ComAUDataset(
        image_dir=None, txt_file=None, transform=transform, flag='train')
    # print(len(emtiondata))
    img = emtiondata.__getitem__(1)
    # print(img[0].shape)
    # print(img[0])
    # # print(img[0].shape)
    # print(type(img))


# merge_two_datasets()
if __name__ == '__main__':
    test_trainset()
