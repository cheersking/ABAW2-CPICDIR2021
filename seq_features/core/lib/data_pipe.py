import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans
from PIL import Image, ImageFile
from torchvision.transforms import autoaugment
from torchvision.transforms.transforms import ColorJitter, RandomResizedCrop, Resize
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
import csv
from typing import Callable
# import pandas as pd
import torch
import torch.utils.data
import torchvision
# from matplotlib import pyplot as plt
import json
import random
from core.lib.helper_new import get_im_dict,get_label_landmark,train_test_split


AU_list = [1, 2, 4, 6, 7, 10,  12,  15, 23, 24, 25, 26]

class AUDataset(Dataset):
    def __init__(self, image_dir=None, txt_file=None, transform=None,flag='train',withBP4D=False,is_random=False,seq_num=30):
        self._image_dir = image_dir
        self._flag = flag
        self.aus = [1, 2, 4, 6, 7, 10,  12,  15, 23, 24, 25, 26]
        self.au_names = [f'AU{au:0>2}' for au in self.aus]
        self._transform = transform
        self.im_dic,self.im_dic_vaild = get_im_dict()
        self.train_person,self.test_person = list(self.im_dic.keys()),list(self.im_dic_vaild.keys())
        # self.person_batch = person_batch
        self.seq_num = seq_num

        #Genrate training dataset
        if flag=='train':
            print("Generate traning dataset!")
            if not withBP4D:
                print("Without BP4D!")
                drop_list = []
                for i in self.train_person:
                    if i[0]=='M' or i[0]=='F':
                        drop_list.append(i)
                for i in drop_list:
                    self.train_person.remove(i)
            else:
                print("With BP4D!")
                # the json file is generated in ./lib/helper_new:get_label_landmark
            with open("./data/train_labels.json",'r', encoding='UTF-8') as f:
                load_dict = json.load(f)
            self._labels = load_dict
                # self._train_list =  np.load('./data/train_select_list_30.npy',allow_pickle=True)
            
            #is_random = True means Randomly select video frames
            if is_random:
                print("Randomly select video frames")
                img_paths = []
                for p in self.train_person:
                    random.shuffle(self.im_dic[p])
                    p_frames,len_p = self.im_dic[p],len(self.im_dic[p])
                    for iii in range(int(len_p/seq_num)+1):
                        if (len_p%seq_num != 0) and (iii==int(len_p/seq_num)):
                            mod_ = len_p % seq_num
                            if p[0]=="F" or p[0]=="M":
                                selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(kk,'04d')+'.jpg' for kk in sorted(p_frames[-1 * seq_num:])]
                            else:
                            #the path need to modify
                                selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+ p + '/' + format(kk,'05d') + '.jpg' for kk in sorted(p_frames[-1 * seq_num:])]
                            # if len(selected_p[-1 * mod_:]==100):
                            if len(selected_p)==seq_num:
                                img_paths.append(selected_p)
                        elif iii==int(len_p/seq_num):
                            continue
                        else:
                            if p[0]=="F" or p[0]=="M":
                                selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(kk,'04d')+'.jpg' for kk in sorted(p_frames[iii*seq_num:(iii+1)*seq_num])]
                            else:
                                selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(kk,'05d')+'.jpg' for kk in sorted(p_frames[iii*seq_num:(iii+1)*seq_num])]
                            if len(selected_p)==seq_num:
                                img_paths.append(selected_p)                
    
            else:
                img_paths = []
                for p in self.train_person:
                    # print('test: ', p, ' ', count, 'of', len(train_person))
                    p_frames,len_p = self.im_dic[p],len(self.im_dic[p])
                    # iii = 0
                    iii = random.randint(0,10)
                    while iii+seq_num<len_p:
                        if p[0]=="F" or p[0]=="M":
                            selected_p = ['/data/tian/Data/BP4D-cropped-aligned/'+p+'/'+format(kk,'04d')+'.jpg' for kk in p_frames[iii:(iii+seq_num)]]
                        else:
                            selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(kk,'05d')+'.jpg' for kk in p_frames[iii: iii+seq_num]]
                        if len(selected_p)==seq_num:
                            img_paths.append(selected_p)
                        iii+=10
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
            #you can save the _train_list and load it next time
            # np.save('./data/train_select_list_50.npy',np.array(img_paths))
            # self._train_list =  np.load('./data/train_select_list_50.npy',allow_pickle=True)
            self._train_list = np.array(img_paths)
        
        #Genrate testing dataset
        else:
            with open("./data/valid_labels.json",'r', encoding='UTF-8') as f:
                load_dict = json.load(f)
            self._labels_valid = load_dict
            img_paths = []
            for p in self.test_person:
                p_frames,len_p = self.im_dic_vaild[p],len(self.im_dic_vaild[p])
                for iii in range(int(len_p/seq_num)+1):
                    if (len_p%seq_num != 0) and (iii==int(len_p/seq_num)):
                        mod_ = len_p % seq_num
                        temp = p_frames[-1 * seq_num:]
                        # temp.sort()
                        selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+ p + '/' + format(kk,'05d') + '.jpg' for kk in temp]
                        if len(selected_p)==seq_num:
                            img_paths.append(selected_p)
                    elif iii==int(len_p/seq_num):
                        continue
                    else:
                        # selected_p = [p+'_'+str(kk)+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                        selected_p = ['/data/tian/Data/ABAW2_Competition/cropped_aligned/'+p+'/'+format(kk,'05d')+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                        if len(selected_p)==seq_num:
                            img_paths.append(np.array(selected_p))
            #you can save the _train_list and load it next time
            self._valid_list = np.array(img_paths)

    def _check_dataset(self):
        print('pass')

    def __len__(self):
        if self._flag=='train':
            return self._train_list.shape[0]
        else:
            return self._valid_list.shape[0]

    def imdata(self,im_paths,is_training=True):
        # print(im_paths)
        data_ = torch.zeros(len(im_paths),3,112,112)
        labels_ = torch.zeros(len(im_paths),12)
        for i in range(len(im_paths)):
            im_path = im_paths[i]
            # print(im_path)
            with Image.open(im_path) as img:
                img = img.convert('RGB')
            img = self._transform(img)
            data_[i,:,:,:] = img
            if is_training:
                labels_[i,:] = torch.tensor(self._labels[im_path])
            else:
                labels_[i,:] = torch.tensor(self._labels_valid[im_path.split('/')[-2]+'/'+im_path.split('/')[-1]])
        return data_,labels_

    def __getitem__(self, index):
        # row = self._samples.loc[index]
        # fname = row['filename']
        if self._flag=='train':
            selected_p = self._train_list[index]
            im_data,im_labels = self.imdata(selected_p)
            return im_data,im_labels
        if self._flag!='train':
            selected_p = self._valid_list[index]
            im_data, im_labels = self.imdata(selected_p,is_training=False)
            return im_data,im_labels

def get_dataloader(image_dir=None, txt_file=None, batch_size=3, is_training=True,flag='train',withBP4D= False,is_random=False,seq_num=30):
    if is_training:
        transform = trans.Compose([
                trans.RandomResizedCrop((112, 112), scale=(0.6, 1.0)),
                trans.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    else:
        transform = trans.Compose([trans.Resize(112),
                                    trans.ToTensor(),
                                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])
    return  DataLoader(AUDataset(image_dir=None, txt_file=None, transform=transform,flag=flag,withBP4D = withBP4D,is_random=is_random,seq_num=seq_num),
                        batch_size = batch_size,
                        shuffle = is_training, 
                        pin_memory= True,
                        num_workers=32, 
                        drop_last=is_training)


def test_trainset():
    # image_dir = '/data/jinyue/dataset/opensource/AffWild2/cropped_aligned'
    # txt_file ='/data/jinyue/dataset/opensource/AffWild2/Annotations/au_annots_train.csv'
    batch_size = 3
    transform = trans.Compose([
                # trans.Resize((112, 112)),
                trans.RandomResizedCrop((112, 112), scale=(0.6, 1.0)),
                trans.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    emtiondata = AUDataset(image_dir=None, txt_file=None, transform = transform,flag='train')
    # print(len(emtiondata))
    img = emtiondata.__getitem__(1)
    print(img[0].shape)
    print(img[0])
    # print(img[0].shape)
    print(type(img))
# merge_two_datasets()
if __name__== '__main__':
    test_trainset()