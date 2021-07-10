import os
import csv
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms as trans
from torch.utils.data import Dataset, DataLoader
# from core.modelimg import IrResNetSe
from core.model_test import TwoStreamAUAttentionModelTest, TwoStreamFusionTDNNTest
from core.lib.helper_new import get_im_dict, get_label_landmark, train_test_split
from core.audio.clip_transforms import *
from core.audio.utils import *
# from models.iresnetse import IrResNetSe
# from models.iresnet import iresnet100
# from models.iresnet_2branches import iresnet100_2branches
import math
import torchaudio
import torch.nn.functional as F
from core.lib.iresnet_2branches import iresnet100_2branches_attention, iresnet100_2branches_attention_add


class TestSeqDataset(Dataset):
    def __init__(self, fps=30,cropped_aligned_images=None, ids=None, flag='test', withBP4D=False, is_random=False, seq_num=30,start=0):
        super(TestSeqDataset, self).__init__()
        self.img_list = cropped_aligned_images
        self.img_ids = ids
        self.fps = fps

        self._transform = trans.Compose([
            trans.Resize((112, 112)),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


        self._flag = flag
        self.aus = [1, 2, 4, 6, 7, 10,  12,  15, 23, 24, 25, 26]
        self.au_names = [f'AU{au:0>2}' for au in self.aus]

        # self.im_dic, self.im_dic_vaild = get_im_dict()
        # self.train_person, self.test_person = list(
        #     self.im_dic.keys()), list(self.im_dic_vaild.keys())
        # self.person_batch = person_batch
        self.seq_num = seq_num
        # audio params
        self.window_size = 10e-3
        self.window_stride = 5e-3
        self.sample_rate = 44100
    
        img_paths = []
        img_ids = []
        len_p = len(cropped_aligned_images)

        # start = start
        for iii in range(int(len_p/seq_num)+1):
            if (len_p % seq_num != 0) and (iii == int(len_p/seq_num)):
                mod_ = len_p % seq_num
                selected_p = cropped_aligned_images[-1 * seq_num:]
                selected_ids = self.img_ids[-1 * seq_num:]
                if len(selected_p) == seq_num:
                    img_paths.append(selected_p)
            elif iii == int(len_p/seq_num):
                continue
            else:
                selected_p = cropped_aligned_images[start+iii*seq_num:start+(iii+1)*seq_num]
                selected_ids = self.img_ids[start+iii*seq_num:start+(iii+1)*seq_num]
                if len(selected_p) == seq_num:
                    img_paths.append(np.array(selected_p))
                    img_ids.append(np.array(selected_ids))

        # you can save the _train_list and load it next time
        self._valid_list = np.array(img_paths)
        self._valid_ids = np.array(img_ids)

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
            # else:
            #     labels_[i, :] = torch.tensor(
            #         self._labels_valid[im_path.split('/')[-2]+'/'+im_path.split('/')[-1]])
        # return data_, labels_
        return data_

    def __getitem__(self, index):
        # row = self._samples.loc[index]
        # fname = row['filename']
        data = {'Index': index}
        # print(index)
        if self._flag == 'train':
            selected_p = self._train_list[index]
            im_data, im_labels = self.imdata(selected_p)
            data['img'] = im_data  # (30,3,112,112)
            data['label'] = im_labels  # (30,12)
            data['audio'] = self.audiodata(selected_p)  # (30,2,64,8)
            # return im_data,im_labels
        else:
            selected_p = self._valid_list[index]
            # im_data, im_labels = self.imdata(selected_p, is_training=False)
            im_data = self.imdata(selected_p, is_training=False)
            # print(selected_p)
            data['img'] = im_data  # (30,3,112,112)
            frame_list = []
            for i in range(selected_p.shape[0]):
                frame_list.append(int(selected_p[i].split('/')[-1][:-4]))
            frame_list = torch.Tensor(frame_list)
            # print(data['img'],data['img'].shape)
            # data['audio'],ids = self.audiodata(selected_p)  # (30,2,64,8)
        # print(data['audio'].shape)

        # start_frame = int(selected_p[0].split('/')[-1][:-4])
        # end_frame = int(selected_p[-1].split('/')[-1][:-4])
        # subject = selected_p[0].split('/')[-2]

        return data,frame_list


def test_seq_data_loader(cropped_aligned_images, ids,fps,start):
    ds = TestSeqDataset(fps=fps,cropped_aligned_images=cropped_aligned_images,
                        ids=ids,start=start)
    loader = DataLoader(ds, batch_size=2,
                        shuffle=False, pin_memory=True,
                        num_workers=16, drop_last=False)
    return loader

#results/au-twobranch-imgseq/AU_Set/2-30-640x360.txt
class Inference():
    def __init__(self, output_exp_name, model_path,start):
        self.start = start
        self.video_fps_list = np.load('/data/tian/Data/ABAW2_Competition/video_fps_dic.npy',allow_pickle=True).item()
        self.cropped_aligned_img_folder = '/data/tian/Data/ABAW2_Competition/cropped_aligned'
        self.au_test_list_file = '/data/tian/developer/AU/ABAW2_AU/data/test_set_AU_Challenge.txt'
        self.expr_test_list_file = '/data/tian/developer/AU/ABAW2_AU/data/test_set_Expr_Challenge.txt'
        self.device = torch.device(
            "cuda:3" if torch.cuda.is_available() else "cpu")
        self.frame_dict = np.load(
            '/data/tian/developer/AU/ABAW2_AU/data/frame_dict.npy', allow_pickle=True).item()
        # self.output_exp = 'au_submit1'
        self.output_exp = output_exp_name
        self.output_results = os.path.join('results', self.output_exp)
        if not os.path.exists(self.output_results):
            os.makedirs(self.output_results)
        self.output_logits = os.path.join('logits', self.output_exp)
        if not os.path.exists(self.output_logits):
            os.makedirs(self.output_logits)
        # self.model = TwoStreamFusionTDNNTest(seq_num=30, encoder_layers=1)
        self.model = iresnet100_2branches_attention_add(seq_num=30,encoder_layers=1)
        # model_path = '/data/tian/developer/output/model_2021-06-17-17-16_acc:0.8787685929713989_f1:0.5328100042602213_bacth:1_step:12053.pth'
        # model_path = '/home/cpicapp/jinyue/emotion/ABAW2_logs/swap_epoch_v2/save/model_expr_2021-07-06-17-11_loss-0.015_acc-0.689_f1-0.479_step-10596_epoch-0.pth'
        # model_path = '/data/tian/developer/ROI-Nets-pytorch/models/tdnn/6.1.1_tdnn_SE50_30_1_lr0.01/score_0.7105params_epoch_2_acc_0.8792_f1_0.5419.pkl'
        self.model.load_state_dict(torch.load(model_path))
        # ,map_location='cuda:5')
        self.model=self.model.to(self.device)
        self.model.eval()
        self.transform = trans.Compose([
            trans.Resize((112, 112)),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def get_all_videos(self):
        video_list1 = []
        video_list2 = []
        if os.path.exists(self.video_folder1):
            video_list1 = [os.path.join(self.video_folder1, video)
                           for video in os.listdir(self.video_folder1)]
        if os.path.exists(self.video_folder2):
            video_list2 = [os.path.join(self.video_folder2, video)
                           for video in os.listdir(self.video_folder2)]
        return video_list1 + video_list2

    def get_test_videos(self, test_list_file):
        video_list = self.get_all_videos()
        test_video_list = []
        with open(test_list_file, 'r') as f:
            for line in f.readlines():
                video = line.strip() + '.mp4'
                if os.path.exists(os.path.join(self.video_folder1, video)):
                    test_video_list.append(
                        os.path.join(self.video_folder1, video))
                if os.path.exists(os.path.join(self.video_folder2, video)):
                    test_video_list.append(
                        os.path.join(self.video_folder2, video))
                video = line.strip() + '.avi'
                if os.path.exists(os.path.join(self.video_folder1, video)):
                    test_video_list.append(
                        os.path.join(self.video_folder1, video))
                if os.path.exists(os.path.join(self.video_folder2, video)):
                    test_video_list.append(
                        os.path.join(self.video_folder2, video))
        return test_video_list

    def get_test_videos_from_crop(self, test_list_file):
        video_list = []
        with open(test_list_file, 'r') as f:
            for line in f.readlines():
                video = line.strip()
                video_list.append(video)
        return video_list

    def get_largest_frame(self, image_list):
        num_list = [int(os.path.splitext(x)[0]) for x in image_list]
        return max(num_list)

    def inference_au(self):
        # au_test_video_list = self.get_test_videos(self.au_test_list_file)
        au_test_video_list = self.get_test_videos_from_crop(
            self.au_test_list_file)
        for video in au_test_video_list:
            subfolder = os.path.join(
                self.cropped_aligned_img_folder, os.path.splitext(os.path.basename(video))[0])
            cropped_aligned_images = [os.path.join(
                subfolder, x)for x in os.listdir(subfolder)]
            largest_frames = self.get_largest_frame(os.listdir(subfolder))
            results = [None] * (largest_frames + 1)
            results[0] = 'AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n'
            for image in tqdm(cropped_aligned_images):
                img = Image.open(image)
                img = self.transform(img)
                img = img.unsqueeze(dim=0).to(self.device)
                logits = self.model(img)[1]
                pred_au = torch.sigmoid(logits).cpu().detach().numpy()
                pred_au_01 = [[1 if x > 0.5 else 0 for x in y]
                              for y in pred_au]
                results[int(os.path.splitext(os.path.basename(image))[0])] = str(
                    pred_au_01[0]).replace('[', '').replace(']', '').replace(' ', '') + '\n'

            for idx, item in enumerate(results):
                if item == None:
                    results[idx] = '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n'

            results[-1].strip()

            # write output file
            if not os.path.exists(os.path.join(self.output_results, 'AU_Set')):
                os.makedirs(os.path.join(self.output_results, 'AU_Set'))
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(
                self.output_results, 'AU_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(results)
            f_write.close()

    def inference_expr(self):
        # expr_test_video_list = self.get_test_videos(self.expr_test_list_file)
        expr_test_video_list = self.get_test_videos_from_crop(
            self.expr_test_list_file)
        for video in expr_test_video_list:
            subfolder = os.path.join(
                self.cropped_aligned_img_folder, os.path.splitext(os.path.basename(video))[0])
            cropped_aligned_images = [os.path.join(
                subfolder, x)for x in os.listdir(subfolder)]
            largest_frames = self.get_largest_frame(os.listdir(subfolder))
            results = [None] * (largest_frames + 1)
            results[0] = 'Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n'
            for image in tqdm(cropped_aligned_images):
                img = Image.open(image)
                img = self.transform(img)
                img = img.unsqueeze(dim=0).to(self.device)
                logits = self.model(img)[0]
                pred_expr_softmax = torch.softmax(
                    logits, dim=1).cpu().detach().numpy()
                pred_expr = pred_expr_softmax.argmax(axis=1)
                results[int(os.path.splitext(os.path.basename(image))[0])] = str(
                    pred_expr[0]) + '\n'

            for idx, item in enumerate(results):
                if item == None:
                    results[idx] = '-1\n'

            results[-1].strip()

            # write output file
            if not os.path.exists(os.path.join(self.output_results, 'EXPR_Set')):
                os.makedirs(os.path.join(self.output_results, 'EXPR_Set'))
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(
                self.output_results, 'EXPR_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(results)
            f_write.close()

    def inference_au_batch(self):
        # au_test_video_list = self.get_test_videos(self.au_test_list_file)
        au_test_video_list = self.get_test_videos_from_crop(
            self.au_test_list_file)
        for video in tqdm(au_test_video_list,ascii=True):
            print(video)
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(
                self.output_results, 'AU_Set', output_file)
            if os.path.exists(output_file_path):
                print(output_file_path)
                continue
            if video.split('_')[-1]=='left' or video.split('_')[-1]=='right':
                video_name = video.split('_')[0]
                fps = self.video_fps_list[video_name]
            else:
                fps = self.video_fps_list[video]

            print(video,' ',fps)
            # if round(fps)==30:
            #     continue
            subfolder = os.path.join(
                self.cropped_aligned_img_folder, os.path.splitext(os.path.basename(video))[0])
            file_list = os.listdir(subfolder)
            file_list.sort()
            # if '.DS_Store' in file_list:
            #     file_list.remove('.DS_Store')
            cropped_aligned_images = [
                os.path.join(subfolder, x)for x in file_list]
            ids = [int(os.path.splitext(os.path.basename(x))[0])
                   for x in file_list]
            tmp = os.path.splitext(os.path.basename(video))[0]
            largest_frames = max(
                int(self.frame_dict[tmp]) + 1, self.get_largest_frame(os.listdir(subfolder)))
            results = [None] * (largest_frames + 1)
            results[0] = 'AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n'
            logits_results = [None] * (largest_frames + 1)
            logits_results[0] = 'AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n'

            test_loader = test_seq_data_loader(cropped_aligned_images, ids,fps,start)

            for data,ids in tqdm(iter(test_loader),ascii=True):
                # print("%%%%%%%%%%%")
                im_data = data['img'].reshape(-1,3, 112, 112).to(self.device)
                # print(im_data.shape)
                # audio_data = data['audio'].to(self.device)
                # # print(audio_data.shape)
                # audio_data = audio_data[:,0,:,:]+audio_data[:,1,:,:]
                # audio_data = torch.swapaxes(audio_data,1,2)
                # print(audio_data.shape)
                # logits = self.model(imgs.to(self.device))[1]
                # logits = self.model(im_data, audio_data,frame_list=ids,batch_size=2)
                logits = self.model(im_data)
                # print(logits)
                # print(ids)
                pred_au = logits.cpu().detach().numpy()
                # print(ids)
                # ids = ids.numpy().flatten().astype(int)
                # print(ids)
                ids = ids.cpu().detach().numpy().flatten().astype(int)
                # print(ids)
                # pred_au = torch.sigmoid(logits).cpu().detach().numpy()
                # print('pred_au:',pred_au,pred_au.shape)
                # print('ids:',ids,ids.shape)
                for pred, id in zip(pred_au, ids):
                    pred_au_01 = [1 if x > 0.5 else 0 for x in pred]
                    # pred_au_01 = [1 if x > 0 else 0 for x in pred]
                    results[id] = str(pred_au_01).replace(
                        '[', '').replace(']', '').replace(' ', '') + '\n'
                    write_str = ''
                    for item in pred:
                        write_str += (str(item) + ',')
                    logits_results[id] = write_str.strip(',') + '\n'
                # print(logits_results,results)
                # exit()

            for idx, item in enumerate(results):
                if item == None:
                    results[idx] = '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n'
            for idx, item in enumerate(logits_results):
                if item == None:
                    logits_results[idx] = '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n'

            results[-1].strip()
            logits_results[-1].strip()

            # write output file
            if not os.path.exists(os.path.join(self.output_results, 'AU_Set')):
                os.makedirs(os.path.join(self.output_results, 'AU_Set'))
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(
                self.output_results, 'AU_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(results)
            f_write.close()

            # write output file
            if not os.path.exists(os.path.join(self.output_logits, 'AU_Set')):
                os.makedirs(os.path.join(self.output_logits, 'AU_Set'))
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(
                self.output_logits, 'AU_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(logits_results)
            f_write.close()


if __name__ == '__main__':
    start = 0
    output_exp_name = 'au-twostream-start0'
    # path_list = '/data/tian/developer/output/pytorch_au/2.5.1_se50_lr0.001_pos0_dataset5_64/save/score_0.7315276637883353_optimizer_2021-07-08-08-47_acc:0.8956518948725117_f1:0.5674034327041589_bacth:3_step:0.pth'
    path_list = ''
    # model_path = '/data/tian/developer/output/best/img_seq/score_0.7103params_epoch_0_acc_0.8781_f1_0.5426.pkl'
    # model_path = '/data/tian/developer/ROI-Nets-pytorch/models/3.6.1.1/SE100_withBP4D_encoder1_seq30step5_premodel_frozen_lr0.01_pos_weight_(feature+encoder)/score_0.7508params_epoch_8_acc_0.8987_f1_0.6029.pkl'
    model_path = '/data/tian/developer/output/best/score_0.7120params_epoch_0_acc_0.8786_f1_0.5454.pkl'
    inference_module = Inference(output_exp_name, model_path,start)
    inference_module.inference_au_batch()

    
