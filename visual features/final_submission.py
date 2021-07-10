import os
import csv
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms as trans
from torch.utils.data import Dataset, DataLoader
from models.iresnetse import IrResNetSe
from models.iresnet import iresnet100
from models.iresnet_2branches import iresnet100_2branches


class TestDataset(Dataset):
    def __init__(self, imgs, ids):
        self.imgs = imgs
        self.ids = ids
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
        img = self.transform(img)
        id = self.ids[index]
        return img, id


def test_data_loader(cropped_aligned_images, ids):
    ds = TestDataset(cropped_aligned_images, ids)
    loader = DataLoader(ds, batch_size=64,
                        shuffle=False, pin_memory=True,
                        num_workers=4, drop_last=False)
    return loader


class Inference():
    def __init__(self):
        self.cropped_aligned_img_folder = '/mnt/data3/jinyue/dataset/opensource/AffWild2/cropped_aligned'
        self.video_folder1 = '/mnt/data3/jinyue/dataset/opensource/AffWild2/batch2'
        self.video_folder2 = '/mnt/data3/jinyue/dataset/opensource/AffWild2/batch1'
        # self.au_test_list_file = '/mnt/data3/jinyue/dataset/opensource/AffWild2/test_set_AU_Challenge.txt'
        # self.expr_test_list_file = '/mnt/data3/jinyue/dataset/opensource/AffWild2/test_set_Expr_Challenge.txt'
        self.au_test_list_file = 'valid2.txt'
        self.expr_test_list_file = 'valid2.txt'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.output_exp = 'swap_epoch_v2_valid_set2'
        self.output_results = os.path.join('results', self.output_exp)
        if not os.path.exists(self.output_results):
            os.makedirs(self.output_results)
        self.output_logits = os.path.join('logits', self.output_exp)
        if not os.path.exists(self.output_logits):
            os.makedirs(self.output_logits)
        # self.model = IrResNetSe(self.conf.net_depth, self.conf.drop_ratio,
        #                         self.conf.net_mode).to(self.device)
        # self.model = iresnet100(num_class=self.conf.class_num).to(self.device)
        # model_path = '/data/jinyue/logs/v2.3.3_IResnet100-dropout0.6_focal-pytorch_sgd0.001_basic+color_nosample-modeleval/save/model_2021-06-15-20-50_loss-0.010_acc-0.630_f1-0.402_step-21760.pth'
        self.model = iresnet100_2branches().to(self.device)
        model_path = 'model_expr_2021-07-06-17-11_loss-0.015_acc-0.689_f1-0.479_step-10596_epoch-0.pth'
        self.model.load_state_dict(torch.load(model_path))
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
            video_list1 = [os.path.join(self.video_folder1, video) for video in os.listdir(self.video_folder1)]
        if os.path.exists(self.video_folder2):
            video_list2 = [os.path.join(self.video_folder2, video) for video in os.listdir(self.video_folder2)]
        return video_list1 + video_list2

    def get_test_videos(self, test_list_file):
        video_list = self.get_all_videos()
        test_video_list = []
        with open(test_list_file, 'r') as f:
            for line in f.readlines():
                video = line.strip() + '.mp4'
                if os.path.exists(os.path.join(self.video_folder1, video)):
                    test_video_list.append(os.path.join(self.video_folder1, video))
                if os.path.exists(os.path.join(self.video_folder2, video)):
                    test_video_list.append(os.path.join(self.video_folder2, video))
                video = line.strip() + '.avi'
                if os.path.exists(os.path.join(self.video_folder1, video)):
                    test_video_list.append(os.path.join(self.video_folder1, video))
                if os.path.exists(os.path.join(self.video_folder2, video)):
                    test_video_list.append(os.path.join(self.video_folder2, video))
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
        au_test_video_list = self.get_test_videos_from_crop(self.au_test_list_file)
        for video in au_test_video_list:
            subfolder = os.path.join(self.cropped_aligned_img_folder, os.path.splitext(os.path.basename(video))[0])
            cropped_aligned_images = [os.path.join(subfolder, x)for x in os.listdir(subfolder)]
            largest_frames = self.get_largest_frame(os.listdir(subfolder))
            results = [None] * (largest_frames + 1)
            results[0] = 'AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n'
            for image in tqdm(cropped_aligned_images):
                img = Image.open(image)
                img = self.transform(img)
                img = img.unsqueeze(dim=0).to(self.device)
                logits = self.model(img)[1]
                pred_au = torch.sigmoid(logits).cpu().detach().numpy()
                pred_au_01 = [[1 if x > 0.5 else 0 for x in y] for y in pred_au]
                results[int(os.path.splitext(os.path.basename(image))[0])] = str(pred_au_01[0]).replace('[', '').replace(']', '').replace(' ', '') + '\n'

            for idx, item in enumerate(results):
                if item == None:
                    results[idx] = '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n'

            results[-1].strip()

            # write output file
            if not os.path.exists(os.path.join(self.output_results, 'AU_Set')):
                os.makedirs(os.path.join(self.output_results, 'AU_Set'))
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(self.output_results, 'AU_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(results)
            f_write.close()

    def inference_expr(self):
        # expr_test_video_list = self.get_test_videos(self.expr_test_list_file)
        expr_test_video_list = self.get_test_videos_from_crop(self.expr_test_list_file)
        for video in expr_test_video_list:
            subfolder = os.path.join(self.cropped_aligned_img_folder, os.path.splitext(os.path.basename(video))[0])
            cropped_aligned_images = [os.path.join(subfolder, x)for x in os.listdir(subfolder)]
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
                results[int(os.path.splitext(os.path.basename(image))[0])] = str(pred_expr[0]) + '\n'

            for idx, item in enumerate(results):
                if item == None:
                    results[idx] = '-1\n'

            results[-1].strip()

            # write output file
            if not os.path.exists(os.path.join(self.output_results, 'EXPR_Set')):
                os.makedirs(os.path.join(self.output_results, 'EXPR_Set'))
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(self.output_results, 'EXPR_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(results)
            f_write.close()

    def inference_au_batch(self):
        # au_test_video_list = self.get_test_videos(self.au_test_list_file)
        au_test_video_list = self.get_test_videos_from_crop(self.au_test_list_file)
        for video in tqdm(au_test_video_list):
            subfolder = os.path.join(self.cropped_aligned_img_folder, os.path.splitext(os.path.basename(video))[0])
            cropped_aligned_images = [os.path.join(subfolder, x)for x in os.listdir(subfolder)]
            ids = [int(os.path.splitext(os.path.basename(x))[0]) for x in os.listdir(subfolder)]
            largest_frames = self.get_largest_frame(os.listdir(subfolder))
            results = [None] * (largest_frames + 1)
            results[0] = 'AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n'
            logits_results = [None] * (largest_frames + 1)
            logits_results[0] = 'AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n'

            test_loader = test_data_loader(cropped_aligned_images, ids)
            for imgs, ids in iter(test_loader):
                logits = self.model(imgs.to(self.device))[1]
                pred_au = torch.sigmoid(logits).cpu().detach().numpy()
                for pred, id in zip(pred_au, ids):
                    pred_au_01 = [1 if x > 0.5 else 0 for x in pred]
                    results[id] = str(pred_au_01).replace('[', '').replace(']', '').replace(' ', '') + '\n'
                    write_str = ''
                    for item in pred:
                        write_str += (str(item) + ',')
                    logits_results[id] = write_str.strip(',') + '\n'

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
            output_file_path = os.path.join(self.output_results, 'AU_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(results)
            f_write.close()

            # write output file
            if not os.path.exists(os.path.join(self.output_logits, 'AU_Set')):
                os.makedirs(os.path.join(self.output_logits, 'AU_Set'))
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(self.output_logits, 'AU_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(logits_results)
            f_write.close()

    def inference_expr_batch(self):
        # expr_test_video_list = self.get_test_videos(self.expr_test_list_file)
        expr_test_video_list = self.get_test_videos_from_crop(self.expr_test_list_file)
        for video in tqdm(expr_test_video_list):
            subfolder = os.path.join(self.cropped_aligned_img_folder, os.path.splitext(os.path.basename(video))[0])
            cropped_aligned_images = [os.path.join(subfolder, x)for x in os.listdir(subfolder)]
            ids = [int(os.path.splitext(os.path.basename(x))[0]) for x in os.listdir(subfolder)]
            largest_frames = self.get_largest_frame(os.listdir(subfolder))
            results = [None] * (largest_frames + 1)
            results[0] = 'Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n'
            logits_results = [None] * (largest_frames + 1)
            logits_results[0] = 'Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n'

            test_loader = test_data_loader(cropped_aligned_images, ids)
            for imgs, ids in iter(test_loader):
                logits = self.model(imgs.to(self.device))[0]
                pred_expr_softmax = torch.softmax(
                    logits, dim=1).cpu().detach().numpy()
                pred_expr = pred_expr_softmax.argmax(axis=1)

                for pred, id, pred_softmax in zip(pred_expr, ids, pred_expr_softmax):
                    results[id] = str(pred) + '\n'
                    write_str = ''
                    for item in pred_softmax:
                        write_str += (str(item) + ',')
                    logits_results[id] = write_str.strip(',') + '\n'

            for idx, item in enumerate(results):
                if item == None:
                    results[idx] = '-1\n'
            for idx, item in enumerate(logits_results):
                if item == None:
                    logits_results[idx] = '-1,-1,-1,-1,-1,-1,-1\n'

            results[-1].strip()
            logits_results[-1].strip()

            # write output file
            if not os.path.exists(os.path.join(self.output_results, 'EXPR_Set')):
                os.makedirs(os.path.join(self.output_results, 'EXPR_Set'))
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(self.output_results, 'EXPR_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(results)
            f_write.close()

            # write output file
            if not os.path.exists(os.path.join(self.output_logits, 'EXPR_Set')):
                os.makedirs(os.path.join(self.output_logits, 'EXPR_Set'))
            output_file = os.path.splitext(os.path.basename(video))[0] + '.txt'
            output_file_path = os.path.join(self.output_logits, 'EXPR_Set', output_file)
            f_write = open(output_file_path, 'w')
            f_write.writelines(logits_results)
            f_write.close()


if __name__ == '__main__':
    inference_module = Inference()
    inference_module.inference_au_batch()
    inference_module.inference_expr_batch()
