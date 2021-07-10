import os

import csv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans
from PIL import Image

from config import get_config
from data import data_pipe
from models.iresnetse import IrResNetSe
from models.iresnet import iresnet100
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


class Inference():
    def __init__(self, model_path):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = trans.Compose([
            trans.Resize((112, 112)),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.class_names = ["Neutral", "Anger", "Disgust",
                            "Fear", "Happiness", "Sadness", "Surprise"]
        self.au_names = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]

        self.output_dir = '/data/jinyue/pseudo_labels/affwild2_not_annotated'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_expr_dir = os.path.join(self.output_dir, 'expr')
        self.output_au_dir = os.path.join(self.output_dir, 'au')
        if not os.path.exists(self.output_expr_dir):
            os.makedirs(self.output_expr_dir)
        if not os.path.exists(self.output_au_dir):
            os.makedirs(self.output_au_dir)

        self.model = iresnet100_2branches().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def inference_single_image(self, image_path):
        img = Image.open(image_path)
        img = self.transform(img)
        output = self.model(img.unsqueeze(0).to(self.device))
        pred_expr_softmax = torch.softmax(
            output[0].squeeze(), dim=0).cpu().detach().numpy()
        pred_expr = pred_expr_softmax.argmax()
        print(pred_expr_softmax)
        pred_expr_score = pred_expr_softmax[pred_expr]
        pred_au = torch.sigmoid(output[1].squeeze()).cpu().detach().numpy()
        pred_au_01 = [1 if x > 0.5 else 0 for x in pred_au]
        print(self.class_names[pred_expr])
        for idx, item in enumerate(pred_au_01):
            if item == 1:
                print('AU', self.au_names[idx])
        return pred_expr, pred_expr_score, pred_au_01, pred_au

    def inference_dataloader(self, test_dataset):
        for video_dir in os.listdir(test_dataset):
            result_txt_file = video_dir + '.txt'
            f_expr_result = open(os.path.join(self.output_expr_dir, result_txt_file), 'a')
            f_expr_result.write('Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n')
            f_au_result = open(os.path.join(self.output_au_dir, result_txt_file), 'a')
            f_au_result.write('AU1,AU2,AU4,AU6,AU7,AU10,AU12,AU15,AU23,AU24,AU25,AU26\n')
            score_txt_file = video_dir + '_score.txt'
            f_expr_score = open(os.path.join(self.output_expr_dir, score_txt_file), 'a')
            f_au_score = open(os.path.join(self.output_au_dir, score_txt_file), 'a')
            video_folder = os.path.join(test_dataset, video_dir)
            test_loader = test_dataloader(video_folder)
            for imgs, image_paths in tqdm(iter(test_loader)):
                imgs = imgs.to(self.device)
                output = self.model(imgs)

                # expr
                pred_expr_softmax = torch.softmax(
                    output[0], dim=1).cpu().detach().numpy()
                pred_expr = pred_expr_softmax.argmax(axis=1)
                for item in pred_expr:
                    f_expr_result.write(str(item) + '\n')
                for idx, item in enumerate(pred_expr_softmax):
                    f_expr_score.write(str(item[pred_expr[idx]]) + '\n')

                # au
                pred_au = torch.sigmoid(output[1]).cpu().detach().numpy()
                pred_au_01 = [[1 if x > 0.5 else 0 for x in y] for y in pred_au]
                for item in pred_au_01:
                    f_au_result.write(str(item).replace('[', '').replace(']', '').replace(' ', '') + '\n')
                for item in pred_au:
                    f_au_score.write(str(item) + '\n')


            f_expr_result.close()
            f_au_result.close()
            f_expr_score.close()
            f_au_score.close()

    def inference_dataloader_to_csv(self, test_dataset):
        # file_path,expr_result,expr_score
        output_expr_csv = os.path.join(self.output_dir, 'unannotated_expr_cropped_aligned_pred_results_expr.csv')
        expr_write_list = []
        expr_csv_file = open(output_expr_csv, 'w', newline='')
        expr_writer = csv.writer(expr_csv_file)

        output_au_csv = os.path.join(self.output_dir, 'unannotated_expr_cropped_aligned_pred_results_au.csv')
        au_write_list = []
        au_csv_file = open(output_au_csv, 'w', newline='')
        au_writer = csv.writer(au_csv_file)

        for video_dir in os.listdir(test_dataset):
            video_folder = os.path.join(test_dataset, video_dir)
            test_loader = test_dataloader(video_folder)
            for imgs, image_paths in tqdm(iter(test_loader)):
                imgs = imgs.to(self.device)
                output = self.model(imgs)
                # expr
                pred_expr_softmax = torch.softmax(
                    output[0], dim=1).cpu().detach().numpy()
                pred_expr = pred_expr_softmax.argmax(axis=1)
                for idx, item in enumerate(zip(image_paths, pred_expr, pred_expr_softmax)):
                    expr_write_list.append([item[0] + ',' + str(item[1]) + ',' + str(item[2][pred_expr[idx]])])

                # au
                pred_au = torch.sigmoid(output[1]).cpu().detach().numpy()
                pred_au_01 = [[1 if x > 0.5 else 0 for x in y] for y in pred_au]
                for item in zip(image_paths, pred_au_01, pred_au):
                    au_write_list.append([item[0] + ',' + str(item).replace('[', '').replace(']', '').replace(' ', '') + ',' + str(item)])

        expr_writer.writerows(expr_write_list)
        expr_csv_file.close()
        au_writer.writerows(au_write_list)
        au_csv_file.close()


if __name__ == '__main__':
    model_path = '/data/jinyue/logs/v4.1.1_2branhces_swapepoch/save/model_expr_2021-06-23-17-21_loss-0.006_acc-0.839_f1-0.729_step-339456_epoch-11.pth'
    inference_module = Inference(model_path)

    # image_path = '00028_5.jpg'
    # pred_expr, pred_expr_score, pred_au_01, pred_au = inference_module.inference_single_image(
    #     image_path)
    # print(pred_expr, pred_expr_score, pred_au_01, pred_au)

    test_dataset = '/data/jinyue/dataset/opensource/AffWild2/unannotated_expr_cropped_aligned'
    inference_module.inference_dataloader_to_csv(test_dataset)
