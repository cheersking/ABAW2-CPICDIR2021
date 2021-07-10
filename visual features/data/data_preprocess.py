import matplotlib.pyplot as plt
import os
import shutil
import sys
import random
from tqdm import tqdm
import csv
sys.path.append("..")


class DataPreprocess():
    def __init__(self, annots, cropped_image_data, dataset_folder):
        self.PLOT = False
        # annot path
        self.annots = annots
        self.cropped_image_data = cropped_image_data  # all crop and aligned images
        self.exprs = os.path.join(annots, "EXPR_Set", "Train_Set")
        self.aus = os.path.join(annots, "AU_Set", "Train_Set")
        self.exprs_valid = os.path.join(annots, "EXPR_Set", "Validation_Set")
        self.aus_valid = os.path.join(annots, "AU_Set", "Validation_Set")
        self.output_unannotated_expr = os.path.join(dataset_folder, "unannotated_expr_cropped_aligned")
        self.output_unannotated_au = os.path.join(dataset_folder, "unannotated_au_cropped_aligned")
        self.output = os.path.join(
            annots, "output", "annots")  # to store output csv

        # expr and au list
        # 加None类别，标注类别为-1
        self.expr_rep = ["Neutral", "Anger", "Disgust",
                         "Fear", "Happiness", "Sadness", "Surprise", "None"]
        self.au_rep = ["AU1", "AU2", "AU4", "AU6", "AU7",
                       "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]
        self.au_header = ["filename", "AU01", "AU02", "AU03", "AU04", "AU05", "AU06",
                          "AU07", "AU08", "AU09", "AU10", "AU11", "AU12", "AU13", "AU14",
                          "AU15", "AU16", "AU17", "AU18", "AU19", "AU20", "AU21", "AU22",
                          "AU23", "AU24", "AU25", "AU26", "AU27", "AU28", "AU29", "AU30",
                          "AU31", "AU32", "AU33", "AU34", "AU35", "AU36", "AU37", "AU38",
                          "AU39", "AU40", "AU41", "AU42", "AU43", "AU44", "AU45", "AU46",
                          "AU47", "AU48", "AU49", "AU50", "AU51", "AU52", "AU53", "AU54",
                          "AU55", "AU56", "AU57", "AU58", "AU59", "AU60"]

        if not os.path.exists(self.output):
            os.makedirs(self.output)
        if not os.path.exists(self.output_unannotated_expr):
            os.makedirs(self.output_unannotated_expr)
        if not os.path.exists(self.output_unannotated_au):
            os.makedirs(self.output_unannotated_au)

    def create_expr_annot(self):
        output_csv = os.path.join(self.output, 'expression_annots_valid.csv')
        write_list = list()
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for video in os.listdir(self.cropped_image_data):
                video_annots = os.path.join(self.annots, "EXPR_Set", "Validation_Set",
                                            video + ".txt")
                if os.path.exists(video_annots):
                    with open(video_annots, 'r') as f_annot:
                        for idx, line in enumerate(f_annot.readlines()):
                            if idx > 0:
                                file_name = video + '/' + \
                                    str(idx).zfill(5) + '.jpg'
                                label = line.strip()
                                write_list.append([file_name, label])
            # print(write_list)
            writer.writerows(write_list)

    def create_au_annot(self):
        output_csv = os.path.join(self.output, 'au_annots_train.csv')
        mapping = {0: 0, 1: 1, 2: 3, 3: 5, 4: 6, 5: 9,
                   6: 11, 7: 14, 8: 22, 9: 23, 10: 24, 11: 25}
        write_list = list()
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.au_header)
            for video in os.listdir(self.cropped_image_data):
                video_annots = os.path.join(self.annots, "AU_Set", "Train_Set",
                                            video + ".txt")
                if os.path.exists(video_annots):
                    with open(video_annots, 'r') as f_annot:
                        for idx, line in enumerate(f_annot.readlines()):
                            if idx > 0:
                                file_name = video + '/' + \
                                    str(idx).zfill(5) + '.jpg'
                                line = line.strip()
                                label_list = ['999' for i in range(60)]
                                annot_line = line.split(',')
                                for au_idx, au_annot in enumerate(annot_line):
                                    label_list[mapping[au_idx]] = au_annot
                                # label = ','.join(label_list)
                                write_list.append([file_name] + label_list)
            # print(write_list)
            writer.writerows(write_list)

    def video_exclude_trainval(self, task):
        if task == 'expr':
            valid_expr_dir = os.path.join(
                self.annots, 'EXPR_Set/Validation_Set')
            train_au_dir = os.path.join(self.annots, 'AU_Set/Train_Set')
            valid_exclude_set = []
            train_aus = os.listdir(train_au_dir)
            for item in os.listdir(valid_expr_dir):
                if item not in train_aus:
                    valid_exclude_set.append(item)
            print(valid_exclude_set)

            # write csv
            output_csv = os.path.join(
                self.output, 'expression_annots_valid_exclude_autrain.csv')
            write_list = list()
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for video in valid_exclude_set:
                    video_annots = os.path.join(valid_expr_dir, video)
                    with open(video_annots, 'r') as f_annot:
                        for idx, line in enumerate(f_annot.readlines()):
                            if idx > 0:
                                file_name = video.split(
                                    '.')[0] + '/' + str(idx).zfill(5) + '.jpg'
                                label = line.strip()
                                write_list.append([file_name, label])
                writer.writerows(write_list)

    def video_not_annotated(self):
        all_videos = os.listdir(self.cropped_image_data)
        annotated_expr_videos = [x.split('.')[0] for x in os.listdir(
            self.exprs)] + [x.split('.')[0] for x in os.listdir(self.exprs_valid)]
        annotated_au_videos = [x.split('.')[0] for x in os.listdir(
            self.aus)] + [x.split('.')[0] for x in os.listdir(self.aus_valid)]
        expr_not_annotated = list(set(all_videos) ^ set(annotated_expr_videos))
        au_not_annotated = list(set(all_videos) ^ set(annotated_au_videos))
        for dir in expr_not_annotated:
            sub_dir = os.path.join(self.output_unannotated_expr, dir)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for img in os.listdir(os.path.join(self.cropped_image_data, dir)):
                shutil.copy(os.path.join(self.cropped_image_data, dir, img), os.path.join(sub_dir, img))
        for dir in au_not_annotated:
            sub_dir = os.path.join(self.output_unannotated_au, dir)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for img in os.listdir(os.path.join(self.cropped_image_data, dir)):
                shutil.copy(os.path.join(self.cropped_image_data, dir, img), os.path.join(sub_dir, img))


if __name__ == '__main__':
    dataset_folder = '/data/jinyue/dataset/opensource/AffWild2/'
    cropped_image_data = os.path.join(dataset_folder, "cropped_aligned")
    annots = os.path.join(dataset_folder, "Annotations/annotations")

    data_preprocess = DataPreprocess(annots, cropped_image_data, dataset_folder)
    # data_preprocess.create_au_annot()
    # data_preprocess.video_exclude_trainval('expr')
    data_preprocess.video_not_annotated()
