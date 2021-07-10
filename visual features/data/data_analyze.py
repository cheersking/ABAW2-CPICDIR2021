import os
import shutil
import sys
import random
from tqdm import tqdm
import csv
import pandas as pd
sys.path.append("..")
import matplotlib.pyplot as plt

class DataAnalyze():
    def __init__(self, annots):
        self.PLOT = True
        # annot path
        self.annots = annots
        self.exprs = os.path.join(annots, "EXPR_Set", "Train_Set")
        self.aus = os.path.join(annots, "AU_Set", "Train_Set")
        # self.exprs = os.path.join(annots, "EXPR_Set", "Validation_Set")
        # self.aus = os.path.join(annots, "AU_Set", "Validation_Set")
        # self.exprs = os.path.join(annots, "test")
        self.output = os.path.join(annots, "output")  # to store output csv

        # expr and au list
        # None，class is -1
        self.expr_rep = ["Neutral", "Anger", "Disgust",
                    "Fear", "Happiness", "Sadness", "Surprise", "None"]
        self.au_rep = ["AU1", "AU2", "AU4", "AU6", "AU7",
                "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]

        if not os.path.exists(self.output):
            os.makedirs(self.output)


    def stat_expression(self):
        zeros_expr = [0] * len(self.expr_rep)
        expr_rep_dict = dict(zip(self.expr_rep, zeros_expr))
        for root, _, files in os.walk(self.exprs):
            for file in files:
                annot = os.path.join(root, file)
                f = open(annot, 'r')
                for idx, line in enumerate(f.readlines()):
                    if idx > 0:
                        line = line.strip()
                        expr_rep_dict[self.expr_rep[int(line)]] += 1
                f.close()
        print(expr_rep_dict)

        # plot
        if self.PLOT:
            # plot expr stat
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            # plt.rcParams['axes.unicode_minus'] = False

            x_axis = []
            for x in expr_rep_dict:
                x_axis.append(x)
            value = []
            for x in expr_rep_dict:
                value.append(expr_rep_dict[x])

            fig = plt.figure(figsize=(20,10))
            plt.bar(x_axis, value)
            plt.title('expr_stat')
            plt.savefig('expr_stat.png')
        return expr_rep_dict


    def stat_au(self):
        zeros_au = [0] * len(self.au_rep)
        au_rep_dict = dict(zip(self.au_rep, zeros_au))
        au_combinition = dict()
        for root, _, files in os.walk(self.aus):
            for file in files:
                annot = os.path.join(root, file)
                f = open(annot, 'r')
                for line in f.readlines():
                    line = line.strip()
                    for idx, au in enumerate(line.split(',')):
                        if au == '1':
                            au_rep_dict[self.au_rep[idx]] += 1
                    # stat au_combinition (0-1 format)
                    if line in au_combinition:
                        au_combinition[line] += 1
                    else:
                        au_combinition[line] = 1
                f.close()
        print(au_rep_dict)
        print('\n\n')

        # translate au_combinition
        translated_au_combinition = dict()
        for key in au_combinition:
            translated_key = ''
            for idx, au in enumerate(key.split(',')):
                if au == '1':
                    translated_key += (self.au_rep[idx] + '_')
            translated_key = translated_key.strip('_')
            translated_au_combinition[translated_key] = au_combinition[key]
        translated_au_combinition = sorted(translated_au_combinition.items(),
                                            key=lambda x: x[1], reverse=True)
        # print(translated_au_combinition)

        # normalize translated_au_combinition
        # normed by mean of contained aus
        # 除以key中每个AU在au_rep_dict出现次数的平均值
        normed_translated_au_combinition = dict()
        for key, value in translated_au_combinition:
            if key:
                temp = []
                for au in key.split('_'):
                    temp.append(au_rep_dict[au])
                normed_translated_au_combinition[key] = value / (sum(temp) / len(temp))
        normed_translated_au_combinition = sorted(normed_translated_au_combinition.items(),
                                            key=lambda x: x[1], reverse=True)
        # print(normed_translated_au_combinition)

        # plot
        if self.PLOT:
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            # plt.rcParams['axes.unicode_minus'] = False

            below_thr = 0
            below_thr_cnt = 0
            x_axis = []
            for x in normed_translated_au_combinition:
                if x[1] > 0.005:
                    x_axis.append(x[0].replace('AU', ''))
            value = []
            for x in normed_translated_au_combinition:
                if x[1] > 0.005:
                    value.append(x[1])
                else:
                    below_thr += x[1]
                    below_thr_cnt += 1
            x_axis.append('below thr cnt:' + str(below_thr_cnt))
            value.append(below_thr)

            # print(x_axis)
            # print(value)
            fig = plt.figure(figsize=(100,10))
            plt.bar(x_axis, value)
            plt.title('normed_au_combinition (thr=0.005)')
            plt.savefig('normed_translated_au_combinition.png')
            # plt.show()

            # plot au stat
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            # plt.rcParams['axes.unicode_minus'] = False

            x_axis = []
            for x in au_rep_dict:
                x_axis.append(x)
            value = []
            for x in au_rep_dict:
                value.append(au_rep_dict[x])

            fig = plt.figure(figsize=(20,10))
            plt.bar(x_axis, value)
            plt.title('au_stat')
            plt.savefig('au_stat.png')


        return au_rep_dict, translated_au_combinition, normed_translated_au_combinition


    def stat_expr_aus_relationship(self):
        expr_au = dict()
        # get aus annot files
        aus_list = os.listdir(self.aus)
        # get exprs annot files
        for root, _, files in os.walk(self.exprs):
            for file in files:
                if file in aus_list:
                    f_expr = open(os.path.join(root, file), 'r')
                    f_aus = open(os.path.join(self.aus, file))
                    expr_lines = f_expr.readlines()
                    for idx, au_line in enumerate(f_aus.readlines()):
                        if idx != 0:
                            au_annot = au_line.strip()  # 0-1 format
                            expr_annot = int(expr_lines[idx].strip())  # int label
                            expr = self.expr_rep[expr_annot]  # word label
                            if expr_annot != -1:
                                if expr not in expr_au:
                                    expr_au[expr] = {}

                                # translate au annot to word format
                                translated_key = ''
                                for idx, au in enumerate(au_annot.split(',')):
                                    if au == '1':
                                        translated_key += (self.au_rep[idx] + '_')
                                translated_key = translated_key.strip('_')

                                # stat au-expr
                                if translated_key not in expr_au[expr]:
                                    expr_au[expr][translated_key] = 1
                                else:
                                    expr_au[expr][translated_key] += 1
                    f_expr.close()
                    f_aus.close()

        # write a csv file
        csv_file = os.path.join(self.annots, 'expr-au.csv')
        # TODO write relationships to a csv file


        print(expr_au)
        return expr_au


if __name__ == '__main__':
    dataset_folder = '/home/jinyue/emotion/dataset/opensource/AffWild2'
    video_data = os.path.join(dataset_folder, "batch2/batch2")
    cropped_image_data = os.path.join(dataset_folder, "cropped_aligned")
    annots = os.path.join(dataset_folder, "Annotations/annotations")

    data_analyze = DataAnalyze(annots)
    data_analyze.stat_expression()
    # data_analyze.stat_au()
    # data_analyze.stat_expr_aus_relationship()

