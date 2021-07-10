import torch
from torch import nn
# from lib.ROILayer import ROILayer
from core.lib.model import get_blocks, bottleneck_IR_SE
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
# from transformer import build_transformer,Transformer,TransformerEncoderLayer,TransformerEncoder
import transformer.Constants as Constants
from transformer.Models import Transformer, Encoder
from transformer.Optim import ScheduledOptim


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ROINet(nn.Module):
    def __init__(self, num_classes=12, seq_num=24, cuda_num=0):
        super(ROINet, self).__init__()
        # self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512]
        # self.features = make_layers(self.vgg_cfg)    #vgg backbone
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        num_layers = 50
        blocks = get_blocks(num_layers)
        modules = []
        unit_module = bottleneck_IR_SE
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        # self.roi = ROILayer(cuda_num)
        drop_ratio = 0.6
        self.fc1 = Sequential(BatchNorm2d(512),
                              Dropout(drop_ratio),
                              Flatten(),
                              Linear(512 * 7 * 7, 2048),
                              BatchNorm1d(2048),
                              nn.ReLU())
        # self.fc1 = nn.Sequential(nn.Linear(in_features=3000,out_features=2048),
        #                          nn.ReLU())
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512,
                            num_layers=1, batch_first=True)
        # self.lstm = self_attention(...)
        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes),
                                 nn.Sigmoid())
        self.seq_num = seq_num

    def forward(self, x, y):
        # vgg_features = self.features(x)
        x = self.input_layer(x)
        se_features = self.body(x)  # (512,7,7,)
        # roi = self.roi(se_features,y)
        # roi_fc = self.fc1(roi)
        # roi_fc = roi_fc.view(-1,self.seq_num,2048)
        out1 = self.fc1(se_features)
        out2 = out1.view(-1, self.seq_num, 2048)
        lstm_features, _ = self.lstm(out2)
        output = self.fc2(lstm_features.contiguous().view(-1, 512))
        return output

    # loss function
    @staticmethod
    def multi_label_ACE(outputs, y_labels):
        batch_size, class_size = outputs.size()
        loss_buff = 0
        # pos_weight = [2,3,1,1,1,1,1,5,5,5,1,5]
        pos_weight = [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        for i in range(class_size):
            target = y_labels[:, i]
            output = outputs[:, i]
            temp = -(pos_weight[i]*target * torch.log((output + 0.05) /
                     1.05) + (1.0 - target) * torch.log((1.05 - output) / 1.05))
            temp = temp*(target != 9)
            loss_au = torch.sum(temp)
            # loss_au = torch.sum(-(target * torch.log((output + 0.05) / 1.05) + (1.0 - target) * torch.log((1.05 - output) / 1.05)))
            loss_buff += loss_au
        return loss_buff / (class_size * batch_size)

    # inputs:[prediction,label,thresh]  outputs: [{TP,FP,TN,FN}*class_num] for class_num AUs
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def statistics(pred, y, thresh):
        batch_size = pred.size(0)
        class_nb = pred.size(1)

        pred = pred > thresh
        pred = pred.long()
        pred[pred == 0] = -1
        y[y == 0] = -1

        statistics_list = []
        for j in range(class_nb):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(batch_size):
                if pred[i][j] == 1:
                    if y[i][j] == 1:
                        TP += 1
                    elif y[i][j] == -1:
                        FP += 1
                    else:
                        pass
                        # assert False
                elif pred[i][j] == -1:
                    if y[i][j] == 1:
                        FN += 1
                    elif y[i][j] == -1:
                        TN += 1
                    else:
                        pass
                        # assert False
                else:
                    assert False
            statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
        return statistics_list

    # inputs: [{TP,FP,TN,FN}*class_num] for class_num AUs  outputs: mean F1 scores and lists
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def calc_f1_score(statistics_list):
        f1_score_list = []

        for i in range(len(statistics_list)):
            TP = statistics_list[i]['TP']
            FP = statistics_list[i]['FP']
            FN = statistics_list[i]['FN']

            precise = TP / (TP + FP + 1e-20)
            recall = TP / (TP + FN + 1e-20)
            f1_score = 2 * precise * recall / (precise + recall + 1e-20)
            f1_score_list.append(f1_score)
        mean_f1_score = sum(f1_score_list) / len(f1_score_list)

        return mean_f1_score, f1_score_list

    # update statistics list
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def update_statistics_list(old_list, new_list):
        if not old_list:
            return new_list

        assert len(old_list) == len(new_list)

        for i in range(len(old_list)):
            old_list[i]['TP'] += new_list[i]['TP']
            old_list[i]['FP'] += new_list[i]['FP']
            old_list[i]['TN'] += new_list[i]['TN']
            old_list[i]['FN'] += new_list[i]['FN']

        return old_list


class AttentionNet(nn.Module):
    def __init__(self, num_classes=12, seq_num=24, cuda_num=0, encoder_layers=1):
        super(AttentionNet, self).__init__()
        # self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512]
        # self.features = make_layers(self.vgg_cfg)    #vgg backbone
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        num_layers = 50
        blocks = get_blocks(num_layers)
        modules = []
        unit_module = bottleneck_IR_SE
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        # self.roi = ROILayer(cuda_num)
        self.fc1 = Sequential(BatchNorm2d(512),
                              Dropout(0.6),
                              Flatten(),
                              Linear(512 * 7 * 7, 512),
                              BatchNorm1d(512),
                            #   nn.ReLU()
                              )
        for p in self.parameters():
            p.requires_grad = False  # 预训练模型加载进来后全部设置为不更新参数，然后再后面加层
        # self.fc1 = nn.Sequential(nn.Linear(in_features=3000,out_features=2048),
        #                          nn.ReLU())
        # self.lstm = nn.LSTM(input_size=2048,hidden_size=512,num_layers=1,batch_first=True)
        # self.lstm = self_attention(...)
        # encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
        #                                         dropout=0.1, activation='relu', normalize_before='store_true')
        # normalize_before='store_true'
        # encoder_norm = nn.LayerNorm(512) if normalize_before else None
        # self.encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=encoder_norm)
        self.encoder = Encoder(
            n_src_vocab=None, n_position=seq_num,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=encoder_layers, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=False)
        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes),
                                 nn.Sigmoid())
        self.seq_num = seq_num

    def forward(self, x):
        # vgg_features = self.features(x)
        x = self.input_layer(x)
        se_features = self.body(x)  # (512,7,7,)
        # roi = self.roi(se_features,y)
        # roi_fc = self.fc1(roi)
        # roi_fc = roi_fc.view(-1,self.seq_num,2048)
        out1 = self.fc1(se_features)
        out2 = out1.view(-1, self.seq_num, 512)
        # lstm_features,_ = self.lstm(out2)
        # src_mask = get_pad_mask(out2, 100)
        src_mask = None
        encoder_features = self.encoder(out2, src_mask)
        output = self.fc2(encoder_features.contiguous().view(-1, 512))
        # output = self.fc2(encoder_features.contiguous().view(-1,512))
        return output

    # loss function
    @staticmethod
    def multi_label_ACE(outputs, y_labels):
        batch_size, class_size = outputs.size()
        loss_buff = 0
        # pos_weight = [2,3,1,1,1,1,1,5,5,5,1,5]
        # pos_weight = [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        pos_weight =   [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]
        for i in range(class_size):
            target = y_labels[:, i]
            output = outputs[:, i]
            temp = -(pos_weight[i]*target * torch.log((output + 0.05) /
                     1.05) + (1.0 - target) * torch.log((1.05 - output) / 1.05))
            temp = temp*(target != 9)
            loss_au = torch.sum(temp)
            # loss_au = torch.sum(-(target * torch.log((output + 0.05) / 1.05) + (1.0 - target) * torch.log((1.05 - output) / 1.05)))
            loss_buff += loss_au
        return loss_buff / (class_size * batch_size)
    # inputs:[prediction,label,thresh]  outputs: [{TP,FP,TN,FN}*class_num] for class_num AUs
    # ref: https://github.com/AlexHex7/DRML_pytorch

    @staticmethod
    def statistics(pred, y, thresh):
        batch_size = pred.size(0)
        class_nb = pred.size(1)

        pred = pred > thresh
        pred = pred.long()
        pred[pred == 0] = -1
        y[y == 0] = -1

        statistics_list = []
        for j in range(class_nb):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(batch_size):
                if pred[i][j] == 1:
                    if y[i][j] == 1:
                        TP += 1
                    elif y[i][j] == -1:
                        FP += 1
                    else:
                        pass
                        # assert False
                elif pred[i][j] == -1:
                    if y[i][j] == 1:
                        FN += 1
                    elif y[i][j] == -1:
                        TN += 1
                    else:
                        pass
                        # assert False
                else:
                    assert False
            statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
        return statistics_list

    # inputs: [{TP,FP,TN,FN}*class_num] for class_num AUs  outputs: mean F1 scores and lists
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def calc_f1_score(statistics_list):
        f1_score_list = []

        for i in range(len(statistics_list)):
            TP = statistics_list[i]['TP']
            FP = statistics_list[i]['FP']
            FN = statistics_list[i]['FN']

            precise = TP / (TP + FP + 1e-20)
            recall = TP / (TP + FN + 1e-20)
            f1_score = 2 * precise * recall / (precise + recall + 1e-20)
            f1_score_list.append(f1_score)
        mean_f1_score = sum(f1_score_list) / len(f1_score_list)

        return mean_f1_score, f1_score_list

    # update statistics list
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def update_statistics_list(old_list, new_list):
        if not old_list:
            return new_list

        assert len(old_list) == len(new_list)

        for i in range(len(old_list)):
            old_list[i]['TP'] += new_list[i]['TP']
            old_list[i]['FP'] += new_list[i]['FP']
            old_list[i]['TN'] += new_list[i]['TN']
            old_list[i]['FN'] += new_list[i]['FN']

        return old_list


class AttentionNetAdd(nn.Module):
    def __init__(self, num_classes=12, seq_num=24, cuda_num=0, encoder_layers=1):
        super(AttentionNetAdd, self).__init__()
        # self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512]
        # self.features = make_layers(self.vgg_cfg)    #vgg backbone
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        num_layers = 50
        blocks = get_blocks(num_layers)
        modules = []
        unit_module = bottleneck_IR_SE
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        # for block in blocks[0:3]:
        #     for bottleneck in block:
        #         modules.append(
        #             unit_module(bottleneck.in_channel,
        #                         bottleneck.depth,
        #                         bottleneck.stride))
        # for p in self.parameters():
        #     p.requires_grad = False  # 预训练模型加载进来后全部设置为不更新参数，然后再后面加层
        # for this_block in blocks[-1]:
        # for this_bottleneck in blocks[-1]:
        #     modules.append(
        #         unit_module(this_bottleneck.in_channel,
        #                     this_bottleneck.depth,
        #                     this_bottleneck.stride))
        self.body = Sequential(*modules)
        # self.roi = ROILayer(cuda_num)
        for p in self.parameters():
            p.requires_grad = False 
        self.fc1 = Sequential(BatchNorm2d(512),
                              Dropout(0.6),
                              nn.ReLU(),
                              Flatten(),
                              Linear(512 * 7 * 7, 512),
                              BatchNorm1d(512),
                              nn.ReLU()
                              )
        # for p in self.parameters():
        #     p.requires_grad = False  # 预训练模型加载进来后全部设置为不更新参数，然后再后面加层
        # self.fc1 = nn.Sequential(nn.Linear(in_features=3000,out_features=2048),
        #                          nn.ReLU())
        # self.lstm = nn.LSTM(input_size=2048,hidden_size=512,num_layers=1,batch_first=True)
        # self.lstm = self_attention(...)
        # encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
        #                                         dropout=0.1, activation='relu', normalize_before='store_true')
        # normalize_before='store_true'
        # encoder_norm = nn.LayerNorm(512) if normalize_before else None
        # self.encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=encoder_norm)
        self.encoder = Encoder(
            n_src_vocab=None, n_position=seq_num,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=encoder_layers, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=False)
        self.fc2 = nn.Sequential(nn.Linear(in_features=512*2, out_features=num_classes),
                                 nn.Sigmoid())
        self.seq_num = seq_num

    def forward(self, x):
        # vgg_features = self.features(x)
        x = self.input_layer(x)
        se_features = self.body(x)  # (512,7,7,)
        # roi = self.roi(se_features,y)
        # roi_fc = self.fc1(roi)
        # roi_fc = roi_fc.view(-1,self.seq_num,2048)
        out1 = self.fc1(se_features)
        out2 = out1.view(-1, self.seq_num, 512)
        # lstm_features,_ = self.lstm(out2)
        # src_mask = get_pad_mask(out2, 100)
        src_mask = None
        encoder_features = self.encoder(out2, src_mask)
        total_features = torch.cat((out2,encoder_features ), 2)
        output = self.fc2(total_features.contiguous().view(-1, 512*2))
        # output = self.fc2(encoder_features.contiguous().view(-1,512))
        return output

    # loss function
    @staticmethod
    def multi_label_ACE(outputs, y_labels):
        batch_size, class_size = outputs.size()
        loss_buff = 0
        # pos_weight = [2,3,1,1,1,1,1,5,5,5,1,5]
        # pos_weight = [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        pos_weight =   [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]
        for i in range(class_size):
            target = y_labels[:, i]
            output = outputs[:, i]
            temp = -(pos_weight[i]*target * torch.log((output + 0.05) /
                     1.05) + (1.0 - target) * torch.log((1.05 - output) / 1.05))
            temp = temp*(target != 9)
            loss_au = torch.sum(temp)
            # loss_au = torch.sum(-(target * torch.log((output + 0.05) / 1.05) + (1.0 - target) * torch.log((1.05 - output) / 1.05)))
            loss_buff += loss_au
        return loss_buff / (class_size * batch_size)
    # inputs:[prediction,label,thresh]  outputs: [{TP,FP,TN,FN}*class_num] for class_num AUs
    # ref: https://github.com/AlexHex7/DRML_pytorch

    @staticmethod
    def statistics(pred, y, thresh):
        batch_size = pred.size(0)
        class_nb = pred.size(1)

        pred = pred > thresh
        pred = pred.long()
        pred[pred == 0] = -1
        y[y == 0] = -1

        statistics_list = []
        for j in range(class_nb):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(batch_size):
                if pred[i][j] == 1:
                    if y[i][j] == 1:
                        TP += 1
                    elif y[i][j] == -1:
                        FP += 1
                    else:
                        pass
                        # assert False
                elif pred[i][j] == -1:
                    if y[i][j] == 1:
                        FN += 1
                    elif y[i][j] == -1:
                        TN += 1
                    else:
                        pass
                        # assert False
                else:
                    assert False
            statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
        return statistics_list

    # inputs: [{TP,FP,TN,FN}*class_num] for class_num AUs  outputs: mean F1 scores and lists
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def calc_f1_score(statistics_list):
        f1_score_list = []

        for i in range(len(statistics_list)):
            TP = statistics_list[i]['TP']
            FP = statistics_list[i]['FP']
            FN = statistics_list[i]['FN']

            precise = TP / (TP + FP + 1e-20)
            recall = TP / (TP + FN + 1e-20)
            f1_score = 2 * precise * recall / (precise + recall + 1e-20)
            f1_score_list.append(f1_score)
        mean_f1_score = sum(f1_score_list) / len(f1_score_list)

        return mean_f1_score, f1_score_list

    # update statistics list
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def update_statistics_list(old_list, new_list):
        if not old_list:
            return new_list

        assert len(old_list) == len(new_list)

        for i in range(len(old_list)):
            old_list[i]['TP'] += new_list[i]['TP']
            old_list[i]['FP'] += new_list[i]['FP']
            old_list[i]['TN'] += new_list[i]['TN']
            old_list[i]['FN'] += new_list[i]['FN']

        return old_list


class TwoBranchesAttentionNet(nn.Module):
    def __init__(self, num_classes=12, seq_num=24, cuda_num=0, encoder_layers=1):
        super(AttentionNet, self).__init__()
        # self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512]
        # self.features = make_layers(self.vgg_cfg)    #vgg backbone
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        num_layers = 50
        blocks = get_blocks(num_layers)
        modules = []
        unit_module = bottleneck_IR_SE
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        # self.roi = ROILayer(cuda_num)
        self.fc1 = Sequential(BatchNorm2d(512),
                              #    Dropout(drop_ratio),
                              Flatten(),
                              Linear(512 * 7 * 7, 512),
                              BatchNorm1d(512),
                            #   nn.ReLU()
                              )
        for p in self.parameters():
            p.requires_grad = False  # 预训练模型加载进来后全部设置为不更新参数，然后再后面加层
        # self.fc1 = nn.Sequential(nn.Linear(in_features=3000,out_features=2048),
        #                          nn.ReLU())
        # self.lstm = nn.LSTM(input_size=2048,hidden_size=512,num_layers=1,batch_first=True)
        # self.lstm = self_attention(...)
        # encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
        #                                         dropout=0.1, activation='relu', normalize_before='store_true')
        # normalize_before='store_true'
        # encoder_norm = nn.LayerNorm(512) if normalize_before else None
        # self.encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=encoder_norm)
        self.encoder = Encoder(
            n_src_vocab=None, n_position=seq_num,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=encoder_layers, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=False)
        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes),
                                 nn.Sigmoid())
        self.seq_num = seq_num

    def forward(self, x, y):
        # vgg_features = self.features(x)
        x = self.input_layer(x)
        se_features = self.body(x)  # (512,7,7,)
        # roi = self.roi(se_features,y)
        # roi_fc = self.fc1(roi)
        # roi_fc = roi_fc.view(-1,self.seq_num,2048)
        out1 = self.fc1(se_features)
        out2 = out1.view(-1, self.seq_num, 512)
        # lstm_features,_ = self.lstm(out2)
        # src_mask = get_pad_mask(out2, 100)
        src_mask = None
        encoder_features = self.encoder(out2, src_mask)
        output = self.fc2(encoder_features.contiguous().view(-1, 512))
        # output = self.fc2(encoder_features.contiguous().view(-1,512))
        return output

    # loss function
    @staticmethod
    def multi_label_ACE(outputs, y_labels):
        batch_size, class_size = outputs.size()
        loss_buff = 0
        # pos_weight = [2,3,1,1,1,1,1,5,5,5,1,5]
        pos_weight = [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        for i in range(class_size):
            target = y_labels[:, i]
            output = outputs[:, i]
            temp = -(pos_weight[i]*target * torch.log((output + 0.05) /
                     1.05) + (1.0 - target) * torch.log((1.05 - output) / 1.05))
            temp = temp*(target != 9)
            loss_au = torch.sum(temp)
            # loss_au = torch.sum(-(target * torch.log((output + 0.05) / 1.05) + (1.0 - target) * torch.log((1.05 - output) / 1.05)))
            loss_buff += loss_au
        return loss_buff / (class_size * batch_size)
    # inputs:[prediction,label,thresh]  outputs: [{TP,FP,TN,FN}*class_num] for class_num AUs
    # ref: https://github.com/AlexHex7/DRML_pytorch

    @staticmethod
    def statistics(pred, y, thresh):
        batch_size = pred.size(0)
        class_nb = pred.size(1)

        pred = pred > thresh
        pred = pred.long()
        pred[pred == 0] = -1
        y[y == 0] = -1

        statistics_list = []
        for j in range(class_nb):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(batch_size):
                if pred[i][j] == 1:
                    if y[i][j] == 1:
                        TP += 1
                    elif y[i][j] == -1:
                        FP += 1
                    else:
                        pass
                        # assert False
                elif pred[i][j] == -1:
                    if y[i][j] == 1:
                        FN += 1
                    elif y[i][j] == -1:
                        TN += 1
                    else:
                        pass
                        # assert False
                else:
                    assert False
            statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
        return statistics_list

    # inputs: [{TP,FP,TN,FN}*class_num] for class_num AUs  outputs: mean F1 scores and lists
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def calc_f1_score(statistics_list):
        f1_score_list = []

        for i in range(len(statistics_list)):
            TP = statistics_list[i]['TP']
            FP = statistics_list[i]['FP']
            FN = statistics_list[i]['FN']

            precise = TP / (TP + FP + 1e-20)
            recall = TP / (TP + FN + 1e-20)
            f1_score = 2 * precise * recall / (precise + recall + 1e-20)
            f1_score_list.append(f1_score)
        mean_f1_score = sum(f1_score_list) / len(f1_score_list)

        return mean_f1_score, f1_score_list

    # update statistics list
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def update_statistics_list(old_list, new_list):
        if not old_list:
            return new_list

        assert len(old_list) == len(new_list)

        for i in range(len(old_list)):
            old_list[i]['TP'] += new_list[i]['TP']
            old_list[i]['FP'] += new_list[i]['FP']
            old_list[i]['TN'] += new_list[i]['TN']
            old_list[i]['FN'] += new_list[i]['FN']

        return old_list

# helper for making vgg net


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
