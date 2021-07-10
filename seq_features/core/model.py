"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch.nn as nn
import torch
from torchvision import models
import torch
from torch import nn
# from lib.ROILayer import ROILayer
from core.lib.model import get_blocks, bottleneck_IR_SE
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
# from transformer import build_transformer,Transformer,TransformerEncoderLayer,TransformerEncoder
import transformer.Constants as Constants
from transformer.Models import Transformer, Encoder
from transformer.Optim import ScheduledOptim
from core.tdnn_models import FTDNNLayer, SOrthConv, FTDNN, TDNN
import torch.nn.functional as F


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


# class VideoModel(nn.Module):
#     def __init__(self, num_channels=3):
#         super(VideoModel, self).__init__()
#         self.r2plus1d = models.video.r2plus1d_18(pretrained=True)
#         self.r2plus1d.fc = nn.Sequential(nn.Dropout(0.0),
#                                          nn.Linear(in_features=self.r2plus1d.fc.in_features, out_features=17))
#         if num_channels == 4:
#             new_first_layer = nn.Conv3d(in_channels=4,
#                                         out_channels=self.r2plus1d.stem[0].out_channels,
#                                         kernel_size=self.r2plus1d.stem[0].kernel_size,
#                                         stride=self.r2plus1d.stem[0].stride,
#                                         padding=self.r2plus1d.stem[0].padding,
#                                         bias=False)
#             # copy pre-trained weights for first 3 channels
#             new_first_layer.weight.data[:, 0:3] = self.r2plus1d.stem[0].weight.data
#             self.r2plus1d.stem[0] = new_first_layer
#         self.modes = ["clip"]

#     def forward(self, x):
#         return self.r2plus1d(x)


class AudioModel(nn.Module):
    def __init__(self, pretrained=False):
        super(AudioModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.0),
                                       nn.Linear(in_features=self.resnet.fc.in_features, out_features=12))

        old_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(2, out_channels=self.resnet.conv1.out_channels,
                                      kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if pretrained == True:
            self.resnet.conv1.weight.data.copy_(torch.mean(
                old_layer.weight.data, dim=1, keepdim=True))  # mean channel

        self.norm = BatchNorm1d(512)

    def forward(self, x):
        out = self.resnet(x)
        out = self.norm(out)
        return out


class AudioFTDNNModel(nn.Module):
    def __init__(self, pretrained=False):
        super(AudioFTDNNModel, self).__init__()
        # self.tdnn = TDNN(input_dim=64*2,output_dim=512,context_size=5, dilation=1)
        # self.frame1 = TDNN(input_dim=64*2, output_dim=512, context_size=5, dilation=1,stride=2)
        # self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        # self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        # self.frame4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        # self.frame5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.tdnn_f = FTDNN(in_dim=64)
        # self.tdnn_f = FTDNNLayer(64, 512, 256, context_size=2, dilations=[2,2,2], paddings=[1,1,1])

    def forward(self, x):
        # x = x.view(-1, 200, 64*2)
        # y = self.frame1(x)
        # y = self.frame2(y)
        # y = self.frame3(y)
        # y = self.frame4(x)
        out = self.tdnn_f(x)
        # out = self.tdnn(x)
        return out


class AudioAttentionModel(nn.Module):
    def __init__(self, pretrained=False, seq_num=30, encoder_layers=1):
        super(AudioAttentionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.0),
                                       nn.Linear(in_features=self.resnet.fc.in_features, out_features=12))

        old_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(2, out_channels=self.resnet.conv1.out_channels,
                                      kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if pretrained == True:
            self.resnet.conv1.weight.data.copy_(torch.mean(
                old_layer.weight.data, dim=1, keepdim=True))  # mean channel

        self.modes = ["audio_features"]
        self.seq_num = seq_num
        self.encoder = Encoder(
            n_src_vocab=None, n_position=seq_num,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=encoder_layers, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=False)

    def forward(self, x):
        audio_feature = self.resnet(x)  # (batch_size*seq_num,512)
        audio_feature = audio_feature.view(-1, self.seq_num, 512)
        src_mask = None
        out = self.encoder(audio_feature, src_mask)
        return out


class VideoModel(nn.Module):
    def __init__(self, num_classes=12, seq_num=30, encoder_layers=1):
        super(VideoModel, self).__init__()
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
        for p in self.parameters():
            p.requires_grad = False
        # self.roi = ROILayer(cuda_num)
        self.fc1 = Sequential(BatchNorm2d(512),
                              Dropout(0.6),
                              nn.ReLU(),
                              Flatten(),
                              Linear(512 * 7 * 7, 512),
                              BatchNorm1d(512),
                              nn.ReLU(),
                              #   nn.LayerNorm(512)
                              )

    def forward(self, x):
        # vgg_features = self.features(x)
        x = self.input_layer(x)
        se_features = self.body(x)  # (512,7,7,)
        img_features = self.fc1(se_features)
        return img_features


class VideoModelAttention(nn.Module):
    def __init__(self, num_classes=12, seq_num=30, encoder_layers=1):
        super(VideoModelAttention, self).__init__()
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
        for p in self.parameters():
            p.requires_grad = False  # 预训练模型加载进来后全部设置为不更新参数，然后再后面加层
        self.fc1 = Sequential(BatchNorm2d(512),
                              Dropout(0.6),
                              Flatten(),
                              Linear(512 * 7 * 7, 512),
                              BatchNorm1d(512),
                              nn.ReLU()
                              )
        # for p in self.parameters():
        #     p.requires_grad = False  # 预训练模型加载进来后全部设置为不更新参数，然后再后面加层
        self.encoder = Encoder(
            n_src_vocab=None, n_position=seq_num,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=encoder_layers, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=False)
        # self.fc2 = nn.Sequential(nn.Linear(in_features=512*2, out_features=num_classes),
        #                          nn.Sigmoid())
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
        total_features = torch.cat((out2, encoder_features), 2)
        # output = self.fc2(total_features.contiguous().view(-1, 512*2))
        # output = self.fc2(encoder_features.contiguous().view(-1,512))
        return total_features


class TwoStreamAUModel(nn.Module):
    def __init__(self, audio_pretrained=False, seq_num=30, encoder_layers=1):
        super(TwoStreamAUModel, self).__init__()
        self.audio_model = AudioModel(pretrained=audio_pretrained)
        # self.video_model = VideoModel(num_channels=num_channels)
        self.video_model = VideoModelAttention(
            seq_num=seq_num, encoder_layers=encoder_layers)
        # pretrained_model = torch.load(
        #     '/data/tian/developer/output/model_2021-06-17-17-16_acc:0.8787685929713989_f1:0.5328100042602213_bacth:1_step:12053.pth')
        pretrained_model = torch.load(
            '/data/tianqing/output/best/model_2021-06-17-17-16_acc:0.8787685929713989_f1:0.5328100042602213_bacth:1_step:12053.pth')
        video_net_dict = self.video_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.items() if k in video_net_dict}
        video_net_dict.update(pretrained_dict)
        self.video_model.load_state_dict(video_net_dict)

        self.fc = nn.Sequential(nn.Dropout(0.0),
                                nn.Linear(in_features=self.audio_model.resnet.fc._modules['1'].in_features + 512*2,
                                          out_features=512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 12),
                                nn.Sigmoid()
                                )

        self.modes = ['clip', 'audio_features']
        self.audio_model.resnet.fc = Dummy()
        # self.video_model.r2plus1d.fc = Dummy()

    def forward(self, clip, audio):
        # audio = x['audio']
        # clip = x['img']
        # (batchsize*seq_num,512)
        audio_model_features = self.audio_model(audio)
        # video_model_features.shape:(batch_size,seq_num,1024)
        video_model_features = self.video_model(clip)
        video_model_features = video_model_features.view(-1, 512*2)

        features = torch.cat(
            [video_model_features, audio_model_features], dim=1)
        # (bacth_size*seq_num,1536)
        out = self.fc(features)
        return out

    # loss function
    @staticmethod
    def multi_label_ACE(outputs, y_labels):
        batch_size, class_size = outputs.size()
        loss_buff = 0
        # pos_weight = [2,3,1,1,1,1,1,5,5,5,1,5]
        # pos_weight = [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        pos_weight = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]
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


class TwoStreamAUAttentionModel(nn.Module):
    def __init__(self, audio_pretrained=False, seq_num=30, encoder_layers=1):
        super(TwoStreamAUAttentionModel, self).__init__()
        self.audio_model = AudioAttentionModel(
            pretrained=audio_pretrained, encoder_layers=encoder_layers)
        # self.video_model = VideoModel(num_channels=num_channels)
        self.video_model = VideoModelAttention(
            seq_num=seq_num, encoder_layers=encoder_layers)
        pretrained_model = torch.load(
            '/data/tianqing/output/best/model_2021-06-17-17-16_acc:0.8787685929713989_f1:0.5328100042602213_bacth:1_step:12053.pth', map_location='cuda:3')
        video_net_dict = self.video_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.items() if k in video_net_dict}
        video_net_dict.update(pretrained_dict)
        self.video_model.load_state_dict(video_net_dict)

        self.fc = nn.Sequential(nn.Dropout(0.0),
                                nn.Linear(in_features=self.audio_model.resnet.fc._modules['1'].in_features + 512*2,
                                          out_features=512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 12),
                                nn.Sigmoid()
                                )

        # self.modes = ['clip', 'audio_features']
        self.audio_model.resnet.fc = Dummy()
        # self.video_model.r2plus1d.fc = Dummy()

    def forward(self, clip, audio):
        # audio = x['audio']
        # clip = x['img']
        # (batchsize*seq_num,512)
        audio_model_features = self.audio_model(audio)
        audio_model_features = audio_model_features.view(-1, 512)
        # video_model_features.shape:(batch_size,seq_num,1024)
        video_model_features = self.video_model(clip)
        video_model_features = video_model_features.view(-1, 512*2)

        features = torch.cat(
            [video_model_features, audio_model_features], dim=1)
        # (bacth_size*seq_num,1536)
        out = self.fc(features)
        return out

    # loss function
    @staticmethod
    def multi_label_ACE(outputs, y_labels):
        batch_size, class_size = outputs.size()
        loss_buff = 0
        # pos_weight = [2,3,1,1,1,1,1,5,5,5,1,5]
        # pos_weight = [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        pos_weight = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]
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


class TwoStreamFusionAUAttentionModel(nn.Module):
    def __init__(self, audio_pretrained=False, seq_num=30, encoder_layers=3, is_fc=False):
        super(TwoStreamFusionAUAttentionModel, self).__init__()
        self.is_fc = is_fc
        self.audio_model = AudioModel(pretrained=audio_pretrained)
        self.audio_model.resnet.fc = Dummy()

        self.video_model = VideoModel(
            seq_num=seq_num, encoder_layers=encoder_layers)
        pretrained_model = torch.load(
            '/data/tianqing/output/best/model_2021-06-17-17-16_acc:0.8787685929713989_f1:0.5328100042602213_bacth:1_step:12053.pth'
        )
        # ,map_location='cuda:3')
        video_net_dict = self.video_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.items() if k in video_net_dict}
        video_net_dict.update(pretrained_dict)
        self.video_model.load_state_dict(video_net_dict)

        self.encoder = Encoder(
            n_src_vocab=None, n_position=seq_num,
            d_word_vec=512*2, d_model=512*2, d_inner=2048,
            n_layers=encoder_layers, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=False)

        self.feature_fc = nn.Linear(512*2, 512*2)

        self.fc = nn.Sequential(nn.Dropout(0.0),
                                nn.Linear(in_features=512*2,
                                          out_features=512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 12),
                                nn.Sigmoid()
                                )
        self.seq_num = seq_num

    def forward(self, clip, audio):
        audio_features = self.audio_model(audio)
        audio_features = audio_features.view(-1, 512)

        # video_model_features.shape:(batch_size,seq_num,1024)
        video_features = self.video_model(clip)
        video_features = video_features.view(-1, 512)

        fusion_features = torch.cat([video_features, audio_features], dim=1)
        if self.is_fc:
            fusion_features = self.feature_fc(fusion_features)
        fusion_features = fusion_features.view(-1, self.seq_num, 512*2)
        src_mask = None
        seq_features = self.encoder(fusion_features, src_mask)

        logits = self.fc(seq_features.contiguous().view(-1, 512*2))
        return logits

    # loss function
    @staticmethod
    def multi_label_ACE(outputs, y_labels):
        batch_size, class_size = outputs.size()
        loss_buff = 0
        # pos_weight = [2,3,1,1,1,1,1,5,5,5,1,5]
        # pos_weight = [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        pos_weight = [1, 2, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]
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


class TwoStreamFusionTDNN(nn.Module):
    def __init__(self, audio_pretrained=False, seq_num=30, encoder_layers=3, is_fc=False):
        super(TwoStreamFusionTDNN, self).__init__()
        self.audio_model = AudioFTDNNModel(pretrained=audio_pretrained)
        # self.audio_model.resnet.fc = Dummy()

        self.video_model = VideoModel(
            seq_num=seq_num, encoder_layers=encoder_layers)
        pretrained_model = torch.load(
            '/data/tianqing/output/best/model_2021-06-17-17-16_acc:0.8787685929713989_f1:0.5328100042602213_bacth:1_step:12053.pth'
        )
        # ,map_location='cuda:3')
        video_net_dict = self.video_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.items() if k in video_net_dict}
        video_net_dict.update(pretrained_dict)
        self.video_model.load_state_dict(video_net_dict)

        self.encoder = Encoder(
            n_src_vocab=None, n_position=seq_num,
            d_word_vec=512*2, d_model=512*2, d_inner=2048,
            n_layers=encoder_layers, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=False)

        self.feature_fc = nn.Linear(512*2, 512*2)

        self.fc = nn.Sequential(nn.Dropout(0.0),
                                nn.Linear(in_features=512*2,
                                          out_features=512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 12),
                                nn.Sigmoid()
                                )
        self.seq_num = seq_num
        self.is_fc = is_fc

    def forward(self, clip, audio, frame_list, batch_size=8, cuda_num=0):
        audio_features = self.audio_model(audio)
        # audio_features_ = torch.zeros(batch_size, self.seq_num, 512).cuda(cuda_num)
        # (8,101,512)
        # (8,30,512)
        # audio_features = audio_features.view(-1, 512)
        # for i in range(8):
        #     for j in range(len(frame_list)):
        #         audio_features_[i, j, :] = audio_features[i,int((frame_list[i][j]-frame_list[i][0])*100/30),:]
        #     if int((frame_list[i]-frame_list[0])*200/30)+8 > audio_features.shape[2]-1:
        #         audio_features_[:, i, :, :] = audio_features[:,:,-8:]
        #     else:
        #         audio_features_[i, :, :, :] = audio_features[:,:,
        #                                         int((frame_list[i]-frame_list[0])*200/30): int((frame_list[i]-frame_list[0])*200/30)+8]
        #     if audio_features_[i, :, :, :].shape[2]<8:
        #         audio_features_[i, :, :, :] = audio_features[:,:,-8:]
        audio_features_ = audio_features.swapaxes(1, 2)
        audio_features_ = F.interpolate(
            audio_features_, size=30, mode='nearest')
        # audio_features_ = F.interpolate(
        #     audio_features_, scale_factor=30/100, mode='nearest')
        audio_features_ = audio_features_.swapaxes(1, 2)
        audio_features_ = audio_features_.reshape(-1, 512)

        # video_model_features.shape:(batch_size,seq_num,1024)
        video_features = self.video_model(clip)
        video_features = video_features.reshape(-1, 512)

        fusion_features = torch.cat([video_features, audio_features_], dim=1)
        if self.is_fc:
            fusion_features = self.feature_fc(
                fusion_features.reshape(-1, self.seq_num, 512*2))
        else:
            fusion_features = fusion_features.reshape(-1, self.seq_num, 512*2)
        # fusion_features = self.feature_fc(fusion_features.view(-1, self.seq_num, 512*2))

        src_mask = None
        seq_features = self.encoder(fusion_features, src_mask)

        logits = self.fc(seq_features.contiguous().view(-1, 512*2))
        return logits

    # loss function
    @staticmethod
    def multi_label_ACE(outputs, y_labels):
        batch_size, class_size = outputs.size()
        loss_buff = 0
        # pos_weight = [2,3,1,1,1,1,1,5,5,5,1,5]
        # pos_weight = [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]
        pos_weight = [1, 2, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]
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
        for p in self.parameters():
            p.requires_grad = False  # 预训练模型加载进来后全部设置为不更新参数，然后再后面加层
        self.fc1 = Sequential(BatchNorm2d(512),
                              Dropout(0.6),
                              Flatten(),
                              Linear(512 * 7 * 7, 512),
                              BatchNorm1d(512),
                              nn.ReLU()
                              )
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
        pos_weight = [1, 2, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]
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
        self.body = Sequential(*modules)
        # self.roi = ROILayer(cuda_num)
        for p in self.parameters():
            p.requires_grad = False  # 预训练模型加载进来后全部设置为不更新参数，然后再后面加层

        self.fc1 = Sequential(BatchNorm2d(512),
                              Dropout(0.6),
                              Flatten(),
                              Linear(512 * 7 * 7, 512),
                              BatchNorm1d(512),
                              nn.ReLU()
                              )
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
        total_features = torch.cat((out2, encoder_features), 2)
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
        pos_weight = [1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 2]
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
