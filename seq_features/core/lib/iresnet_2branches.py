import torch
from torch import nn
import transformer.Constants as Constants
from transformer.Models import Transformer, Encoder
from transformer.Optim import ScheduledOptim

__all__ = ['iresnet100_2branches']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet2Branches(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False, expr_class=7, au_class=12):
        super(IResNet2Branches, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion *
                            self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self.expr_head = nn.Linear(num_features, expr_class)
        self.au_head = nn.Linear(num_features, au_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        # expr_out = self.expr_head(x)
        # au_out = self.au_head(x)
        # return expr_out, au_out
        return x 


class IResNet2BranchesAttention(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False, expr_class=7, au_class=12,
                 encoder_layers=6, seq_num=30, num_classes=12):
        super(IResNet2BranchesAttention, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion *
                            self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self.expr_head = nn.Linear(num_features, expr_class)
        self.au_head = nn.Linear(num_features, au_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        for p in self.parameters():
            p.requires_grad = False
            
        self.encoder = Encoder(
            n_src_vocab=None, n_position=seq_num,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=encoder_layers, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=False)

        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes),
                                 nn.Sigmoid())
        self.seq_num = seq_num

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        out2 = x.view(-1, self.seq_num, 512)
        src_mask = None
        encoder_features = self.encoder(out2, src_mask)
        output = self.fc2(encoder_features.contiguous().view(-1, 512))

        # expr_out = self.expr_head(x)
        # au_out = self.au_head(x)
        # return expr_out, au_out
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

class IResNet2BranchesAttentionAdd(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False, expr_class=7, au_class=12,
                 encoder_layers=1, seq_num=30, num_classes=12):
        super(IResNet2BranchesAttentionAdd, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion *
                            self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        ########################################################
        # for p in self.parameters():
        #     p.requires_grad = False
        ########################################################
        self.expr_head = nn.Linear(num_features, expr_class)
        self.au_head = nn.Linear(num_features, au_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        for p in self.parameters():
            p.requires_grad = False
            
        self.encoder = Encoder(
            n_src_vocab=None, n_position=seq_num,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=encoder_layers, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=False)

        self.fc2 = nn.Sequential(nn.Linear(in_features=512*2, out_features=num_classes),
                                 nn.Sigmoid())
        self.seq_num = seq_num

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        out2 = x.view(-1, self.seq_num, 512)
        src_mask = None
        encoder_features = self.encoder(out2, src_mask)
        #concat features and encoder_features
        total_features = torch.cat((out2,encoder_features ), 2)
        output = self.fc2(total_features.contiguous().view(-1, 512*2))

        # expr_out = self.expr_head(x)
        # au_out = self.au_head(x)
        # return expr_out, au_out
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



def _iresnet_2branches(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet2Branches(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet100_2branches(pretrained=False, progress=True, **kwargs):
    return _iresnet_2branches('iresnet100_2branches', IBasicBlock, [3, 13, 30, 3], pretrained,
                              progress, **kwargs)


def _iresnet_2branches_attention(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet2BranchesAttention(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model

def _iresnet_2branches_attention_add(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet2BranchesAttentionAdd(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet100_2branches_attention(pretrained=False, progress=True, **kwargs):
    return _iresnet_2branches_attention('iresnet100_2branches', IBasicBlock, [3, 13, 30, 3], pretrained,
                                        progress, **kwargs)


def iresnet100_2branches_attention_add(pretrained=False, progress=True, **kwargs):
    return _iresnet_2branches_attention_add('iresnet100_2branches', IBasicBlock, [3, 13, 30, 3], pretrained,
                                        progress, **kwargs)
