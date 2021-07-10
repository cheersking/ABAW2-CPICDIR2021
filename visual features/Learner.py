import os

from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from utils import separate_bn_paras, get_time
from data.data_pipe import train_data_loader, val_data_loader
from models.iresnetse import IrResNetSe
from models.iresnet import iresnet100, iresnet100_nohead
from models.se_resnext import se_resnext_101
from loss.focal_loss import FocalLossV2
from loss.amsoftmax import AMSoftmax


class EmotionLearner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        self.conf = conf

        # build model
        if conf.model == 'iresnetse50':
            self.model = IrResNetSe(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{} model generated'.format(conf.model))
        elif conf.model == 'iresnet100':
            self.model = iresnet100(dropout=conf.drop_ratio, num_class=conf.class_num).to(conf.device)
        elif conf.model == 'resnextse101':
            self.model = se_resnext_101().to(conf.device)
        elif conf.model == 'iresnet100_nohead':
            self.model = iresnet100_nohead(dropout=conf.drop_ratio, num_class=conf.class_num).to(conf.device)

        if not inference:
            self.eval_or_not = conf.eval_or_not
            self.milestones = conf.milestones
            self.loader = train_data_loader(conf)
            self.eval_loader = val_data_loader(conf)
            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            # self.head = Am_softmax(embedding_size=conf.embedding_size,
            #                     classnum=conf.class_num).to(conf.device)

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.optimizer == 'SGD':
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn, 'weight_decay': conf.weight_decay},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            elif conf.optimizer == 'RMSprop':
                self.optimizer = torch.optim.RMSprop([
                                    {'params': paras_wo_bn, 'weight_decay': conf.weight_decay},
                                    {'params': paras_only_bn}],
                                    lr=conf.lr, alpha=0.9)
            elif conf.optimizer == 'Adam':
                self.optimizer = torch.optim.Adam([
                                    {'params': paras_wo_bn, 'weight_decay': conf.weight_decay},
                                    {'params': paras_only_bn}],
                                    lr=conf.lr, betas=(0.9, 0.99))
            elif conf.optimizer == 'Adadelta':
                self.optimizer = torch.optim.Adadelta([
                                    {'params': paras_wo_bn, 'weight_decay': conf.weight_decay},
                                    {'params': paras_only_bn}],
                                    rho=0.9)
            # self.optimizer = optim.SGD((param for param in self.model.parameters()
            #                    if param.requires_grad), lr = conf.lr, momentum = conf.momentum)

            print('optimizers generated')
            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//10
            self.save_every = len(self.loader)//2
        else:
            self.loader = val_data_loader(conf)

        # loss
        if self.conf.loss == 'ce_loss':
            self.loss = nn.CrossEntropyLoss()
        elif self.conf.loss == 'focal':
            # focal loss pytorch
            self.loss = FocalLossV2()
        elif self.conf.loss == 'amsoftmax':
            self.loss = AMSoftmax(in_feats=512, n_classes=7, conf=self.conf)

    def save_state(self, loss, accuracy, f1, to_save_folder=True, extra='1', model_only=False):
        if to_save_folder:
            save_path = self.conf.save_path
        else:
            save_path = self.conf.pretrain_path
        torch.save(
            self.model.state_dict(), os.path.join(save_path,
                ('model_{}_loss-{:.3f}_acc-{:.3f}_f1-{:.3f}_step-{}.pth'.format(get_time(), loss, accuracy, f1, self.step))))
        if not model_only:
            torch.save(
                self.optimizer.state_dict(), os.path.join(save_path,
                    ('optimizer_{}_loss-{:.3f}_acc-{:.3f}_f1-{:.3f}_step-{}.pth'.format(get_time(), loss, accuracy, f1, self.step))))

    def load_state(self, fixed_str='', use_pretrain=True, model_only=True):
        if use_pretrain:
            save_path = self.conf.pretrain_path
            pretrained_dict = torch.load(save_path)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            model_path = self.conf.model_path
            self.model.load_state_dict(torch.load(model_path))

            if not model_only:
                self.optimizer.load_state_dict(torch.load(os.path.join(self.conf.save_path, fixed_str)))


    def train(self):
        self.model.train()
        running_loss = 0.
        for e in range(self.conf.epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(self.conf.device)
                labels = labels.to(self.conf.device)
                self.optimizer.zero_grad()
                output = self.model(imgs)

                if self.conf.loss == 'focal':
                    labels = torch.nn.functional.one_hot(labels, self.conf.class_num)
                loss = self.loss(output, labels)

                loss.backward()
                print('loss: ', loss.item())
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                    # print('loss: ', loss_board)

                if self.step % self.save_every == 0 and self.step != 0:
                    if self.eval_or_not:
                        self.model.eval()
                        acc, totalf1 = self.eval()
                    else:
                        acc, totalf1 = 1, 1
                    self.save_state(loss=loss_board, accuracy=acc, f1=totalf1)
                    self.model.train()

                self.step += 1


    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)


    def eval(self):
        predicted = []
        true_labels = []
        correct = 0
        total = 0
        for imgs, labels in tqdm(iter(self.eval_loader)):
            imgs = imgs.to(self.conf.device)
            labels = labels.to(self.conf.device)
            logits = self.model(imgs)
            predicts = logits.argmax(dim=1)
            cmp = predicts.eq(labels).cpu().numpy()
            correct += cmp.sum()
            total += len(cmp)
            predicted += predicts.cpu().numpy().tolist()
            true_labels += labels.cpu().numpy().tolist()
        acc = correct / total
        total_f1 = f1_score(true_labels, predicted, average='macro')

        return acc, total_f1
