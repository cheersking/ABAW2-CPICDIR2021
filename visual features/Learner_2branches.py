
from loss.amsoftmax import AMSoftmax
from loss.losses import CosFace
from loss.focal_loss import FocalLossV2
from utils import separate_bn_paras, get_time
from models.iresnet_2branches import iresnet100_2branches
from models.se_resnext import se_resnext_101
from models.iresnet import iresnet100
from models.iresnetse import IrResNetSe
from models.iresnet2060_2branches import iresnet2060_2branches
from data.data_pipe_2branches import train_data_loader, val_data_loader, au_dataloader
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
import os

import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from itertools import cycle
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')


class TwoBranchLearner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        self.conf = conf

        # build model
        if conf.model == 'iresnetse50':
            self.model = IrResNetSe(
                conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{} model generated'.format(conf.model))
        elif conf.model == 'iresnet100_2branches':
            self.model = iresnet100_2branches(
                dropout=conf.drop_ratio).to(conf.device)
        elif conf.model == 'resnextse101':
            self.model = se_resnext_101().to(conf.device)
        elif conf.model == 'iresnet2060_2branches':
            self.model = iresnet2060_2branches().to(conf.device)

        if not inference:
            self.eval_or_not = conf.eval_or_not
            self.milestones = conf.milestones
            self.expr_train_loader = train_data_loader(conf)
            self.expr_eval_loader = val_data_loader(conf)
            self.au_train_loader = au_dataloader(conf.train_data_path, conf.train_annot_path,
                                                 batch_size=conf.batch_size, is_training=True,
                                                 pred_txt_file=None, flag='train')
            self.au_eval_loader = au_dataloader(conf.train_data_path, conf.valid_annot_path,
                                                batch_size=conf.batch_size, is_training=False,
                                                pred_txt_file=None, flag='valid')

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.optimizer == 'SGD':
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn, 'weight_decay': conf.weight_decay},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
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
            self.board_loss_every = len(self.expr_train_loader)//100
            self.evaluate_every = len(self.expr_train_loader)//10
            self.save_every = len(self.expr_train_loader)//2
            self.au_step = 0
            self.au_board_loss_every = len(self.au_train_loader)//100
            self.au_evaluate_every = len(self.au_train_loader)//10
            self.au_save_every = len(self.au_train_loader)//2

        else:
            self.expr_eval_loader = val_data_loader(conf)

        # expr_loss
        if self.conf.expr_loss == 'ce_loss':
            self.expr_loss = nn.CrossEntropyLoss()
        elif self.conf.expr_loss == 'focal':
            self.expr_loss = FocalLossV2()
        elif self.conf.expr_loss == 'amsoftmax':
            self.expr_loss = AMSoftmax(
                in_feats=512, n_classes=7, conf=self.conf)

        # au loss
        if self.conf.au_loss == 'bce':
            self.au_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor(
                [1, 2, 1, 1, 1, 1, 1, 6, 6, 5, 1, 5]).to(self.conf.device))
        elif self.conf.au_loss == 'focal':
            self.au_loss = FocalLossV2()

    def save_state(self, loss, accuracy, f1, epoch, to_save_folder=True, annot='expr', model_only=False):
        if to_save_folder:
            save_path = self.conf.save_path
        else:
            save_path = self.conf.pretrain_path
        torch.save(
            self.model.state_dict(), os.path.join(save_path,
                                                  ('model_{}_{}_loss-{:.3f}_acc-{:.3f}_f1-{:.3f}_step-{}_epoch-{}.pth'.format(
                                                      annot, get_time(), loss, accuracy, f1, self.step, epoch))))
        if not model_only:
            torch.save(
                self.optimizer.state_dict(), os.path.join(save_path,
                                                          ('optimizer_{}_{}_loss-{:.3f}_acc-{:.3f}_f1-{:.3f}_step-{}_epoch-{}.pth'.format(
                                                              annot, get_time(), loss, accuracy, f1, self.step, epoch))))

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
                self.optimizer.load_state_dict(torch.load(
                    os.path.join(self.conf.save_path, fixed_str)))

    def train_swap_epoch(self):
        '''
        learner to train expression and au jointly
        epoch by epoch
        e.g. epoch1: expr, epoch2: au, epoch3: expr, ...
        '''
        self.model.train()
        running_loss_expr = 0.
        running_loss_au = 0.
        for e in range(self.conf.epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            for imgs, labels in tqdm(iter(self.expr_train_loader)):
                imgs = imgs.to(self.conf.device)
                labels = labels.to(self.conf.device)
                self.optimizer.zero_grad()
                output = self.model(imgs)

                if self.conf.expr_loss == 'focal':
                    labels = torch.nn.functional.one_hot(
                        labels, self.conf.expr_class_num)
                expr_loss = self.expr_loss(output[0], labels)

                expr_loss.backward()
                print('expr_loss: ', expr_loss.item())
                running_loss_expr += expr_loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    expr_loss_board = running_loss_expr / self.board_loss_every
                    self.writer.add_scalar(
                        'expr_loss', expr_loss_board, self.step)
                    running_loss_expr = 0.
                    # print('loss: ', loss_board)

                # if self.step % self.save_every == 0 and self.step != 0:
                #     if self.eval_or_not:
                #         self.model.eval()
                #         acc, totalf1 = self.expr_eval(self.expr_eval_loader)
                #     else:
                #         acc, totalf1 = 1, 1
                #     self.save_state(loss=expr_loss_board, accuracy=acc,
                #                     f1=totalf1, epoch=e, annot='expr')
                #     self.model.train()

                self.step += 1

            if self.eval_or_not:
                self.model.eval()
                acc, totalf1 = self.expr_eval(self.expr_eval_loader)
            else:
                acc, totalf1 = 1, 1
            self.save_state(loss=expr_loss_board, accuracy=acc,
                            f1=totalf1, epoch=e, annot='expr')
            self.model.train()

            for imgs, labels in tqdm(iter(self.au_train_loader)):
                imgs = imgs.to(self.conf.device)
                labels = labels.to(self.conf.device)
                self.optimizer.zero_grad()
                output = self.model(imgs)
                au_loss = self.au_loss(output[1], labels.float())
                mask = (labels < 9).type_as(labels)
                au_loss = (au_loss * mask).mean()
                au_loss.backward()
                print('au loss: ', au_loss.item())
                running_loss_au += au_loss.item()
                self.optimizer.step()

                if self.au_step % self.au_board_loss_every == 0 and self.au_step != 0:
                    au_loss_board = running_loss_au / self.au_board_loss_every
                    self.writer.add_scalar('au_loss', au_loss_board, self.au_step)
                    running_loss_au = 0.

                # if self.au_step % self.au_save_every == 0 and self.au_step != 0:
                #     if self.eval_or_not:
                #         self.model.eval()
                #         acc, totalf1 = self.eval_au(self.au_eval_loader)
                #     else:
                #         acc, totalf1 = 1, 1
                #     self.save_state(loss=au_loss_board, accuracy=acc,
                #                     f1=totalf1, epoch=e, annot='au')
                #     self.model.train()

                self.au_step += 1

            if self.eval_or_not:
                self.model.eval()
                acc, totalf1 = self.eval_au(self.au_eval_loader)
            else:
                acc, totalf1 = 1, 1
            self.save_state(loss=au_loss_board, accuracy=acc,
                            f1=totalf1, epoch=e, annot='au')
            self.model.train()

    def train_swap_batch(self):
        '''
        learner to train expression and au jointly
        batch by batch
        e.g. batch1: expr, batch2: au, batch3: expr, ...
        '''
        self.model.train()
        running_loss_expr = 0.
        running_loss_au = 0.
        for e in range(self.conf.epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()

            for loader_1, loader_2 in zip(cycle(self.expr_train_loader), self.au_train_loader):
                # expr batch
                expr_imgs = loader_1[0]
                expr_labels = loader_1[1]

                expr_imgs = expr_imgs.to(self.conf.device)
                expr_labels = expr_labels.to(self.conf.device)
                self.optimizer.zero_grad()
                output = self.model(expr_imgs)

                if self.conf.expr_loss == 'focal':
                    expr_labels = torch.nn.functional.one_hot(
                        expr_labels, self.conf.expr_class_num)
                expr_loss = self.expr_loss(output[0], expr_labels)

                expr_loss.backward()
                print('expr_loss: ', expr_loss.item())
                running_loss_expr += expr_loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    expr_loss_board = running_loss_expr / self.board_loss_every
                    self.writer.add_scalar(
                        'expr_loss', expr_loss_board, self.step)
                    running_loss_expr = 0.
                    # print('loss: ', loss_board)

                if self.step % self.save_every == 0 and self.step != 0:
                    if self.eval_or_not:
                        self.model.eval()
                        acc, totalf1 = self.expr_eval(self.expr_eval_loader)
                    else:
                        acc, totalf1 = 1, 1
                    self.save_state(loss=expr_loss_board, accuracy=acc,
                                    f1=totalf1, epoch=e, annot='expr')
                    self.model.train()

                # au batch
                au_imgs = loader_2[0]
                au_labels = loader_2[1]
                au_imgs = au_imgs.to(self.conf.device)
                au_labels = au_labels.float().to(self.conf.device)
                self.optimizer.zero_grad()
                output = self.model(au_imgs)
                au_loss = self.au_loss(output[1], au_labels)
                mask = (au_labels < 9).type_as(au_labels)
                au_loss = (au_loss * mask).mean()
                au_loss.backward()
                print('au loss: ', au_loss.item())
                running_loss_au += au_loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    au_loss_board = running_loss_au / self.board_loss_every
                    self.writer.add_scalar('au_loss', au_loss_board, self.step)
                    running_loss_au = 0.

                if self.step % self.save_every == 0 and self.step != 0:
                    if self.eval_or_not:
                        self.model.eval()
                        acc, totalf1 = self.eval_au(self.au_eval_loader)
                    else:
                        acc, totalf1 = 1, 1
                    self.save_state(loss=au_loss_board, accuracy=acc,
                                    f1=totalf1, epoch=e, annot='au')
                    self.model.train()

                self.step += 1

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

    def expr_eval(self, eval_loader):
        predicted = []
        true_labels = []
        correct = 0
        total = 0
        for imgs, labels in tqdm(iter(eval_loader)):
            imgs = imgs.to(self.conf.device)
            labels = labels.to(self.conf.device)
            logits = self.model(imgs)[0]
            predicts = logits.argmax(dim=1)
            cmp = predicts.eq(labels).cpu().numpy()
            correct += cmp.sum()
            total += len(cmp)
            predicted += predicts.cpu().numpy().tolist()
            true_labels += labels.cpu().numpy().tolist()
        acc = correct / total
        total_f1 = f1_score(true_labels, predicted, average='macro')
        return acc, total_f1

    def eval_au(self, eval_loader):
        self.model.eval()
        predicted = []
        true_labels = []
        correct = 0
        total = 0
        for imgs, labels in tqdm(iter(eval_loader)):
            imgs = imgs.to(self.conf.device)
            labels = labels.to(self.conf.device)
            logits = self.model(imgs)[1]
            # predicts = logits.argmax(dim=1)
            predicts = torch.greater(logits, 0).type_as(labels)
            cmp = predicts.eq(labels).cpu().numpy()
            correct += cmp.sum()
            total += len(cmp) * 12
            predicted += predicts.cpu().numpy().tolist()
            true_labels += labels.cpu().numpy().tolist()

        acc = correct / total
        print('acc:', acc)
        total_f1 = f1_score(true_labels, predicted, average='macro')
        print('f1:', total_f1)
        class_names = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12',
                       'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
        print(classification_report(true_labels,
              predicted, target_names=class_names))
        return acc, total_f1
