import os
from easydict import EasyDict as edict
import torch
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.train_data_path = '/mnt/data3/jinyue/dataset/opensource/AffWild2/cropped_aligned'
    conf.expr_train_annot_path = '/mnt/data3/jinyue/dataset/opensource/AffWild2/expression_annots_train2.csv'
    conf.expr_valid_annot_path = '/mnt/data3/jinyue/dataset/opensource/AffWild2/expression_annots_valid2.csv'
    conf.train_annot_path = ''
    conf.valid_annot_path = ''
    conf.work_path = '/mnt/data3/jinyue/code/ABAW2/ABAW2_logs'
    conf.exp_name = 'v4.3.1_2branhces_test_swap_epoch_train_valid2/'
    conf.log_path = os.path.join(conf.work_path, conf.exp_name, 'log')
    conf.save_path = os.path.join(conf.work_path, conf.exp_name, 'save')
    # conf.model_path = os.path.join(conf.save_path, 'model_2021-06-08-00-36_loss:0.062315932410054425_step:60900.pth')
#--------------------model config-----------------------------------
    conf.model = 'iresnet100'  # iresnetse50, iresnet100, resnextse101

    # if conf.model == 'iresnet100':
    #     conf.pretrain_path = os.path.join('output', 'pretrain', 'backbone_glint360k_cosface_r100_fp16_0.1.pth')
    # elif conf.model == 'iresnetse50':
    #     conf.pretrain_path = os.path.join('output', 'pretrain', 'model_ir_se50.pth')
    # elif conf.model == 'resnextse101':
    #     conf.pretrain_path = os.path.join('output', 'pretrain', 'ResNeXtSE101.pth')

    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
#--------------------Training Config ------------------------
    if training:
        conf.expr_class_num = 7
        conf.au_class_num = 12
        conf.expr_loss = 'focal'
        conf.au_loss = 'bce'
        conf.batch_size = 64
        conf.epochs = 15
        conf.lr = 0.001
        # conf.milestones = [8,10,12]
        conf.milestones = [15, 15, 15]
        conf.momentum = 0.9
        conf.weight_decay = 5e-4
        conf.pin_memory = True
        conf.num_workers = 4
        conf.sample = False
        conf.optimizer = 'SGD'
        conf.eval_or_not = True
#--------------------Inference Config ------------------------
    else:
        conf.batch_size = 64
        conf.pin_memory = True
        conf.num_workers = 1
    return conf
