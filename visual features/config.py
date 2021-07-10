import os
from easydict import EasyDict as edict
import torch
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.train_data_path = '/data/jinyue/dataset/opensource/AffWild2/cropped_aligned'
    conf.pseudo_train_data_path = '/data/jinyue/dataset/lipread_mp4/images'
    conf.expr_train_annot_path = '/data/jinyue/dataset/opensource/AffWild2/Annotations/expression_annots_train.csv'
    conf.expr_valid_annot_path = '/data/jinyue/dataset/opensource/AffWild2/Annotations/expression_annots_valid.csv'
    conf.work_path = '/data/jinyue/logs/'
    conf.exp_name = 'v2.0.9_IResnet100-dropout0.6_focal-pytorch_sgd0.001_basic+color_nosample-modeleval'
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
    conf.class_num = 7
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
        conf.loss = 'focal'
        conf.eval_or_not = True
#--------------------Inference Config ------------------------
    else:
        conf.batch_size = 64
        conf.pin_memory = True
        conf.num_workers = 1
    return conf
