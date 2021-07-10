import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from config_2branches import get_config
from Learner_2branches import TwoBranchLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--drop_ratio', type=float, default=0.6, help='dropout ratio')
    parser.add_argument('--exp_name', type=str, default='test', help='exp name')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer name')
    parser.add_argument('--model', type=str, default='iresnet100_2branches', help='network name')
    parser.add_argument('--expr_loss', type=str, default='focal', help='loss name')
    parser.add_argument('--au_loss', type=str, default='bce', help='loss name')
    parser.add_argument('--milestones', type=int, nargs='+', default=[8, 10, 12], help='milestones')
    parser.add_argument('--eval_or_not', type=bool, default=True, help='eval or not')
    parser.add_argument('--mode', type=str, default='swap_epoch', help='joint learning mode')
    args = parser.parse_args()

    conf = get_config()

    conf.drop_ratio = args.drop_ratio
    conf.lr = args.lr
    conf.weight_decay = args.weight_decay
    conf.batch_size = args.batch_size
    conf.optimizer = args.optimizer
    conf.model = args.model
    conf.expr_loss = args.expr_loss
    conf.au_loss = args.au_loss
    conf.eval_or_not = args.eval_or_not
    conf.milestones = args.milestones

    if conf.model == 'iresnet100':
        conf.pretrain_path = os.path.join('output', 'pretrain', 'backbone_glint360k_cosface_r100_fp16_0.1.pth')
    elif conf.model == 'iresnet100_nohead':
        conf.pretrain_path = os.path.join('output', 'pretrain', 'backbone_glint360k_cosface_r100_fp16_0.1.pth')
    elif conf.model == 'iresnetse50':
        conf.pretrain_path = os.path.join('output', 'pretrain', 'model_ir_se50.pth')
    elif conf.model == 'resnextse101':
        conf.pretrain_path = os.path.join('output', 'pretrain', 'ResNeXtSE101.pth')
    elif conf.model == 'iresnet100_2branches':
        conf.pretrain_path = os.path.join('output', 'pretrain', 'backbone_glint360k_cosface_r100_fp16_0.1.pth')
    elif conf.model == 'iresnet2060_2branches':
        conf.pretrain_path = os.path.join('output', 'pretrain', 'backbone_iresnet2060.pth')


    if args.exp_name != 'test':
        conf.exp_name = args.exp_name
        conf.log_path = os.path.join(conf.work_path, conf.exp_name, 'log')
        conf.save_path = os.path.join(conf.work_path, conf.exp_name, 'save')

    print(conf)

    # create folder
    if not os.path.exists(conf.log_path):
        os.makedirs(conf.log_path)
    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)

    learner = TwoBranchLearner(conf)
    learner.load_state()
    if args.mode == 'swap_epoch':
        learner.train_swap_epoch()
    elif args.mode == 'swap_batch':
        learner.train_swap_batch()
