import torch
import os
import random
# from torchvision import transforms
from torchvision import transforms as trans
from PIL import Image
from helper_new import get_im_dict,get_label_landmark,train_test_split
from network_new import ROINet
import logging,time,shutil
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
from data_pipe import get_dataloader


save_dir = './models_withoutBP4D_withdropout/lr0.1'  
os.makedirs(save_dir,exist_ok=True)                                                                                          # directory to save models and log
                                                                                             # directory to save models and log
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# else:
#     shutil.rmtree(save_dir)
#     os.mkdir(save_dir)
# log_path = save_dir+'/run_BP4D.log'                                                                                        # log path
image_dir = "your_image_dir"          # images dir
class_number = 12                                                                                                     # number of AUs
cuda_num = 6                                                                                                   # only support signle gpu card
person_batch = 3                                                                                                   # how many sequences in one batch
seq_num = 24                                                                                                          # sequence length
au_thresh = 0.5                                                                                                       # AU thresh, default:0.5
epochs = 50                                                                                                           # number of epochs 
batches = 2000                                                                                                        # how many iterations in one epoch
print_every = 100                                                                                                      # print info every

print('loading...')
# im_dic,im_dic_vaild = get_im_dict()

# # exit()                                                                                                # im_dic: {'F001_T1':[2440,2441,...],'F002_T2':[],...}
# # labels, landmarks = get_label_landmark() 
# labels, labels_valid = get_label_landmark()                                                                       # labels: {im_name:[labels]} landmarks: {im_name:[landmark]}
# # train_person,test_person = train_test_split()                                                                         # train/test_person = ['F001_T1','F001_T2','F001_T3','F001_T4','F001_T5','F001_T6','F001_T7','F001_T8'...]
# train_person,test_person = list(im_dic.keys()),list(im_dic_vaild.keys())
# print(type(train_person))

pretrained_model = torch.load('/data/tian/pretrained_model/model_ir_se50.pth')                                                                    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
net = ROINet(num_classes=class_number,seq_num=seq_num,cuda_num=cuda_num)
net_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_model.items() if k in net_dict}
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)

# for i,p in enumerate(net.parameters()):
#     if i < 16:
#         p.requires_grad = False #fix first 8 conv

if torch.cuda.is_available():
    net.cuda(cuda_num)
    print('cuda available',cuda_num)

print('loaded...')



# transform = transforms.Compose([
#     transforms.Resize(size=(224,224)),
#     transforms.ToTensor(),
#     ])


# learning rate and optimizer
learning_rate = 0.1
opt = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=learning_rate,momentum=0.9,weight_decay=5e-4)

image_dir = '/data/jinyue/dataset/opensource/AffWild2/cropped_aligned'
txt_file ='/data/jinyue/dataset/opensource/AffWild2/Annotations/au_annots_train.csv'
train_loader = get_dataloader(image_dir,txt_file,batch_size=1,is_training=True, pred_txt_file=None,flag='train',mode='without')
#batch_size
# epoch = 0
# acc = 0
# total_f1 = 0




for epoch in range(epochs):
    ### train
    # learning rate decay
    # if epoch in [10,30,60,90]:
    #     learning_rate = learning_rate*0.1
    #     opt = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=0.9,
    #                           nesterov=True)

    tmp_loss = 0                # tmp loss for every print_every iterations
    tmp_statistics_list = []    # tmp statistics list for every print_every iterations
    net.train()                 # train mode
  
    i = 0
    for im_data, im_labels in tqdm(iter(train_loader)):
        i+=1
        im_data = im_data.reshape(-1,3, 112, 112)
        im_labels = im_labels.reshape(-1,12)
        if torch.cuda.is_available():
            im_data = im_data.cuda(cuda_num)
            im_labels = im_labels.cuda(cuda_num)
            # im_landmarks = im_landmarks.cuda(cuda_num)
        # pred = net(im_data,im_landmarks)
        pred = net(im_data,0)
        loss = net.multi_label_ACE(pred,im_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # statistics_list = net.statistics(pred.data, im_labels.data, au_thresh)
        # tmp_statistics_list = net.update_statistics_list(tmp_statistics_list,statistics_list)
        tmp_loss += loss.detach().item()

        if i%print_every == 0:
            # mean_f1_score, f1_score_list = net.calc_f1_score(tmp_statistics_list)
            # f1_score_list = ['%.4f' % f1_score for f1_score in f1_score_list]
            print(tmp_loss/i)
            # mean_f1_score)
            # print(f1_score_list）
            # tmp_statistics_list = []
            # start = time.clock()
    torch.save(net.state_dict(), save_dir+'/params_epoch_%d.pkl' %epoch)  #save params

    ### test
    pred_test = torch.empty((0,12))
    y_test = torch.empty((0,12))
    if torch.cuda.is_available():
        pred_test = pred_test.cuda(cuda_num)
        y_test = y_test.cuda(cuda_num)
    loss_test = 0
    loss_count = 0
    im_names_ = []
    count = 0
    net.eval() #eval mode
    # test_person = test_person[:2]
    val_loader = get_dataloader(image_dir,txt_file,batch_size=2,is_training=False, pred_txt_file=None,flag='test')
    with torch.no_grad():
        for im_data, im_labels in tqdm(iter(train_loader)):
        # for im_data, im_labels in tqdm(iter(val_loader)):
            im_data = im_data.reshape(-1,3, 112, 112)
            im_labels = im_labels.reshape(-1,12)
            if torch.cuda.is_available():
                im_data = im_data.cuda(cuda_num)
                im_labels = im_labels.cuda(cuda_num)
                # im_landmarks = im_landmarks.cuda(cuda_num)
            # pred_ts = net(im_data,im_landmarks)
            pred_ts = net(im_data,0)
            pred_test = torch.cat((pred_test,pred_ts),dim=0)
            y_test = torch.cat((y_test,im_labels),dim=0)

    predicts = torch.greater(pred_test.data, 0.5).type_as(y_test.data)
    cmp = (predicts).eq(y_test.data).cpu().numpy()
    correct = cmp.sum()
    total = len(cmp)*12
    predicted = predicts.cpu().numpy().tolist()
    true_labels = y_test.data.cpu().numpy().tolist()
    #         
    acc = correct / total
    print('acc:',acc)
    total_f1 = f1_score(true_labels, predicted, average='macro')
    print('total_f1:',total_f1)
    class_names = ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26']
    print(classification_report(true_labels, predicted, target_names=class_names))
    # torch.save(net.state_dict(), save_dir+'/params_epoch_%d_acc_%.04f_f1_%.04f.pkl' %epoch %acc %total_f1)  #save params
    # torch.save(net.state_dict(), f'{save_dir}/params_epoch_{epoch}_acc_{acc:.04f}_f1_{total_f1:.04f}.pkl')  #save params
    torch.save(net.state_dict(), f'{save_dir}/score_{0.5*acc+0.5*total_f1:.04f}params_epoch_{epoch}_acc_{acc:.04f}_f1_{total_f1:.04f}.pkl')  #save params

   

