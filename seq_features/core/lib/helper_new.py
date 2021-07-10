import os,json,random
import numpy as np
# process all images names -> {'F001_T1':[2440,2441,...],'F002_T2':[],...,'F002_T3':[],...}
def get_im_dict():
    im_dic = {}
    with open('/data/tian/developer/ROI-Nets-pytorch/data/train_images.txt') as f:
        all_im = f.readlines()
        for l in all_im:
            if l.split('/')[4]=='ABAW2_Competition':
                person = l.split('/')[6]
                num = int(l.split('/')[-1][:-5])#\n
                if person not in im_dic:
                    im_dic[person]=[]
                    im_dic[person].append(num)
                else:
                    im_dic[person].append(num)
            else:
                person = l.split('/')[5]+'/'+l.split('/')[6]
                num = int(l.split('/')[-1][:-5])
                if person not in im_dic:
                    im_dic[person]=[]
                    im_dic[person].append(num)
                else:
                    im_dic[person].append(num)
        # for k in im_dic.keys():
        #     im_dic[k].sort()
        im_dic_valid = {}
        with open('/data/tian/developer/ROI-Nets-pytorch/data/valid_images.txt') as f:
    # with open('../data/all_images.txt') as f:
            all_im = f.readlines()
            for l in all_im:
                # if l.split('/')[4]=='ABAW2_Competition':
                person = l.split('/')[0]
                num = int(l.split('/')[-1][:-5])#\n
                if person not in im_dic_valid:
                    im_dic_valid[person]=[]
                    im_dic_valid[person].append(num)
                else:
                    im_dic_valid[person].append(num)
                # else:
                #     person = l.split('/')[5]+'/'+l.split('/')[6]
                #     num = int(l.split('/')[-1][:-5])
                #     if person not in im_dic_valid:
                #         im_dic_valid[person]=[]
                #         im_dic_valid[person].append(num)
                #     else:
                #         im_dic_valid[person].append(num)

    return im_dic,im_dic_valid

# process labels.txt -> labels: {im_name:[labels]}, landmarks: {im_name:[landmark]}
def get_label_landmark():
    if not os.path.exists('/data/tian/developer/ROI-Nets-pytorch/data/train_labels.json'):
        landmarks = {}
        labels = {}
        label_table = np.load('/data/tian/Data/au_label_merged.npy',allow_pickle=True)
        path_table = np.load('/data/tian/Data/au_path_merged_new.npy',allow_pickle=True)
        for i in range(label_table.shape[0]):
            labels[path_table[i]] = list(label_table[i])
        with open('/data/tian/developer/ROI-Nets-pytorch/data/train_labels.json', 'w') as f:
            json.dump(labels, f,indent=1)
    else:
        with open('/data/tian/developer/ROI-Nets-pytorch/data/train_labels.json') as f:
            labels = json.load(f)
    if not os.path.exists('/data/tian/developer/ROI-Nets-pytorch/data/valid_labels.json'):
        # landmarks = {}
        labels_valid = {}
        data =  np.load('/data/tian/Data/ABAW2_Competition/au_valid_list(without-1).npy',allow_pickle=True)
        for i in range(data.shape[0]):
            labels_valid[data[i,0]] = list(data[i,1:])
        with open('/data/tian/developer/ROI-Nets-pytorch/data/valid_labels.json', 'w') as f:
            json.dump(labels_valid, f,indent=1)
    else:
        with open('/data/tian/developer/ROI-Nets-pytorch/data/valid_labels.json') as f:
            labels_valid = json.load(f)
        # with open('/data/tian/developer/ROI-Nets-pytorch/data/landmarks.json') as f:
        #     landmarks = json.load(f)
    return labels,labels_valid

# random split train/test data
def train_test_split():
    if not os.path.exists('../data/train.txt'):
        females = ['F%03d' % i for i in random.sample(range(1, 24), 23)]
        males = ['M%03d' % i for i in random.sample(range(1, 19), 18)]
        train_person = females[0:18] + males[0:15]
        train_person = [i + '_T' + str(j) for j in range(1, 9) for i in train_person]
        random.shuffle(train_person)
        test_person = females[18:] + males[15:]
        test_person = [i + '_T' + str(j) for j in range(1, 9) for i in test_person]
        random.shuffle(test_person)
        with open('./data/train.txt','w') as f:
            f.write(str(train_person))
        with open('./data/test.txt','w') as f:
            f.write(str(test_person))
    else:
        with open('./data/train.txt') as f:
            train_person = eval(f.readlines()[0])
        with open('./data/test.txt') as f:
            test_person = eval(f.readlines()[0])
    return train_person,test_person

if __name__=='__main__':
    get_label_landmark()
    # im_dic,im_dic_valid = get_im_dict()
    # print(im_dic.keys())
    # print(im_dic_valid.keys())
    # print(len(im_dic.keys()))
    # print(len(im_dic_valid.keys()))
    # ret = [ i for i in im_dic_valid.keys() if i in im_dic.keys() ]
    # print(ret)



