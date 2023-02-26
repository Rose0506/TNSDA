import scipy.io as sio
import os
import numpy as np
import pandas as pd
import pickle
from preprocess import pre_normalization

# yFile = 'D:/Skeleton joint dataset/Data/MSRA.mat'   #
# yFile = 'D:/HDM-master/data/UTD/feature_locs.mat'
data_path = '../data/UTD_MHAD_raw'
out_path = '../data/UTD_MHAD'
action_names = ['_', 'right arm swipe to the left',  'right arm swipe to the right', 'right hand wave', 'two hand front clap',
                'right arm throw', 'cross arms in the chest', 'basketball shoot', 'right hand draw x', 'right hand draw circle (clockwise)',
                'right hand draw circle (counter clockwise)', 'draw triangle', 'bowling (right hand)', 'front boxing', 'baseball swing from right',
                'tennis right hand forehand swing',  'arm curl (two arms)', 'tennis serve', 'two hand push', 'right hand knock on door',
                'right hand catch an object', 'right hand pick up and throw', 'jogging in place', 'walking in place', 'sit to stand',
                'stand to sit', 'forward lunge (left foot forward)', 'squat (two arms stretch out)']
N = len(os.listdir(data_path))
fp = np.zeros((N, 20, 3, 300, 2), dtype=np.float32)
sample_name = []
sample_label = []
a = []
# os.listdir().sort(key=lambda x:int(x.split('.')[0]))
for n, filename in enumerate(os.listdir(data_path)):
    # keypoint=pd.read_csv('../../UTD_MHAD/Skeleton/{}'.format(filename),sep='\t',header=None)
    keypoint = sio.loadmat('../data/UTD_MHAD_raw/{}'.format(filename))  # 20 3 T
    action_class = int(filename[filename.find('a')+1:filename.find('_')])
    frame_num = keypoint['d_skel'].shape[2]
    fp[n,:,:,:frame_num,0] = keypoint['d_skel']
    a.append(keypoint['d_skel'].shape[2])
    sample_name.append(action_names[action_class])
    sample_label.append(action_class - 1)
    # print(action_class)
    # print(keypoint['d_skel'].shape[2])
# print(a)
with open('{}/all_label.pkl'.format(out_path), 'wb') as f:
    pickle.dump((sample_name, list(sample_label)), f)
fp = np.transpose(fp,(0,2,3,1,4))   # N C T V M
fp = pre_normalization(fp)
print(np.mean(a))
print(fp.shape)
# np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)
np.save('{}/all_data_joint.npy'.format(out_path), fp)


# print(MSRA)
# print(MSRA['data'],MSRA['data'].shape)


