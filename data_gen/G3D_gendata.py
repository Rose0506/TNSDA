import scipy.io as sio
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from preprocess import pre_normalization

def pre_normalization0(data):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break
    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data

sample_name = []
sample_label = []
# keypoint = sio.loadmat('D:/HDM-master/data/G3D/feature_locs.mat')
keypoint = sio.loadmat('../data/skeleton_data/MSRP.mat')
N = keypoint['data'].size
# for i in range(N):
#     print(keypoint['data'][i][0].shape[2])
fp = np.zeros((N, 3, 20, 400, 2), dtype=np.float32)    #   N, 20, 3, 300, 2
out_path = '../data/G3D'
cal = []
for i in range(N):
    frames_num = keypoint['data'][i][0].shape[2]
    print(frames_num)
    cal.append(frames_num)
    fp[i,:,1:,:frames_num,0] = keypoint['data'][i][0].astype(np.float32)
    action_class = int(keypoint['action_label'][i])
    fp.transpose([0,2,1,3,4])    #
    sample_name.append(action_class)
    sample_label.append(action_class - 1)
with open('{}/all_label.pkl'.format(out_path), 'wb') as f:
    pickle.dump((sample_name, list(sample_label)), f)
fp = np.transpose(fp,(0,2,3,1,4))   # N C T V M
fp = pre_normalization0(fp)
print(np.mean(cal))
# print(fp.shape)
# np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)
np.save('{}/all_data_joint.npy'.format(out_path), fp)

# print('a')


# data_path = '../data/UTD_MHAD_raw'
# out_path = '../data/UTD_MHAD'
# action_names = ['_', 'right arm swipe to the left',  'right arm swipe to the right', 'right hand wave', 'two hand front clap',
#                 'right arm throw', 'cross arms in the chest', 'basketball shoot', 'right hand draw x', 'right hand draw circle (clockwise)',
#                 'right hand draw circle (counter clockwise)', 'draw triangle', 'bowling (right hand)', 'front boxing', 'baseball swing from right',
#                 'tennis right hand forehand swing',  'arm curl (two arms)', 'tennis serve', 'two hand push', 'right hand knock on door',
#                 'right hand catch an object', 'right hand pick up and throw', 'jogging in place', 'walking in place', 'sit to stand',
#                 'stand to sit', 'forward lunge (left foot forward)', 'squat (two arms stretch out)']
# N = len(os.listdir(data_path))
# fp = np.zeros((N, 20, 3, 300, 2), dtype=np.float32)
# sample_name = []
# sample_label = []
# a = []
# # os.listdir().sort(key=lambda x:int(x.split('.')[0]))
# for n, filename in enumerate(os.listdir(data_path)):
#     # keypoint=pd.read_csv('../../UTD_MHAD/Skeleton/{}'.format(filename),sep='\t',header=None)
#     keypoint = sio.loadmat('../data/UTD_MHAD_raw/{}'.format(filename))  # 20 3 T
#     action_class = int(filename[filename.find('a')+1:filename.find('_')])
#     frame_num = keypoint['d_skel'].shape[2]
#     fp[n,:,:,:frame_num,0] = keypoint['d_skel']
#     a.append(keypoint['d_skel'].shape[2])
#     sample_name.append(action_names[action_class])
#     sample_label.append(action_class - 1)
#     # print(action_class)
#     # print(keypoint['d_skel'].shape[2])
# # print(a)
# with open('{}/all_label.pkl'.format(out_path), 'wb') as f:
#     pickle.dump((sample_name, list(sample_label)), f)
# fp = np.transpose(fp,(0,2,3,1,4))   # N C T V M
# fp = pre_normalization(fp)
# print(fp.shape)
# # np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)
# np.save('{}/all_data_joint.npy'.format(out_path), fp)