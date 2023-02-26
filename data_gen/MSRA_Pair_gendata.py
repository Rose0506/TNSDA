import pandas as pd     #引入pandas包
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from preprocess import pre_normalization

n = 0
# data_path = 'D:/MSRAction3D/MSRAction3D_download/MSRAction3DSkeletonReal3D'
data_path = '../data/MSRAP_raw'    #400
# data_path = '../data/MSRDaily_Activity3D_raw'  #700
out_path = '../data/MSRAP'
# out_path = '../data/MSRDaily_Activity3D'
N = len(os.listdir(data_path))
fp = np.zeros((N, 400, 20, 3, 2), dtype=np.float32)
sample_name = []
sample_label = []
cal = []
for n,filename in enumerate(os.listdir(data_path)):
    # print(filename)
    file_path = os.path.join(data_path,filename)
    with open(file_path, 'r') as f:
        frames_num = int(f.readline().split()[0])
        # print(frames_num)
        # print(filename)
        # if frames_num > 300:
        #     frames_num = 300
        #     print(filename)
        #     print(frames_num)
        cal.append(frames_num)
        for f_id in range(frames_num):
            g = f.readline()
            # if int(g[0]) == 0:
            #     print(filename)
            #     continue
            for j in range(20):
                # f.readline()
                jointInfo = f.readline()
                # print(n,f_id,j)
                # fp[n,f_id,j,:,0] = float(jointInfo.split()[0]),float(jointInfo.split()[1]), \
                #                    float(jointInfo.split()[2])
                # print(filename)
                fp[n, f_id, j, 0, 0] = float(jointInfo.split()[0])
                fp[n, f_id, j, 1, 0] = float(jointInfo.split()[1])
                fp[n, f_id, j, 2, 0] = float(jointInfo.split()[2])

                f.readline()
    action_class = int(filename[filename.find('a') + 1:filename.find('a') + 3])
    sample_name.append(filename)
    sample_label.append(action_class - 1)

with open('{}/all_label.pkl'.format(out_path), 'wb') as f:
    pickle.dump((sample_name, list(sample_label)), f)
fp = np.transpose(fp,(0,3,1,2,4))
fp = pre_normalization(fp)
print(fp.shape)
print(np.mean(cal))
# np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)
np.save('{}/all_data_joint.npy'.format(out_path), fp)
