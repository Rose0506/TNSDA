import pandas as pd     #引入pandas包
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from preprocess import pre_normalization

n = 0
# data_path = 'D:/MSRAction3D/MSRAction3D_download/MSRAction3DSkeletonReal3D'
data_path = '../data/MSR_Action3D_raw'
out_path = '../data/MSR_Action3D'
N = len(os.listdir(data_path))
fp = np.zeros((N, 300, 20, 3, 2), dtype=np.float32)
sample_name = []
sample_label = []
cal = []
for filename in os.listdir(data_path):
    keypoint=pd.read_csv('../data/MSR_Action3D_raw/{}'.format(filename),sep='\t',header=None)     #读入txt文件，分隔符为\t
    action_class = int(filename[filename.find('a')+1:filename.find('a')+3])
    # keypoint=pd.read_csv('../data/MSR Action3D_test/a01_s01_e01_skeleton3D.txt',sep='\t',header=None)     #读入txt文件，分隔符为\t
    keypoint.columns=['x']
    keypoint['y']=None
    keypoint['z']=None
    keypoint['c']=None
    for i in range(len(keypoint)):         #遍历每一行
        coordinate = keypoint['x'][i].split() #分开第i行，u列的数据。split()默认是以空格等符号来分割，返回一个列表
        keypoint['x'][i]=float(coordinate[0])       #分割形成的列表第一个数据给u列
        keypoint['y'][i]=float(coordinate[1])        #分割形成的列表第二个数据给v列
        keypoint['z'][i]=float(coordinate[2])        #分割形成的列表第一个数据给d列
        keypoint['c'][i]=float(coordinate[3])        #分割形成的列表第二个数据给c列

    frames_num = int(len(keypoint)/20)
    cal.append(frames_num)
    print(frames_num)
    keypoint = np.array(keypoint,dtype=np.float32).reshape(frames_num,20,4)

    sample_name.append(filename)
    sample_label.append(action_class - 1)
    for f in range(frames_num):
        for v in range(20):
            fp[n,f,v,:,0] = keypoint[f,v,0:3]
    n += 1

with open('{}/all_label.pkl'.format(out_path), 'wb') as f:
    pickle.dump((sample_name, list(sample_label)), f)
fp = np.transpose(fp,(0,3,1,2,4))
fp = pre_normalization(fp)
print(fp.shape)
print(np.mean(cal))
# np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)
np.save('{}/all_data_joint.npy'.format(out_path), fp)

