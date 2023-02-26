import random
import pickle
import pandas as pd
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from gtda.time_series import SlidingWindow
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import TakensEmbedding
from gtda.diagrams import BettiCurve,PersistenceLandscape,PersistenceImage,Silhouette
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
# import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import seaborn as sns
import nolds
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import time
from NLFE import nlfe
import time
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from labels import *
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("Experiment BettiCurve")
# print("Experiment PersistenceLandscape")
# print("Experiment Silhouette")

seed = int(3407)
# data_num = 314    # MSRDaily_Activity3D    192    700
# data_num = 663      # G3D                   46    400

# data_num = 352       # MSRAction-Pair      119    400
data_num = 567       # MSR_Action3D           40    300
# data_num = 861       # UTD_MHAD             67    300
# data_num = 56578     # NTU RGB+D
# data_num = 113945    # NTU RGB+D 120
# data_num = 260232    # Kinetics-400
SlidingWindow_size = 100
SlidingWindow_stride = 100
random.seed(seed)
# dataset = 'G3D'
# dataset = 'MSRDaily_Activity3D'
# dataset = 'MSRAP'
# dataset = 'UTD_MHAD'
dataset = 'MSR_Action3D'
# dataset = 'kinetics_without_mask'
# dataset = 'ntu/xsub'
# dataset = 'ntu/xview_with_1st_frame_center'
# dataset = 'ntu120/xsub'
set = 'all'
label_dict ={'MSR_Action3D':label_MSR_Action3D, 'MSRAP':label_MSRAP, 'MSRDaily_Activity3D':label_MSRDaily_Activity3D,\
             'UTD_MHAD':label_UTD_MHAD, 'G3D':label_G3D}
modal = ['joint', 'bone', 'joint_motion', 'bone_motion',    #  0 1 2 3
         'joint_merge_bone', 'joint_merge_joint_motion', 'joint_merge_bone_motion',  # 4 5 6
         '2order_joint', 'joint_motion_acceleration', 'bone_motion_acceleration',  # 7 8 9
         'bone_merge_2order_joint',  'jpt_merge_2order_joint',  # 10 11
         'joint_fusion_bone','joint_fusion_joint_motion','joint_fusion_bone_motion','2order_joint_fusion_bone']  # 12 13 14 15
# modal = 'bone'
data1 = np.load('./data/{}/{}_data_{}.npy'.format(dataset, set, modal[0]))
# data1 = data1[:,0:2,:,:,:]
print('./data/{}/{}_data_{}.npy'.format(dataset, set, modal[0]))
print('time_delay={}, dimension={}, SlidingWindow_size={}, SlidingWindow_stride={}'
      .format(5,5,SlidingWindow_size,SlidingWindow_stride))
print('metric="euclidean", homology_dimensions=[0]')
N, C, T, V, M = data1.shape
print(data1.shape)
# 1-base of the spine 2-middle of the spine
# 3-neck 4-head 5-left shoulder 6-left elbow 7-left wrist 8-
# left hand 9-right shoulder 10-right elbow 11-right wrist 12-
# right hand 13-left hip 14-left knee 15-left ankle 16-left foot 17-
# right hip 18-right knee 19-right ankle 20-right foot 21-spine 22-
# tip of the left hand 23-left thumb 24-tip of the right hand 25-
# right thumb
colum = ['base of the spine_X', 'base of the spineY', 'base of the spine_Z', 'middle of the spine_X', 'middle of the spine_Y', 'middle of the spine_Z', 'neck_X', 'neck_Y','neck_Z',
         'head_X', 'head_Y', 'head_Z', 'left shoulder_X', 'left shoulder_Y', 'left shoulder_Z', 'left elbow_X', 'left elbow_Y','left elbow_Z',
         'left wrist_X', 'left wrist_Y', 'left wrist_Z', 'left hand_X', 'left hand_Y', 'left hand_Z', 'right shoulder_X', 'right shoulder_Y','right shoulder_Z',
         'right elbow_X', 'right elbow_Y', 'right elbow_Z', 'right wrist_X', 'right wrist_Y', 'right wrist_Z', 'right hand_X', 'right hand_Y','right hand_Z',
         'left hip_X', 'left hip_Y', 'left hip_Z', 'left knee_X', 'left knee_Y', 'left knee_Z', 'left ankle_X', 'left ankle_Y','left ankle_Z',
         'left foot_X', 'left foot_Y', 'left foot_Z', 'right hip_X', 'right hip_Y', 'right hip_Z', 'right knee_X', 'right knee_Y','right knee_Z',
         'right ankle_X', 'right ankle_Y', 'right ankle_Z', 'right foot_X', 'right foot_Y', 'right foot_Z', 'spine_X', 'spine_Y','spine_Z',
         'tip of the left hand', 'tip of the left hand_Y', 'tip of the left hand_Z', 'left thumb_X', 'left thumb_Y', 'left thumb_Z', 'tip of the right hand_X', 'tip of the right hand_Y','tip of the right hand_Z',
         'right thumb_X', 'right thumb_Y', 'right thumb_Z']
data = data1[:,:,:,:,0].transpose(0,2,3,1).reshape(N*T,V*C)
DF_data = pd.DataFrame(data[:(data_num)*T],columns=None)   # 113945  56578
# DF_data.to_csv('./tnda_NTU/ntu_processed_data.csv',index=None)
l = open('./data/{}/{}_label.pkl'.format(dataset, set),'rb')
label_pkl = pickle.load(l)
label = label_pkl[1]
# DF_label = pd.DataFrame(label)
# Series_label = DF_label.iloc[:,0]
# print(Series_label)
# fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)
label_total = []
for i in label:
    for j in range(T):
        label_total.append(i)
DF_labels = pd.DataFrame(label_total,columns=['class'])
Series_labels = DF_labels.iloc[:,0]
DF_label = pd.DataFrame(Series_labels[:(data_num)*T])    #  113945   56578
# DF_label.to_csv('./tnda_NTU/ntu_label.csv',index=None)
x = DF_data
y = DF_label.iloc[:,0]
# data_label = pd.concat([DF_data,DF_label],axis=1)
# data_label.to_csv('./tnda_NTU/ntu_all.csv',index=None)
# labels = pd.concat(label_total, axis=1)
# print(DF_data.shape)
# print(Series_labels.shape)
# print(data_label.shape)

# def data_preprocesssing(data, label, size, stride):
#     Scaler = MinMaxScaler()
#     data_ = Scaler.fit_transform(data)
#     SW = SlidingWindow(size=size, stride=stride)
#     X, y = SW.fit_transform_resample(data_, label)
#     return X, y

def data_preprocesssing(data, label, size, stride):
    Scaler = MinMaxScaler()
    data_ = Scaler.fit_transform(data)
    SW = SlidingWindow(size=size, stride=stride)
    S = []
    L = []
    # T, F = data_.shape
    # X = np.zeros((T, F), dtype=np.float32)
    for i in range(data_num):
        s = data_[i*700:(i+1)*700,:]
        l = label[i*700:(i+1)*700]
        x, y = SW.fit_transform_resample(s, l)
        # x = np.concatenate([s[:(size),:].reshape(1,size,-1),x], axis=0)
        # y = np.concatenate([y, np.array([y[0]])], axis=0)
        S.append(x)
        L.append(y)
    X = np.concatenate(S, axis=0)
    Y = np.concatenate(L, axis=0)
    return X, Y


def topo_feature(data, time_delay, dimension):
    featuress = []

    for i in tqdm(range(data.shape[2])):
        data_ = data[:, :, i]
        TE = TakensEmbedding(time_delay=time_delay, dimension=dimension)
        Taken = TE.fit_transform(data_)
        VR = VietorisRipsPersistence(
            metric="euclidean",
            # metric="cosine",
            # metric="manhattan",
            # homology_dimensions=[0, 1, 2],
            # homology_dimensions=[0, 1],
            homology_dimensions=[0],
            # homology_dimensions=[1],
            # homology_dimensions=[2],
            # homology_dimensions=[0,2],
            # 由6改为8
            n_jobs=-1,
            collapse_edges=True)
        VRs = VR.fit_transform(Taken)
        BE = BettiCurve()
        # BE = PersistenceLandscape()
        # BE = Silhouette()
        feature = BE.fit_transform(VRs)
        feature = feature.mean(axis=1)
        # feature = np.around(0.9 * feature[:, 0, :] + 0.1 * feature[:, 1, :])
        featuress.append(feature)
        # time.sleep(0.1)
    featuress = np.concatenate(featuress, axis=1)
    return featuress

def model_RFC(feature, label, n_splits, data_name):
    X, X_valid, y, y_valid = train_test_split(feature, label, test_size=0.2, random_state=42)
    RFC = RandomForestClassifier(n_estimators=46, max_features=0.2, bootstrap=True, n_jobs=-1, random_state=3407)
    RFC.fit(X, y)
    acc_score = RFC.score(X_valid, y_valid)
    print(acc_score)
    y_pred = RFC.predict(X_valid)
    print(classification_report(y_valid, y_pred, digits=4))
    #
    # LABELS = ['1', '2', '3', '4', '5', '6', '7', '8']
    LABELS = label_dict[data_name]
    confusion_matrix = metrics.confusion_matrix(y_valid, y_pred)  ###混淆矩阵TPTF
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    # sns.set(font_scale=1.5)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=".2f");
    plt.title("CONFUSION MATRIX_RFC : ")
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    plt.savefig('./tnda_NTU/'+data_name+'_cmatrix.png')
    plt.show()
    #
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=None)
    score = cross_val_score(RFC, feature, label, cv=cv)
    print(score)
    print(score.mean())

data_processed, label_processed = data_preprocesssing(x, y, SlidingWindow_size, SlidingWindow_stride)
print(data_processed.shape)
print(label_processed.shape)


in_data_topo = topo_feature(data_processed, 5, 5)
print('in_data_topo shape:', in_data_topo.shape)


# #%%
#
# feture = pd.DataFrame(in_data_topo)
# feture.to_csv('./tnda_NTU/topo_feature.csv',index=None)

model_RFC(in_data_topo, label_processed, 5, dataset)
