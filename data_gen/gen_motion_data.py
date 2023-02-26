import os
import numpy as np
from numpy.lib.format import open_memmap

sets = {
    # 'train', 'val', 'all'
    'all'
}

datasets = {
    # 'ntu/xview', 'ntu/xsub', 'ntu120/xsetup', 'ntu120/xsub'
    # 'ntu/xview', 'ntu/xsub'
    # 'ntu120/xsetup', 'ntu120/xsub'
    'ntu120/xsub'
}

# parts = {
#     'joint', 'bone'
# }

parts = {
    'joint_motion', 'bone_motion'
}

from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        for part in parts:
            print(dataset, set, part)
            data = np.load('../data/{}/{}_data_{}.npy'.format(dataset, set, part))
            N, C, T, V, M = data.shape    # N:文件数量 C:坐标数 T:帧数 V:节点数 M:身体数
            fp_sp = open_memmap(
                # '../data/{}/{}_data_{}_motion.npy'.format(dataset, set, part),
                '../data/{}/{}_data_{}_acceleration.npy'.format(dataset, set, part),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, V, M))
            for t in tqdm(range(T - 1)):
                fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
            fp_sp[:, :, T - 1, :, :] = 0
