import os
import numpy as np

sets = {
    # 'train', 'val', 'all'
    'all'
}

# 'ntu/xview', 'ntu/xsub', 'kinetics'
datasets = {
    # 'ntu/xview', 'ntu/xsub',
    # 'ntu120/xsetup', 'ntu120/xsub'
    'ntu120/xsub'
}

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))
        data_bone = np.load('../data/{}/{}_data_bone.npy'.format(dataset, set))
        data_jpt_motion = np.load('../data/{}/{}_data_joint_motion.npy'.format(dataset, set))
        data_bone_motion = np.load('../data/{}/{}_data_bone_motion.npy'.format(dataset, set))
        N, C, T, V, M = data_jpt.shape
        data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)
        data_fusion0 = data_jpt_bone.reshape(N, 2, C, T, V, M).mean(axis=1)
        np.save('../data/{}/{}_data_joint_fusion_bone.npy'.format(dataset, set), data_fusion0)
        np.save('../data/{}/{}_data_joint_merge_bone.npy'.format(dataset, set), data_jpt_bone)

        data_jpt_jptmotion = np.concatenate((data_jpt, data_jpt_motion), axis=1)
        data_fusion1 = data_jpt_jptmotion.reshape(N, 2, C, T, V, M).mean(axis=1)
        np.save('../data/{}/{}_data_joint_merge_joint_motion.npy'.format(dataset, set), data_jpt_jptmotion)
        np.save('../data/{}/{}_data_joint_fusion_joint_motion.npy'.format(dataset, set), data_fusion1)

        data_jpt_bonemotion = np.concatenate((data_jpt, data_bone_motion), axis=1)
        data_fusion2 = data_jpt_bonemotion.reshape(N, 2, C, T, V, M).mean(axis=1)
        np.save('../data/{}/{}_data_joint_merge_bone_motion.npy'.format(dataset, set), data_jpt_bonemotion)
        np.save('../data/{}/{}_data_joint_fusion_bone_motion.npy'.format(dataset, set), data_fusion2)

        data_bone_jptmotion = np.concatenate((data_bone, data_jpt_motion), axis=1)
        data_fusion3 = data_bone_jptmotion.reshape(N, 2, C, T, V, M).mean(axis=1)
        np.save('../data/{}/{}_data_bone_merge_jpt_motion.npy'.format(dataset, set), data_bone_jptmotion)
        np.save('../data/{}/{}_data_bone_fusion_jpt_motion.npy'.format(dataset, set), data_fusion3)

        data_bone_bonemotion = np.concatenate((data_bone, data_bone_motion), axis=1)
        data_fusion4 = data_bone_bonemotion.reshape(N, 2, C, T, V, M).mean(axis=1)
        np.save('../data/{}/{}_data_bone_merge_bone_motion.npy'.format(dataset, set), data_bone_bonemotion)
        np.save('../data/{}/{}_data_bone_fusion_bone_motion.npy'.format(dataset, set), data_fusion4)

        data_2order_joint = np.load('../data/{}/{}_data_2order_joint.npy'.format(dataset, set))
        data_bone_2order_joint = np.concatenate((data_bone, data_2order_joint), axis=1)
        data_fusion5 = data_bone_2order_joint.reshape(N, 2, C, T, V, M).mean(axis=1)
        np.save('../data/{}/{}_data_bone_merge_2order_joint.npy'.format(dataset, set), data_bone_2order_joint)
        np.save('../data/{}/{}_data_bone_fusion_2order_joint.npy'.format(dataset, set), data_fusion5)

        data_jpt_2order_joint = np.concatenate((data_jpt, data_2order_joint), axis=1)
        data_fusion6 = data_jpt_2order_joint.reshape(N, 2, C, T, V, M).mean(axis=1)
        np.save('../data/{}/{}_data_jpt_merge_2order_joint.npy'.format(dataset, set), data_jpt_2order_joint)
        np.save('../data/{}/{}_data_jpt_fusion_2order_joint.npy'.format(dataset, set), data_fusion6)

# for dataset in datasets:
#     for set in sets:
#         print(dataset, set)
#         data_bone = np.load('../data/{}/{}_data_bone.npy'.format(dataset, set))
#         data_2order_joint = np.load('../data/{}/{}_data_2order_joint.npy'.format(dataset, set))
#         N, C, T, V, M = data_bone.shape
#         data_merge = np.concatenate((data_2order_joint, data_bone), axis=1)
#         data_fusion = data_merge.reshape(N, 2, C, T, V, M).mean(axis=1)
#         # np.save('../data/{}/{}_data_2order_joint_merge_bone.npy'.format(dataset, set), data_merge)
#         np.save('../data/{}/{}_data_2order_joint_fusion_bone.npy'.format(dataset, set), data_fusion)

# for dataset in datasets:
#     for set in sets:
#         print(dataset, set)
#         data_jpt = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))
#         data_bone = np.load('../data/{}/{}_data_bone.npy'.format(dataset, set))
#         N, C, T, V, M = data_jpt.shape
#         data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)
#         np.save('../data/{}/{}_data_joint_bone.npy'.format(dataset, set), data_jpt_bone)