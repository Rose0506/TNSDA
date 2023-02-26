# import torch.nn as nn
# import torch
# from dropblock import DropBlock2D, LinearScheduler
# import numpy as np
# # from packages.fastdvdnet.utils import disassemble
# # from packages.fastdvdnet.utils import disassemble
#
#
# class FE(nn.Module):
#     def __init__(self,
#                  in_channels, out_channels1, out_channels2, out_channels3, out_channels4,
#                  ksize=3, stride=1, pad=1):
#
#         super(FE, self).__init__()
#         # self.body1 = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels1, 1, 1, 0),
#         #     nn.BatchNorm2d(out_channels1),
#         #     nn.ReLU(inplace=True))
#         #
#         # self.body2 = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels2, ksize, stride, pad),
#         #     nn.BatchNorm2d(out_channels2),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(out_channels2, out_channels2, ksize, stride, pad),
#         #     nn.BatchNorm2d(out_channels2),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(out_channels2, out_channels2, ksize, stride, pad),
#         #     nn.BatchNorm2d(out_channels2),
#         #     nn.ReLU(inplace=True))
#         #
#         # self.body3 = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels3, ksize, stride, pad),
#         #     nn.BatchNorm2d(out_channels3),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(out_channels3, out_channels3, ksize, stride, pad),
#         #     nn.BatchNorm2d(out_channels3),
#         #     nn.ReLU(inplace=True)
#         # )
#         #
#         # self.body4 = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels4, ksize, stride, pad),
#         #     nn.BatchNorm2d(out_channels4),
#         #     nn.ReLU(inplace=True),
#         # )
#
#         self.body1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels1, \
#                       kernel_size=1, stride=1, padding=0, groups=3, bias=False),
#             nn.BatchNorm2d(out_channels1),
#             nn.ReLU(inplace=True))
#
#         self.body2 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels1, \
#                       kernel_size=3, stride=1, padding=1, groups=3, bias=False),
#             nn.BatchNorm2d(out_channels1),
#             nn.ReLU(inplace=True))
#
#         self.body3 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels1, \
#                       kernel_size=5, stride=1, padding=2, groups=3, bias=False),
#             nn.BatchNorm2d(out_channels1),
#             nn.ReLU(inplace=True))
#
#         self.body4 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels1, \
#                       kernel_size=7, stride=1, padding=3, groups=3, bias=False),
#             nn.BatchNorm2d(out_channels1),
#             nn.ReLU(inplace=True))
#
#     def forward(self, x):
#         out1 = self.body1(x)
#
#         out2 = self.body2(x)
#
#         out3 = self.body3(x)
#
#         out4 = self.body4(x)
#
#         out = torch.cat([out1, out2, out3, out4], dim=1)
#
#         return out
#
#
# class OneBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(OneBlock, self).__init__()
#         self.forw = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
#             # nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
#             nn.BatchNorm2d(out_channels),
#             nn.ELU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.forw(x)
#         return x
#
#
# class Block(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(Block, self).__init__()
#         self.forw = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-04),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.forw(x)
#         return x
#
#
# class EBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(EBlock, self).__init__()
#         self.forw = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#             # nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-04),
#             nn.BatchNorm2d(out_channels),
#             nn.ELU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.forw(x)
#         return x
#
#
# class Down_Block(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(Down_Block, self).__init__()
#         self.forw = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
#             # nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.forw(x)
#         return x
#
#
# class Up_Block(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(Up_Block, self).__init__()
#         self.forw = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0,bias=True),
#             # nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-04),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.forw(x)
#         return x
#
#
# class Encoder(nn.Module):  # EB
#     def __init__(self, in_channel, out_channel):
#         super(Encoder, self).__init__()
#         self.en1 = EBlock(in_channel, in_channel)
#         self.en2 = EBlock(in_channel, in_channel)
#         self.en3 = Down_Block(in_channel, out_channel)
#
#     def forward(self, x):
#         e1 = self.en1(x)
#         e2 = self.en2(e1)
#         e3 = self.en3(e2)
#
#         return e3
#
#
# class Decoder(nn.Module):  # EB
#     def __init__(self, in_channel, out_channel):
#         super(Decoder, self).__init__()
#         self.en1 = Up_Block(in_channel, in_channel)
#         self.en2 = EBlock(in_channel, in_channel)
#         self.en3 = EBlock(in_channel, out_channel)
#
#     def forward(self, x):
#         e1 = self.en1(x)
#         e2 = self.en2(e1)
#         e3 = self.en3(e2)
#
#         return e3
#
#
# class Mid(nn.Module):
#     def __init__(self, num, in_ch, out_ch):
#         super(Mid, self).__init__()
#         self.block = self._make_layer(Block, num, in_ch, out_ch)
#
#     def _make_layer(self, block, block_num, in_channels, out_channels, **kwargs):
#         layers = []
#
#         for _ in range(1, block_num+1):
#             layers.append(block(in_channels, out_channels, **kwargs))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.block(x)
#
#         return out
#
#
# class net(nn.Module):
#     def __init__(self):
#         super(net, self).__init__()
#         self.fe = FE(21, 32, 32, 32, 32)  # 13 for gray, 39 for color
#         # self.dropblock = LinearScheduler(
#         #     DropBlock2D(drop_prob=0., block_size=5),
#         #     start_value=0.,
#         #     stop_value=0.,
#         #     nr_steps=5e3
#         # )
#         self.dropblock = DropBlock2D(block_size=7, drop_prob=0.6)
#         self.one = OneBlock(128, 64)
#         self.en1 = Encoder(64, 64)
#         self.en2 = Encoder(64, 128)   # downsample
#         self.en3 = Encoder(128, 256)   # downsample
#         self.en4 = Encoder(256, 512)    # downsample
#         self.mid = Block(512, 1024)  # 512-1024
#         self.mid1 = Block(1024, 512)
#         self.de4 = Decoder(512, 256)
#         self.de3 = Decoder(256, 128)
#         self.de2 = Decoder(128, 64)
#         self.de1 = Decoder(64, 64)  # 64
#         self.sp = nn.Conv2d(64, 1, 3, 1, 1)
#
#         self.head1 = Block(128, 64)
#         self.en11 = Encoder(64, 64)   # downsample
#         self.en22 = Encoder(64, 128)   # downsample
#         self.en33 = Encoder(128, 256)   # downsample
#         self.en44 = Encoder(256, 512)    # downsample
#         self.mid00 = Block(512, 1024)
#
#         self.mid11 = Block(1024, 512)
#         self.de44 = Decoder(512, 256)
#         self.de33 = Decoder(256, 128)
#         self.de22 = Decoder(128, 64)
#         self.de11 = Decoder(64, 64)
#         self.tail = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)  # 1 for gray
#
#         self.weight = nn.Conv2d(1,1,1,1,0,bias=False)
#         self.weight1 = nn.Conv2d(1,1,1,1,0,bias=False)
#
#         self.m1 = Mid(4, 64, 64)
#         self.m2 = Mid(3, 128, 128)
#         self.m3 = Mid(2, 256, 256)
#         self.m4 = Mid(1, 512, 512)
#
#     def forward(self, x):
#         # self.dropblock.step()
#         # print(x.shape)
#         fe = self.dropblock(self.fe(x))  # 21-128   96*96-96*96
#         head = self.dropblock(self.one(fe)) # 128-64   96*96-96*96
#         en1 = self.en1(head)   # 64-64   96*96-48*48
#         en2 = self.en2(en1)    # 64-128   48*48-24*24
#         # en3 = self.en3(en2)    # 128-256   24*24-12*12
#         # en4 = self.en4(en3)  # 256-512   12*12-6*6
#         # mid = self.mid(en4)  # 512-1024   6*6
#         # mid1 = self.mid1(mid) # 1024-512  6*6
#
#         # de4 = self.de4(mid1+en4)  # 512-256  6*6-12*12
#         # print(de4.shape,en3.shape) #
#         # de3 = self.de3(de4+en3)   # 256-128  12*12-24*24
#         # de2 = self.de2(de3+en2)  # 128-64  24*24-48*48
#         de2 = self.de2(en2)  # 128-64  24*24-48*48
#         de1 = self.de1(de2+en1)  # 64-64   48*48-96*96
#         sp = self.sp(de1)   #  64-1    96*96
#         mask = torch.sigmoid(10 * self.weight(sp.detach())) - self.weight1(torch.sigmoid(sp.detach()+10))
#
#         head1 = self.dropblock(self.head1(fe))  # 128-64  96*96
#         en11 = self.en11(head1+mask)  # 64-64   96*96-48*48
#         en22 = self.en22(en11)    # 64-128    48*48-24*24
#         # en33 = self.en33(en22)    # 128-256    24*24-12*12
#         # en44 = self.en44(en33)    # 256-512    12*12-6*6
#
#         # mid00 = self.mid00(en44)  # 512-1024   6*6
#         # mid11 = self.mid11(mid00) # 1024-512   6*6
#         # de44 = self.de44(mid11+self.m4(en44))  # 512-256   6*6-12*12
#         # de33 = self.de33(de44+self.m3(en33))   # 256-128   12*12-24*24
#         # de22 = self.de22(de33+self.m2(en22))   # 128-64    24*24-48*48
#         # de11 = self.de11(de22+self.m1(en11))   # 64-64     48*48-96*96
#
#         de22 = self.de22(en22)   # 128-64    24*24-48*48
#         de11 = self.de11(de22+en11)   # 64-64     48*48-96*96
#
#         res = self.tail(de11) + x[:,3:6,:,:]   # 64-3      96*96
#         # + x[:,0:3,:,:] for color
#         return sp, res
#
#
#
# # class FastDVDnet(nn.Module):
# # 	""" Definition of the FastDVDnet model.
# # 	Inputs of forward():
# # 		xn: input frames of dim [N, C, H, W], (C=3 RGB)
# # 		gradi_map: array with noise map of dim [N, 12, H, W] if l=4
# # 	"""
# #
# # 	def __init__(self, num_input_frames=5):
# # 		super(FastDVDnet, self).__init__()
# # 		self.num_input_frames = num_input_frames
# # 		# Define models of each denoising stage
# # 		self.temp1 = net()
# # 		self.temp2 = net()
# # 		# Init weights
# # 		self.reset_params()
# #
# # 	@staticmethod
# # 	def weight_init(m):
# # 		if isinstance(m, nn.Conv2d):
# # 			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
# #
# # 	def reset_params(self):
# # 		for _, m in enumerate(self.modules()):
# # 			self.weight_init(m)
# #
# # 	def forward(self, x):
# # 		'''Args:
# # 			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
# # 			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
# # 		'''
# # 		# Unpack inputs
# #
# # 		(x0, x1, x2, x3, x4) = tuple(x[:, 3 * m:3 * m + 3, :, :] for m in range(self.num_input_frames))
# # 		# x0, x1, x2, x3, x4 = x[:, 3 * m:3 * m + 3, :, :] for m in range(self.num_input_frames)
# #
# #         return x
#
# class FastDVDnet(nn.Module):
#     """ Definition of the FastDVDnet model.
#     Inputs of forward():
#         xn: input frames of dim [N, C, H, W], (C=3 RGB)
#         noise_map: array with noise map of dim [N, 1, H, W]
#     """
#
#     def __init__(self, num_input_frames=5):
#         super(FastDVDnet, self).__init__()
#         self.num_input_frames = num_input_frames
#         # Define models of each denoising stage
#         self.temp1 = net()
#         self.temp2 = net()
#         # Init weights
#         self.reset_params()
#
#     @staticmethod
#     def weight_init(m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#
#     def reset_params(self):
#         for _, m in enumerate(self.modules()):
#             self.weight_init(m)
#
#     def forward(self, x, g):
#         '''Args:
#             x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
#             noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
#         '''
#         # Unpack inputs
#         (x0, x1, x2, x3, x4) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))
#         # a = [x1,x2,x3]
#         # g = []
#         # for i in a:
#         #     N,C,H,W = x.shape
#         #     gradient = np.zeros((N,12,H,W))
#         #     i = i.cpu().numpy()
#         #     for n in range(N):
#         #         batch = np.transpose(i[n],(1,2,0))  # h w c
#         #         batch = np.transpose(disassemble(batch,3,4),(2,0,1))
#         #         gradient[n,:,:,:] = batch
#         #     gradient = torch.from_numpy(gradient).float().to(x.device)
#         #     g.append(gradient)
#         # First stage
#         _, x20 = self.temp1(torch.concat((x0,x1,x2,g[0]),dim=1))
#         _, x21 = self.temp1(torch.concat((x1,x2,x3,g[1]),dim=1))
#         _, x22 = self.temp1(torch.concat((x2,x3,x4,g[2]),dim=1))
#
#         #Second stage
#         _, x = self.temp2(torch.concat((x20,x21,x22,g[1]),dim=1))
#
#         return x
#
#
# # model = InputCvBlock(3,32)
# # video = torch.randn(1,12,16,16)    # 3*(3+1)
# # out = model(video)
# # print(out.shape)
# #
# # model = DenBlock()    # U-net模块
# # # model = DenBlock_CReLU()    # U-net模块
# # imgs = torch.rand(1,3,16,16)   # 1帧
# # noise = torch.rand(1,1,16,16)
# # out = model(imgs,imgs,imgs,noise)
# # print(out.shape)
# # # # #
# # model = DownBlock(3,6)
# # imgs = torch.rand(1,3,8,8)
# # out = model(imgs)
# # print(out.shape)
# ###
# model1 = FastDVDnet()     # 完整模型
# input = torch.rand(1,15,480,912)   # 5帧
# g = []
# for i in range(3):
#     g.append(torch.rand(1,12,480,912))
# # gradi = torch.rand(1,12,96,96)
# output = model1(input,g)
# print(output.shape)
# #
# # inp = torch.randn((1, 21, 96, 96))
# # model2 = net()
# # sp,out = model2(inp)
# # print(sp.shape,out.shape)
# # if __name__ == '__main__':
# # 	inp = torch.randn((1, 21, 128, 128))
# # 	net = net()
# # 	sp,out = net(inp)
# # 	print(sp.shape,out.shape)


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


