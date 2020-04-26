from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1: 4])
        fan_out = np.prod(weight_shape[2: 4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)
    elif classname.find('LSTMCell') != -1:
        m.bias_ih.data.fill_(0.0)
        m.bias_hh.data.fill_(0.0)


class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x --> 3 * 3
        batchsize = x.shape[0]
        if batchsize > 1:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.mp1(x)
            x = x.view(-1, 1024)

            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.mp1(x)
            x = x.view(-1, 1024)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

        x = self.fc3(x)

        iden = Variable(torch.eye(3)).view(1, -1).repeat(batchsize, 1)
        if x.is_cuda:
            device = torch.device('cuda:%d' % x.get_device())
            iden = iden.to(device=device)
        x = x + iden
        x = x.view(-1, 3, 3)

        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        trans = self.stn(x)
        x = torch.cat([torch.bmm(trans, x[:, :3, :]), x[:, 3, :].unsqueeze(1)], dim=1)

        if x.shape[0] > 1:
            x = F.relu(self.bn1(self.conv1(x)))
            pointfeat = x
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu(self.conv1(x))
            pointfeat = x
            x = F.relu(self.conv2(x))
            x = self.conv3(x)

        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans


class end_layer(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1):
        super(end_layer, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_channels)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.apply(weights_init)

    def forward(self, x):
        if x.size()[0] == 1:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class PointNetActorCritic(nn.Module):
    def __init__(self, num_points=2500, num_actions=4):
        super(PointNetActorCritic, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)

        self.lstm = nn.LSTMCell(1024, 1024)

        self.critic_linear = end_layer(in_channels=1024, out_channels=1)
        self.actor_linear = end_layer(in_channels=1024, out_channels=num_actions)

        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x, _ = self.feat(x)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


if __name__ == '__main__':
    sim_data = Variable(torch.rand(10, 4, 2500))

    # trans = STN3d()
    # out = trans(sim_data)
    # print('stn', out.size())

    # pointfeat = PointNetfeat(global_feat=True)
    # out, _ = pointfeat(sim_data)
    # print('global feat', out.size())

    # pointfeat = PointNetfeat(global_feat=False)
    # out, _ = pointfeat(sim_data)
    # print('point feat', out.size())

    cls = PointNetActorCritic(num_actions=4)
    hx, cx = Variable(torch.zeros(10, 1024)), Variable(torch.zeros(10, 1024))
    if torch.cuda.is_available():
        sim_data = sim_data.cuda()
        cls = cls.cuda()
        hx, cx = hx.cuda(), cx.cuda()
    v, q, (hx ,cx) = cls((sim_data, (hx, cx)))
    print(v.shape, q.shape, hx.shape, cx.shape)
    print(v)
    print(q)
