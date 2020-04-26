import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import argparse
import os
import pickle
import random
import sys
sys.path.append('/media/sdc/seg3d/T-Pointnet')
sys.path.append('/media/sdc/seg3d/T-Pointnet/a2c')

from pointnet import *
from environment import ActiveAgent
from utils import setup_logger


class CB(nn.Module):
    def __init__(self, num_points=3000, output=5):
        super(CB, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc = end_layer(in_channels=1024, out_channels=128)

        self.fc11 = nn.Linear(128, 64)
        self.fc12 = nn.Linear(64, 32)
        self.fc13 = nn.Linear(32, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, output)

        self.apply(weights_init)
        self.train()

    def forward(self, x):
        x, _ = self.feat(x)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNNCB(nn.Module):
    def __init__(self, output=5):
        super(CNNCB, self).__init__()
        self.feat = CNNfeat()
        self.fc = end_layer(in_channels=1024, out_channels=128)

        self.fc11 = nn.Linear(128, 64)
        self.fc12 = nn.Linear(64, 32)
        self.fc13 = nn.Linear(32, 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, output)

        self.apply(weights_init)
        self.train()

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def para_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='A2C')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--hidden-size', type=int, default=1024,
                        help='Hidden size for LSTM')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00,
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=50,
                        help='value loss coefficient (default: 50)')
    parser.add_argument('--seed', type=int, default=456,
                        help='random seed (default: 1)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 20)')
    parser.add_argument('--max-episode-length', type=int, default=20,
                        help='maximum length of an episode (default: 1000000)')
    parser.add_argument('--env-name', default='PointNetActorCritic',
                        help='environment to train on')
    parser.add_argument('--no-shared', default=False,
                        help='use an optimizer without shared momentum.')
    parser.add_argument('--n-points', type=int, default=3000,
                        help='the number of points feed to pointnet')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Folder to save logs')
    parser.add_argument('--model-dir', type=str, default='trained_models',
                        help='Folder to save models')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Folder to IORD')
    parser.add_argument('--resume', default=False,
                        help='resume latest model or not')
    parser.add_argument('--num-actions', type=int, default=5,
                        help='discrete action space')
    parser.add_argument('--num-test', type=int, default=10,
                        help='test time')
    parser.add_argument('--feat-archi', type=bool, default='cnn',
                        help='Feature extraction mode (pointnet or cnn)')

    # segmentation settings
    parser.add_argument("--depth-fusion", type=str, default='no-depth',
                        choices=['no-depth', 'pixel-concat', 'feature-concat'])
    parser.add_argument("--vote-mode", metavar="NAME",
                        type=str, choices=["plain", "mean", "voting", "max",
                        "mean+flip", "voting+flip", "max+flip"], default="mean")
    parser.add_argument("--vote-scales", type=list, default=[0.7, 1.2])
    parser.add_argument("--output-mode", metavar="NAME", type=str, choices=["palette", "raw", "prob"],
                        default="class",
                        help="How the output files are formatted."
                             " -- palette: color coded predictions"
                             " -- raw: gray-scale predictions"
                             " -- prob: gray-scale predictions plus probabilities")
    parser.add_argument("--snapshot", metavar="SNAPSHOT_FILE", type=str, default='wide_resnet38_deeplab_vistas.pth.tar', help="Snapshot file to load")
    parser.add_argument("--seg-model-dir", type=str, default="path of segmentation model")

    return parser


if __name__ == '__main__':
    parser = para_setting()
    seg_args = init_seg_parser()
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.feat_archi == 'pointnet':
        model = CB()
    elif args.feat_archi == 'cnn':
        model = CNNCB()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = model.cuda()
    criterion = criterion.cuda()

    env = ActiveAgent(idx=0, n_points=args.n_points, 
        seg_args=args, mode='sim', root_path=args.data_dir)
    env.seed(args.seed)

    # resume latest model
    if args.resume:
        model_path = os.path.join(args.model_dir, 'latest.pth')
        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)
        elif os.path.exists(model_path):
            print('Loading model from %s...' % model_path)
            model.load_state_dict(torch.load(model_path))

    # expert
    with open('expert_traj.pkl', 'rb') as f:
        expert_traj_all = pickle.load(f)

    itr = 0
    epoch = 0
    training_time = 50
    train_logger = setup_logger('trainer', os.path.join(args.log_dir, 'trainer_log.txt'))
    test_logger = setup_logger('test', os.path.join(args.log_dir, 'test_log.txt'))

    # test parameters
    all_success_time = 0
    all_time = 0
    ep_success_time = 0
    success_phase = 0.1
    check_flag = False

    while True:
        epoch += 1
        ################### training phase ###################
        model = model.train()
        for train_itr in range(training_time):
            expert_traj = random.choice(expert_traj_all)

            grasp_expert = random.random()
            if grasp_expert < 0.3:
                expert_s, expert_a = expert_traj[-1]
            else:
                expert_s, expert_a = random.choice(expert_traj)
            expert_a = expert_a.squeeze(1)

            # expert_s = np.array([x[0] for x in expert_traj]).squeeze(1)
            # expert_a = np.array([x[1] for x in expert_traj]).squeeze(1).squeeze(1)

            logit = model(torch.from_numpy(expert_s).cuda())
            itr += 1

            loss = criterion(logit, torch.from_numpy(expert_a).cuda())
            print('behaviour cloning loss: ', loss.data.cpu().numpy())
            train_logger.info('behaviour cloning loss: ' + str(loss.data.cpu().numpy()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ################### testing phase ###################
        model = model.eval()

        state, _ = env.reset()
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        if torch.cuda.is_available():
            state = state.cuda()
        reward_sum = 0
        done = True

        episode_length = 0
        testing = True
        while testing:
            episode_length += 1

            with torch.no_grad():
                logit = model(state)
            prob = F.softmax(logit, dim=1)
            action = prob.max(1, keepdim=True)[1].data.cpu().numpy()

            # path_info = '%s %s %s %d' % (env.target_group, env.scene_idx, env.coord, action[0, 0])
            # test_logger.info(path_info)

            state, reward, done = env.step(action[0, 0])
            reward_sum += reward

            if done:
                success = env.end_flag
                all_success_time += success
                ep_success_time += success
                all_time += 1
                if all_time % args.num_test == 0:
                    check_flag = True

                state, _ = env.reset()

            state = Variable(torch.from_numpy(state).unsqueeze(0))
            if torch.cuda.is_available():
                state = state.cuda()

            if check_flag:
                all_success_rate = all_success_time / all_time
                log_info = 'Num steps: %d, Episode length: %d, Reward: %0.2f, EP Success: %0.2f, ALL Success: %0.3f' \
                            % (itr, episode_length, reward_sum, ep_success_time / args.num_test, all_success_rate)
                test_logger.info(log_info)
                print(log_info)
                torch.save(model.state_dict(), os.path.join(args.model_dir, 'latest.pth'))

                # save models in some important phases
                if all_success_rate > success_phase:
                    torch.save(model.state_dict(),
                               os.path.join(args.model_dir, 'success_rate_%0.2f.pth' % success_phase))
                    success_phase += 0.1

                # save models according to steps
                if epoch % 20 == 0:
                    torch.save(model.state_dict(),
                               os.path.join(args.model_dir, 'model_%d.pth' % epoch))

                reward_sum = 0
                episode_length = 0
                ep_success_time = 0
                check_flag = False
                testing = False