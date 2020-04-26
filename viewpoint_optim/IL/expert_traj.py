import pickle
import argparse

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
import sys

# for PCD
# sys.path.append('..')
# from environment import ActiveAgent

# for CNN
sys.path.append('../a2cCNN')
from environment import ActiveAgent

from pointnet import PointNetActorCritic
from utils import setup_logger


def init_parser():
    parser = argparse.ArgumentParser(description='expert_traj')

    parser.add_argument('--model-dir', type=str, default='trained_models',
                    help='Folder to expert models')
    parser.add_argument('--mode', type=str, default='pointnet',
                    help='Feature extraction mode')

    args = parser.parse_args()
    return args


def collect_expert_traj(model_path='latest.pth', mode='pointnet'):
    hidden_size = 1024
    n_traj = 100

    env = ActiveAgent(idx=666, n_points=3000)
    env.seed(456)
    logger = setup_logger('test', 'logs/expert_traj.txt')
    expert_traj = []
    traj = []

    model = PointNetActorCritic(num_points=env.n_points, num_actions=env.n_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    state, _ = env.reset(verbose=True)
    state = Variable(torch.from_numpy(state).unsqueeze(0))
    if torch.cuda.is_available():
        state = state.cuda()
    reward_sum = 0
    done = True
    episode_length = 0

    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            with torch.no_grad():
                cx = torch.zeros(1, hidden_size)
                hx = torch.zeros(1, hidden_size)
        else:
            with torch.no_grad():
                cx = cx.data
                hx = hx.data
        if torch.cuda.is_available():
            hx = hx.cuda()
            cx = cx.cuda()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state, (hx, cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1, keepdim=True)[1].data.cpu().numpy()

        path_info = '%s %s %s %d' % (env.target_group, env.target_scene, env.coord, action[0, 0])
        logger.info(path_info)
        print(path_info)
        traj.append((state.data.cpu().numpy(), action))

        state, reward, done = env.step(action[0, 0])
        reward_sum += reward

        if done:
            success = env.end_flag
            state, _ = env.reset()

            # collect an expert trajectory
            if success and episode_length <= 20:
                log_info = 'Traj %d, episode_length %d, reward %0.2f' \
                            % (n_traj, episode_length, reward_sum)
                logger.info(log_info)
                print(log_info)
                expert_traj.append(traj)
                n_traj -= 1

            traj = []
            episode_length = 0
            reward_sum = 0

        state = Variable(torch.from_numpy(state).unsqueeze(0))
        if torch.cuda.is_available():
            state = state.cuda()

        if not n_traj:
            break

    # save expert trajectory
    with open('expert_traj_%s.pkl' % mode, 'wb') as f:
        pickle.dump(expert_traj, f)


if __name__ == '__main__':
    args = init_parser()
    collect_expert_traj(model_path=args.model_dir, mode=args.mode)