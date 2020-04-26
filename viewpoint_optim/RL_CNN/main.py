from __future__ import print_function

import numpy as np
import argparse
import os
import sys
sys.path.append('..')
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from environment import ActiveAgent
from cnn import PointNetActorCritic
from utils import setup_logger


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
parser.add_argument('--max-grad-norm', type=float, default=20,
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
parser.add_argument('--num-test', type=int, default=20,
                    help='test time')

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


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = PointNetActorCritic(num_points=args.n_points, num_actions=args.num_actions)
    model = model.cuda()
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

    itr = 0
    epoch = 0
    training_time = 50
    train_logger = setup_logger('trainer', os.path.join(args.log_dir, 'trainer_log.txt'))
    test_logger = setup_logger('test', os.path.join(args.log_dir, 'test_log.txt'))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
            training = True
            episode_length = 0

            state, _ = env.reset(min_vis=False)
            state = Variable(torch.from_numpy(state).unsqueeze(0))
            if torch.cuda.is_available():
                state = state.cuda()
            done = True

            while training:
                if done:
                    cx = Variable(torch.zeros(1, args.hidden_size))
                    hx = Variable(torch.zeros(1, args.hidden_size))
                else:
                    cx = Variable(cx.data)
                    hx = Variable(hx.data)
                if torch.cuda.is_available():
                    hx = hx.cuda()
                    cx = cx.cuda()

                values = []
                log_probs = []
                rewards = []
                entropies = []

                for step in range(args.num_steps):
                    itr += 1
                    episode_length += 1

                    value, logit, (hx, cx) = model((state, (hx, cx)))
                    prob = F.softmax(logit, dim=1)
                    log_prob = F.log_softmax(logit, dim=1)
                    entropy = -(log_prob * prob).sum(1, keepdim=True)
                    entropies.append(entropy)

                    action = prob.multinomial(num_samples=1).data.cpu()
                    _action = Variable(action)
                    if torch.cuda.is_available():
                        _action = _action.cuda()
                    log_prob = log_prob.gather(1, _action)

                    path_info = '%s %s %s %d' % (env.target_group, env.scene_idx, env.coord, action.numpy())
                    # train_logger.info(path_info)

                    state, reward, done = env.step(action.numpy())

                    if done:
                        training = False
                        success = env.end_flag
                        log_info = 'Training Step: [%d - %d], Episode length: %d, Reward: %0.2f, Success: %s' \
                                   % (epoch, train_itr, episode_length, sum(rewards) + reward, str(success))
                        train_logger.info(log_info)
                        print(log_info)
                        print(prob.cpu().detach().numpy()[0])
                        episode_length = 0
                        # state, _ = env.reset(up=min(max(itr // 2500, 3), 6))
                        state, _ = env.reset(min_vis=False)

                    state = Variable(torch.from_numpy(state).unsqueeze(0))
                    if torch.cuda.is_available():
                        state = state.cuda()
                    values.append(value)
                    log_probs.append(log_prob)
                    rewards.append(reward)

                    if done:
                        break

                R = torch.zeros(1, 1)
                if not done:
                    value, _, _ = model((state, (hx, cx)))
                    R = value.data

                policy_loss = 0
                value_loss = 0
                R = Variable(R)
                gae = torch.zeros(1, 1)
                if torch.cuda.is_available():
                    R = R.cuda()
                    gae = gae.cuda()
                values.append(R)
                for i in reversed(range(len(rewards))):
                    R = args.gamma * R + rewards[i]
                    advantage = R - values[i]
                    value_loss = value_loss + 0.5 * advantage.pow(2)

                    # Generalized Advantage Estimataion
                    delta_t = rewards[i] + args.gamma * \
                              values[i + 1].data - values[i].data
                    gae = gae * args.gamma * args.tau + delta_t

                    policy_loss = policy_loss - \
                                  log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

                optimizer.zero_grad()

                (policy_loss + args.value_loss_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()


        ################### testing phase ###################
        model = model.eval()

        state, _ = env.reset(min_vis=False)
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        if torch.cuda.is_available():
            state = state.cuda()
        reward_sum = 0
        done = True

        episode_length = 0
        testing = True
        while testing:
            episode_length += 1
            # Sync with the shared model
            if done:
                with torch.no_grad():
                    cx = torch.zeros(1, args.hidden_size)
                    hx = torch.zeros(1, args.hidden_size)
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

            path_info = '%s %s %s %d' % (env.target_group, env.scene_idx, env.coord, action[0, 0])
            # test_logger.info(path_info)

            state, reward, done = env.step(action[0, 0])
            reward_sum += reward

            if done:
                # print('testing: ', all_time)
                success = env.end_flag
                all_success_time += success
                ep_success_time += success
                all_time += 1
                if all_time % args.num_test == 0:
                    check_flag = True

                state, _ = env.reset(min_vis=False)
                time.sleep(0.1)

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

                time.sleep(1)