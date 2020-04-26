import numpy as np
np.set_printoptions(threshold=np.nan)
import random
import os
import sys
import json
from scipy.misc import imread
from PIL import Image
import argparse

import torch
import torch.nn as nn
from torchvision import transforms

from segmodel import SegmentationModule
from utils import load_snapshot


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img.astype(np.int32)).long()


object_map = {
    'cube': 1,
    'stapler': 2,
    'cup': 3,
    'orange': 4,
    'tape': 5,
    'bowl': 6,
    'box': 7,
    'cola': 8,
    'chip_jar': 9,
    'juice': 10,
    'sugar_jar': 11,
    'spoon': 12,
    'triangle': 13,
    'knife': 14,
    'notebook': 15,
    'rubik_cube': 16,
    'laundry_liquid': 17
}
inv_cls_map = {str(v): k for k, v in object_map.items()}


class ActiveAgent():
    def __init__(self, idx, n_points, seg_args,
                 mode='sim', root_path='the root path of IORD'):
        self.root_path = root_path
        self.scene_path = None
        self.group_list = [4, 6, 9] + list(range(10, 15)) + list(range(20, 36))
        self.idx = idx
        self.logger = open('logs/env_%d.txt' % idx, 'w')

        self.mode = mode
        self.n_points = n_points
        self.n_actions = 5

        # moving information
        self.target_object = 0
        self.coord = [30, 0]
        self.pre_vis = 0
        self.end_thres = 0.85
        self.end_flag = False

        # camera parameters
        self.focalLength_x = 615.747
        self.focalLength_y = 616.041
        self.centerX = 317.017
        self.centerY = 241.722
        self.scalingFactor = 1000.0

        # load segment model
        self.args = seg_args
        print('using mode ', self.mode)
        if self.mode == 'semantic':
            model_dict = load_snapshot(self.args.snapshot, self.args.depth_fusion)
            self.segmodel = SegmentationModule(
                model_dict, 256, 18, self.args.depth_fusion,
                self.args.vote_mode, self.args.vote_scales
            )
            self.segmodel = nn.DataParallel(self.segmodel)
            self.segmodel.load_state_dict(torch.load(
                os.path.join(self.args.model_dir, self.args.depth_fusion, 'epoch_2.pth')
            ))
            self.segmodel = self.segmodel.cuda().eval()

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.40384353, 0.45469216, 0.48145765],
                [0.20519882, 0.21251819, 0.22867874]
            )
        ])
        self.target_transform = MaskToTensor()

    def reset(self, min_vis=True, up=3, verbose=False):
        self.timestep = 0
        self.path = []

        # choose starting point
        self.target_group = 'Group_%d_a' % random.choice(self.group_list)
        self.scene_idx = random.randint(3, 6)
        self.target_scene = sorted(os.listdir(os.path.join(self.root_path, self.target_group)))[self.scene_idx]
        self.scene_path = os.path.join(self.root_path, self.target_group, self.target_scene)
        self.coord = [random.randint(0, 4) * 10 + 30, random.randint(0, 35)]
        self.path.append(self.coord)

        # get objects
        gt = self._get_gt()
        objects = np.unique(gt)[1:]

        # choose target object according to visibility
        done = False
        self.end_flag = False
        with open(os.path.join(self.scene_path, 'RGB', '%d_RGB_%d'
                    % tuple(self.coord), 'vis_demo.json'), 'r') as f:
            object_vis_dic = json.loads(f.read())
            vis_order = sorted(object_vis_dic.items(), key=lambda x: x[1])
            if min_vis:
                target_object_name = vis_order[random.randint(0, min(len(vis_order), 1))][0]
            else:
                min_objects = [object_map[vis_order[0][0]]] * 2
                target_object_name = inv_cls_map[str(random.choice(objects.tolist() + min_objects))]
            self.target_object = object_map[target_object_name]
            self.pre_vis = object_vis_dic.get(target_object_name, 0)

        if verbose:
            self.logger.write('Agent %d starting at %s, scene %s, coord %s \n' % \
                  (self.idx, self.target_group, self.target_scene, str(self.coord)))
            self.logger.write('Agent %d target object is [%d : %s] \n' % (self.idx, self.target_object, target_object_name))
            self.logger.write('Agent %d the initial visibility is %f \n' % (self.idx, self.pre_vis))
            self.logger.flush()

        return self._get_state_from_gt(gt), done

    def step(self, action):
        self.timestep += 1

        # 1-up 2-down 3-left 4-right 0-finish
        assert action in list(range(self.n_actions))
        invalid_ops = False
        if action == 1:
            if self.coord[0] < 70:
                self.coord[0] += 10
            else:
                # self.coord[1] = (self.coord[1] + 18) % 36
                invalid_ops = True
        elif action == 2:
            if self.coord[0] > 30:
                self.coord[0] -= 10
            else:
                invalid_ops = True
        elif action == 3:
            self.coord[1] = (self.coord[1] + 35) % 36
        elif action == 4:
            self.coord[1] = (self.coord[1] + 1) % 36

        # if self.coord in self.path:
        #     invalid_ops = True
        # else:
        #     self.path.append(self.coord)

        done = False
        with open(os.path.join(self.scene_path, 'RGB', '%d_RGB_%d' % tuple(self.coord), 'vis_demo.json'), 'r') as f:
            vis_dic = json.loads(f.read())
            vis = vis_dic.get(inv_cls_map[str(self.target_object)], 0)

        reward = vis - self.pre_vis - int(invalid_ops)
        # reward = 0
        self.pre_vis = vis
        if action == 0:
            done = vis > self.end_thres
            self.end_flag = done
            if done:
                reward = vis * 0.25
            else:
                reward = -0.5 - (1 - vis) - 0.05 * (20 - self.timestep)
                # reward = -1
            done = True

        gt = self._get_gt()

        if self.timestep >= 20:
            done = True
            # reward -= self.timestep * 0.1
            reward = -1

        return self._get_state_from_gt(gt), reward, done

    def _get_gt(self):
        if self.mode == 'sim':
            gt = imread(os.path.join(self.scene_path, 'RGB', '%d_RGB_%d' % tuple(self.coord), 'direct_mask.png')).astype(np.int8)
        elif self.mode == 'semantic':
            # get img
            img = np.array(Image.open(os.path.join(
                self.scene_path, 'RGB', '%d_RGB_%d.jpg' % tuple(self.coord)
            )).convert('RGB')).astype(np.float32) / 255.0
            depth = np.load(os.path.join(
                self.scene_path, 'depth', '%d_depth_%d.npy' % tuple(self.coord)
            ))
            img = self.input_transform(img)
            if self.args.depth_fusion != 'no-depth':
                depth_trans = transforms.ToTensor()(np.expand_dims(depth.astype(np.float32) / 1000.0, axis=2))
                depth_trans = depth_trans.cuda()
            else:
                depth_trans = None

            # forward segment model
            with torch.no_grad():
                probs = self.segmodel(img.unsqueeze(0).cuda(), depth_trans)
            preds = torch.argmax(probs, dim=1).data.cpu().numpy()
            gt = preds.astype(np.int8).squeeze(0)
        elif self.mode == 'instance':
            pass

        return gt

    def _get_state_from_gt(self, gt, step=3):
        tgt_mask = np.zeros(gt.shape).astype(np.int8)
        tgt_mask[gt == self.target_object] = 1
        tgt_mask = tgt_mask.reshape(gt.shape + (1,))

        im_arr = imread(os.path.join(self.scene_path, 'RGB', '%d_RGB_%d.jpg' % tuple(self.coord))).astype(np.float32) / 255.0
        # rgbt = np.concatenate((im_arr, tgt_mask), axis=2).astype(np.float32)

        tgt_mask = np.concatenate((tgt_mask, tgt_mask, tgt_mask), axis=2).astype(np.float32)
        rgbt = np.concatenate((im_arr, tgt_mask), axis=0)

        return rgbt.transpose(2, 0, 1)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def __del__(self):
        self.logger.close()


if __name__ == '__main__':
    pass