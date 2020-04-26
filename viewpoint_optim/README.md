# Viewpoint Optimization

Here is the code for training an agent with RL (reinforcement learning) or IL (imitation learning) to optimize the viewpoint.

## Requirements

* Python 3.x

* pytorch 0.4.1

## To run

### Training & Evaluation with RL

Use PointNet to extract feature:

  1. `cd RL_pointnet`
  2. `python main.py --data_dir [folder to IORD] --model_dir [folder to save models] --lr [learning rate] --n_points [number of points feed to PointNet]`

  For evaluation:
  `python evaluate.py --data_dir [folder to IORD] --model_dir [folder to save models]`

Use CNN to extract feature:

  1. `cd RL_CNN`
  2. `python main.py --data_dir [folder to IORD] --model_dir [folder to save models] --lr [learning rate]`

### Training & Evaluation with IL

Use PointNet to extract feature:

  1. `cd IL`
  2. `python expert_traj.py --model_dir [expert model] --mode pointnet`
  3. `python main.py --data_dir [folder to IORD] --model_dir [folder to save models] --lr [learning rate --n_points [number of points feed to PointNet]`

  For evaluation:
  `python evaluate.py --data_dir [folder to IORD] --model_dir [folder to save models]`


Use CNN to extract feature:

  1. `cd il`
  2. `python expert_traj.py --model_dir [expert model] --mode cnn`
  3. `python main.py --data_dir [folder to IORD] --model_dir [folder to save models] --lr [learning rate]`