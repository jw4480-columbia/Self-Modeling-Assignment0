import time
import math
import numpy as np
import pybullet as p
import pybullet_data as pd
import gym
from gym.spaces import Box
import logging
import os
from V000_env import V000_sm_Env
from model import MyDNN,DynamicDataset,MyEnv,train_one_epoch,test,train_forward_model,run_one_epoch,test_one_epoch
import pickle
import torch
import torch.nn as nn

'''This program is for visualization only implementing a model that is already train and tested in main.py, 
no model training is done by simply running this file'''

Myconfig = {
        'para': np.loadtxt("CAD2URDF/para.csv"),
        'urdf_path': "CAD2URDF/V000/urdf/V000.urdf",
        'sleep_time': 1/960,
        'n_sim_steps': 30,
        'sub_step_num': 16,
        'force': 1.8,
        'maxVelocity': 1.5,
        'friction':0.99,
        'robot_name': 'V000',
        'param_path': "CAD2URDF/para.csv",
        'log_path': "data/babbling/",
        'random_seed': 2022,
        'gait_gaussian': 0.1,
        'num_steps': 20,
        'N_epoch': 5000
    }

#p.connect(p.GUI)  # Connect to the PyBullet GUI for visualization
p.connect(p.DIRECT) # Use this to delete GUI for speeding up the program

np.random.seed(Myconfig['random_seed'])
#Load initial parameters
gait_parameters = np.loadtxt(Myconfig['param_path'])
os.makedirs(Myconfig['log_path'], exist_ok=True)  # Ensure log directory exists

# Initialize the environment

env = V000_sm_Env(Myconfig)
env.sleep_time = Myconfig['sleep_time']
_ = env.reset()
print('Pybullet environment initialized\n')
#model=MyDNN(env=env)
model_path='dynamics.pth'
myEnv=MyEnv(Myconfig)
myEnv.sleep_time = Myconfig['sleep_time']
_ = myEnv.reset()
myEnv.initModel(model_path,env=env)
'''print('Test model performance without mpc:')
losses_rand=[]
rewards_rand=[]
for epoch in range(30):
    _,loss,reward=test_one_epoch(epoch,Myconfig=Myconfig,env=myEnv,ismpc=False)
    losses_rand.append(loss)
    rewards_rand.append(reward)

ave_loss_rand=np.mean(losses_rand)
dev_loss_rand=np.var(losses_rand)
ave_reward_rand=np.mean(rewards_rand)
dev_reward_rand=np.var(rewards_rand)
print(f'Loss ave:{ave_loss_rand}, dev: {dev_loss_rand}; Rewards ave:{ave_reward_rand}; dev:{dev_reward_rand}')'''
print('With mpc:')
losses_mpc=[]
rewards_mpc=[]
_ = myEnv.reset()
for epoch in range(30):
    _,loss,reward=test_one_epoch(epoch,Myconfig=Myconfig,env=myEnv,ismpc=True)
    losses_mpc.append(loss)
    rewards_mpc.append(reward)
ave_loss_mpc=np.mean(losses_mpc)
dev_loss_mpc=np.var(losses_mpc)
ave_reward_mpc=np.mean(rewards_mpc)
dev_reward_mpc=np.var(rewards_mpc)
print(f'Loss ave:{ave_loss_mpc}, dev: {dev_loss_mpc}; Rewards ave:{ave_reward_mpc}; dev:{dev_reward_mpc}')
p.disconnect()


