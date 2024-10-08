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
import mpc
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
        'num_steps': 10,
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


num_samples = Myconfig['N_epoch']*Myconfig['num_steps']
index=0
X = np.zeros((num_samples, 18+16))
Y = np.zeros((num_samples, 18))



#run serial execution
print('Start running simulation for SANS data by epoch\n')
for epoch in range(Myconfig['N_epoch']):
    epoch_data=run_one_epoch(epoch,Myconfig=Myconfig,env=env)
    X[epoch:epoch+Myconfig['num_steps'],:]=epoch_data[:,0:18+16]
    Y[epoch:epoch+Myconfig['num_steps'],:]=epoch_data[:,18+16:]



save_dir='dataset'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
# Save the collected data in the data.pkl file
data = {'X': X, 'Y': Y}
pickle.dump(data, open(os.path.join(save_dir, 'data.pkl'), "wb" ))
print('Initialize DNN model and train\n')
#train model
model=MyDNN(env=env)
model_path=train_forward_model(model,datafile = 'dataset/data.pkl')
print(f'Training finished, model path saved to: {model_path}\n')
#manually test controller and model trained

        
myEnv=MyEnv(Myconfig)
myEnv.sleep_time = Myconfig['sleep_time']
_ = myEnv.reset()
myEnv.initModel(model_path,env=env)
print('New dynamic model intialized, start verification\n')
loss_fn = nn.MSELoss()

for epoch in range(10):
    test_one_epoch(epoch,Myconfig=Myconfig,env=myEnv)

p.disconnect()

