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
from model import MyDNN,DynamicDataset,train_one_epoch,test,train_forward_model
import pickle
import torch
import torch.nn as nn


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
        'num_steps': 6,
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
def applyNoise(action,dev):
    for i in range(len(action)):
        action[i]+=np.random.normal(loc=0,scale=dev)

    return action

num_samples = Myconfig['N_epoch']*Myconfig['num_steps']
index=0
X = np.zeros((num_samples, 18+16))
Y = np.zeros((num_samples, 18))
def run_one_epoch(n_epoch,Myconfig=Myconfig,env=env):
    obs = env.resetBase()
    all_rewards = []
    SANS_data = []
    epoch_data=np.zeros((Myconfig['num_steps'], 18+16+18))
    for step in range(Myconfig['num_steps']):
        action = np.random.normal(gait_parameters, scale=Myconfig['gait_gaussian'])
        next_obs, reward, done, _ = env.step(action)
        SANS_data.append(np.hstack((obs, action, next_obs)))
        epoch_data[step,:]=np.hstack((obs, action, next_obs))

        obs = next_obs
        all_rewards.append(reward)
        if done:
            break
    logging.info(f"Epoch {n_epoch + 1}: Total Reward: {sum(all_rewards)}")
    return epoch_data


#run serial execution
print('Start running simulation for SANS data by epoch\n')
for epoch in range(Myconfig['N_epoch']):
    epoch_data=run_one_epoch(epoch)
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

#start a new env that implement the trained model
class MyEnv(V000_sm_Env):
    def initModel(self, model_path,env):
        self.model = MyDNN(env=env)
        self.model.load_state_dict(torch.load(model_path))
        self.model_loaded = True
    def dynamics_step(self, action):
        if self.model_loaded:
            # Use the loaded model to predict new state given the current state and action
            self.model.eval()
            state=self.get_obs()
            x=np.r_[state,action]
            x=x.transpose()
            #x=x.tolist()
            #x=x[0]

            with torch.no_grad():
                x = torch.tensor(x,dtype=torch.float32)
                pred = self.model(x)
            # Output should be an array of shape (18,1)
            pred=pred.numpy()
            pred=np.array(pred)
            pred=pred.transpose()
            new_state = pred
            return new_state
            # ---
        else:
            return state
        
myEnv=MyEnv(Myconfig)
myEnv.sleep_time = Myconfig['sleep_time']
_ = myEnv.reset()
myEnv.initModel(model_path,env=env)
print('New dynamic model intialized, start verification\n')
loss_fn = nn.MSELoss()
def test_one_epoch(n_epoch,Myconfig=Myconfig,env=myEnv):
    obs = env.resetBase()
    epoch_loss=[]
    epoch_data=np.zeros((Myconfig['num_steps'], 18+16+18))
    for step in range(Myconfig['num_steps']):
        action = np.random.normal(gait_parameters, scale=Myconfig['gait_gaussian'])
        next_obs, reward, done, _ = env.step(action)
        pred_obs=env.dynamics_step(action)
        obs = next_obs
        next_obs = torch.tensor(next_obs,dtype=torch.float32)
        pred_obs=torch.tensor(pred_obs,dtype=torch.float32)
        step_loss=loss_fn(pred_obs,next_obs)
        epoch_loss.append(step_loss)
        if done:
            break
    logging.info(f"Epoch {n_epoch + 1}: Total Loss: {sum(epoch_loss)}")
    return epoch_data

for epoch in range(10):
    test_one_epoch(epoch)

p.disconnect()

