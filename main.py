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
        'N_epoch': 100
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

# Perform gradient descent for every step before simulation
def checkBoundary(action):
    isChecked = False
    for i  in range(action):
        if action[i] >1 or action[i] <-1:
            isChecked=True
            break
    return isChecked

def copy_action(action):
    action_copy=np.zeros(len(action))
    for i in range(len(action)):
      action_copy[i]=action[i]
    return action_copy

def getReward(env,stateID,action):
    #set pybullet to given state
    p.restoreState(stateID)
    #get reward by step forward
    action4env=GD2env(action)
    action4env=action
    _,reward,_,_=env.step(action4env)
    #rewind to given state
    p.restoreState(stateID)
    return reward

#to simplify GD reduce the len of action from 16 to 7
def env2GD(gait_parameters):
    action4GD=np.zeros(7)
    action4GD[0]=gait_parameters[0]
    action4GD[1]=gait_parameters[2]
    action4GD[2]=gait_parameters[3]
    action4GD[3]=gait_parameters[4]
    action4GD[4]=gait_parameters[5]
    action4GD[5]=gait_parameters[6]
    action4GD[6]=gait_parameters[8]
    return action4GD

#convert action for GD back to action for env(7 to 16)
def GD2env(action):
    action4env=np.zeros(16)
    action4env[0]=action[0]
    action4env[2]=action[1]
    action4env[3]=action[2]
    action4env[4]=action[3]
    action4env[5]=action[4]
    action4env[6]=action[5]
    action4env[8]=action[6]
    return action4env

def MyGD(env,action,stepID):
    #save the original state for current step
    len_a=len(action)
    print(f'action_length:{len_a}')
    #stepID=p.saveState()
    current_action=action
    current_reward=getReward(env,stepID,current_action)
    best_action=current_action
    action_i=current_action
    best_action_i=current_action
    best_reward=current_reward
    best_reward_i=current_reward
    while True:
        count=0
        p.restoreState(stepID)
        #start of prediction
        #step forward in one direction
        for i in range(len_a):
            action_i=copy_action(best_action)
            action_i[i]-=0.05
            reward_i=getReward(env,stepID,action_i)
            count+=1
            #manually set reward to null if any parameter exceeds boundary
            #if checkBoundary(action_i):
                #reward_i=-math.inf
            if reward_i>best_reward_i:
                best_action_i=action_i
                best_reward_i=reward_i
            #print(f' {count}th time attempted, current action: {action_i}, current reward: {reward_i}')
        #repeat in the other direction
        for i in range(len_a):
            action_i=copy_action(best_action)
            action_i[i]+=0.05
            reward_i=getReward(env,stepID,action_i)
            count+=1
            #manually set reward to null if any parameter exceeds boundary
            #if checkBoundary(action_i):
                #reward_i=-math.inf
            if reward_i>best_reward_i:
                best_action_i=action_i
                best_reward_i=reward_i
        # print(f' {count}th time attempted, current action: {action_i}, current reward: {reward_i}')
        action_rand=np.random.uniform(-1,1,len(best_action_i))
        reward_rand=getReward(env,stepID,action_rand)
        if reward_rand>best_reward_i:
            best_action_i=action_rand
            best_reward_i=reward_rand
            print(f'  updated to a random position, \nbest action_i: {best_action_i}, \nbest reward_i: {best_reward_i}\n')


        if best_reward_i>best_reward:
            best_action=best_action_i
            best_reward=best_reward_i
            print(f'  updated, \nbest action: {best_action}, \nbest reward: {best_reward}\n')
        else:
            break

    #rewind one last time for execution in simulation
    p.restoreState(stepID)
    return best_action

def applyNoise(action,dev):
    for i in range(len(action)):
        action[i]+=np.random.normal(loc=0,scale=dev)

    return action

SANS_data = []  # Sensor and Actuator data storage
num_samples = Myconfig['N_epoch']*Myconfig['num_steps']
index=0
X = np.zeros((num_samples, 18+16))
Y = np.zeros((num_samples, 18))
#run GD and collect data
for epoch in range(Myconfig['N_epoch']):
    obs = env.resetBase()
    all_rewards = []
    current_action4GD=env2GD(gait_parameters)
    #current_action4GD=gait_parameters
    for step in range(Myconfig['num_steps']):
        print(f'Start of step {step}th\n')
        stepID=p.saveState()
        print(f'stepID:{stepID}')
        #action = np.random.normal(gait_parameters, scale=Myconfig['gait_gaussian'])
        next_action4GD=MyGD(env,current_action4GD,stepID)
        #p.restoreState(stepID)
        print(f'best action calculated: {next_action4GD}')
        next_action=GD2env(next_action4GD)
        #action=applyNoise(next_action,dev=0.1)
        action=next_action
        '''
        print(f'action taken: {action}')
        for i in range(4):

            reward_in_algo=getReward(env,stepID,next_action4GD)
            print(f'reward in the algorithm: {reward_in_algo},attempt {i}')
        '''
        p.restoreState(stepID)
        next_obs, reward, done, _ = env.step(action)
        X[index,:]=np.hstack((obs, action))
        Y[index,:]=np.hstack((next_obs))
        index+=1

        SANS_data.append(np.hstack((obs, next_action, next_obs)))

        obs = next_obs
        current_action4GD=action

        all_rewards.append(reward)
        print(f'end of {step}th step, action taken: {next_action}, step reward:{reward}')
        p.removeState(stepID)

        if done:
            break

    # Example logging
    logging.info(f"Epoch {epoch + 1}: Total Reward: {sum(all_rewards)}")
#store data after collection
save_dir='dataset'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
# Save the collected data in the data.pkl file
data = {'X': X, 'Y': Y}
pickle.dump(data, open(os.path.join(save_dir, 'data.pkl'), "wb" ))

#train model
model=MyDNN(env=env)
model_path=train_forward_model(model,datafile = 'dataset/data.pkl')

#manually test controller and model trained

#start a new env that implement the trained model
class MyEnv(V000_sm_Env):
    def initModel(self, model_path,env):
        self.model = MyDNN(env=env)
        self.model.load_state_dict(torch.load(model_path))
        self.model_loaded = True
    def dynamics_step(self, state, action):
        if self.model_loaded:
            # ---
            # Your code goes here
            # Use the loaded model to predict new state given the current state and action
            self.model.eval()
            x=np.r_[state,action]
            x=x.transpose()
            #x=x.tolist()
            #x=x[0]

            with torch.no_grad():
                x = torch.tensor(x,dtype=torch.float32)
                pred = self.model(x)
            # Output should be an array of shape (6,1)
            pred=pred.numpy()
            pred=np.array(pred)
            pred=pred.transpose()
            new_state = pred
            return new_state
            # ---
        else:
            return state
        
myEnv=MyEnv(Myconfig)
myEnv.initModel(model_path,env=env)





p.disconnect()

