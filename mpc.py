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

def getReward(env,action,p_horizon=1):
    action4env=GD2env(action)
    rewards=[]
    for _ in range(10):
        pred_state=env.dynamics_step(action4env,n=p_horizon)
        reward = 3 * pred_state[1] - abs(pred_state[5]) - 0.5 * abs(pred_state[0]) + 1
        rewards.append(reward)

    pred_reward=np.mean(rewards)
    return pred_reward

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

def MyGD(env,action):
    current_state=env.get_obs()
    current_action=env2GD(action)
    len_a=len(current_action)
    #print(f'action_length:{len_a}')
    current_reward=getReward(env,current_action)
    best_action=current_action
    action_i=current_action
    best_action_i=current_action
    best_reward=current_reward
    best_reward_i=current_reward
    while True:
        count=0
        #start of prediction
        #step forward in one direction
        for i in range(len_a):
            action_i=copy_action(best_action)
            action_i[i]-=0.01
            reward_i=getReward(env,action_i)
            count+=1
            #manually set reward to null if any parameter exceeds boundary
            #if checkBoundary(action_i):
                #reward_i=-math.inf
            if reward_i>best_reward_i:
                #print(f'action_i updated, -0.01 of index{i}')
                best_action_i=action_i
                best_reward_i=reward_i
            #print(f' {count}th time attempted, current action: {action_i}, current reward: {reward_i}')
        #repeat in the other direction
        for i in range(len_a):
            action_i=copy_action(best_action)
            action_i[i]+=0.01
            reward_i=getReward(env,action_i)
            count+=1
            #manually set reward to null if any parameter exceeds boundary
            #if checkBoundary(action_i):
                #reward_i=-math.inf
            if reward_i>best_reward_i:
                #print(f'action_i updated, +0.01 of index{i}')
                best_action_i=action_i
                best_reward_i=reward_i
        # print(f' {count}th time attempted, current action: {action_i}, current reward: {reward_i}')
        '''action_rand=np.random.uniform(-1,1,len(best_action_i))
        reward_rand=getReward(env,action_rand)
        if reward_rand>best_reward_i:
            best_action_i=action_rand
            best_reward_i=reward_rand
            #print(f'  updated to a random position, \nbest action_i: {best_action_i}, \nbest reward_i: {best_reward_i}\n')'''


        if best_reward_i>best_reward:
            best_action=best_action_i
            best_reward=best_reward_i
            #print(f'  updated, \nbest action: {best_action}, \nbest reward: {best_reward}\n')
        else:
            break
    best_action4env=GD2env(best_action)
    #print(f'  GD finished, \nbest action converged to: {best_action}, \nbest reward: {best_reward}\n')
    return best_action4env

def applyNoise(action,dev):
    for i in range(len(action)):
        action[i]+=np.random.normal(loc=0,scale=dev)

    return action
