import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import tqdm
import pickle
import torch.optim as optim
import argparse
import time
import numpy as np

class MyDNN(nn.Module):
    def __init__(self,env):
        super(MyDNN, self).__init__()
        self.num_obs=len(env.obs)
        self.num_act=16
        self.fc1 = nn.Linear(self.num_obs+self.num_act, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc2_1 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,self.num_obs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_1(x))
        x = self.fc3(x)
        return x
    
class DynamicDataset(Dataset):
  def __init__(self, datafile):
    data = pickle.load(open(datafile, 'rb'))
    # X: (N, 18+16=34), Y: (N, 18)
    self.X = data['X'].astype(np.float32)
    self.Y = data['Y'].astype(np.float32)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]

loss_fn = nn.MSELoss()

def train_one_epoch(train_loader, model,optim):
  model.train()
	# ---
	# Your code goes here
  size = len(train_loader.dataset)
  num_batches = len(train_loader)
  optimizer = optim
  scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.95,patience=5)
  loss_fn = nn.MSELoss()
  train_loss=0
  for X,y in train_loader:
      #print(X)
      # Compute prediction error
      pred = model(X)
      loss = loss_fn(pred, y)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss +=  loss_fn(pred, y).item()
  train_loss /= num_batches
  print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")
  return train_loss

	# ---

def test(test_loader, model):
  model.eval()
	# --
	# Your code goes here
  # --
  size = len(test_loader.dataset)
  num_batches = len(test_loader)
  test_loss = 0
  loss_fn = nn.MSELoss()
  with torch.no_grad():
    for X, y in test_loader:
        pred = model(X)
        test_loss +=  loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
  return test_loss

def train_forward_model(model,datafile):

	# --
	# Implement this function
  # --

  # Keep track of the checkpoint with the smallest test loss and save in model_path
  #model_path = None
  max_test_loss = 1e4

  split = 0.2
  dataset = DynamicDataset(datafile)
  dataset_size = len(dataset)
  test_size = int(np.floor(split * dataset_size))
  train_size = dataset_size - test_size
  train_set, test_set = random_split(dataset, [train_size, test_size])

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

  # The name of the directory to save all the checkpoints
  timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
  model_dir = os.path.join('models', timestr)

  loss_min=10

  epochs = 30
  optim=torch.optim.Adam(model.parameters(),lr=0.01)
  for epoch in range(1, 1 + epochs):
    # --
    # Your code goes here
    print(f"Epoch {epoch}\n-------------------------------")
    if epoch>20:
      optim=torch.optim.SGD(model.parameters(),lr=0.01)
    train_loss=train_one_epoch(train_loader, model,optim)
    test_loss=test(test_loader, model)
    if epoch==1:
      loss_min=test_loss
    model_folder_name = f'epoch_{epoch:04d}_loss_{test_loss:.8f}'
    if not os.path.exists(os.path.join(model_dir, model_folder_name)):
        os.makedirs(os.path.join(model_dir, model_folder_name))
    torch.save(model.state_dict(), os.path.join(model_dir, model_folder_name, 'dynamics.pth'))
    if epoch==1:
      loss_min=test_loss
      model_path=os.path.join(model_dir, model_folder_name, 'dynamics.pth')
    if test_loss< loss_min:
      print('model updated')
      loss_min=test_loss
      model_path=os.path.join(model_dir, model_folder_name, 'dynamics.pth')
      torch.save(model.state_dict(), 'dynamics.pth')


    # --

  return model_path