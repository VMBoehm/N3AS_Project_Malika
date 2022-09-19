#!/usr/bin/env python
# coding: utf-8

# In[3]:


## all the imports all of them just increase 
import torch
import torchvision
from torchvision import datasets
from torchvision import io
from torchvision import models
from torchvision import ops
from torchvision import transforms
from torchvision import utils
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random


# In[19]:


from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# In[78]:


class MassiveNus(Dataset):
    def __init__(self, datapath, transforms, split='train', seed=123):
        
        np.random.seed(seed)
        files           = list(Path(datapath).rglob('*.npy'))
        np.random.shuffle(files)
        length          = len(files)
        train_split     = int(length*0.7)
        valid_split     = int(length*0.2)
        test_split      = length-train_split-valid_split
        if split=='train':
            self.files  = files[0:train_split]
        elif split=='valid':
            self.files  = files[train_split:train_split+valid_split]
        else:
            self.files  = files[train_split+valid_split::]
        self.length = len(self.files)    
        self.datapath   = Path(datapath)
        self.transforms = transforms
        
    def __len__(self):
        return self.length
                               
    def __getitem__(self, idx):
                               
        y, x, num = np.load(self.files[idx],allow_pickle=True)
        #x    = np.log(x+0.02)
        mean  = 0.00028171288
        std   = 0.0076154615
        x    = (x-mean)/std
                               
        x    = np.expand_dims(x,-1)
        
        x    = self.transforms(x)

                               
        return x.float(), torch.tensor(y).long()


# In[79]:


## I am loading in the data
SEED       = 123
image_size = 512
BATCH_SIZE = 32
DATA_DIR   = '/global/cscratch1/sd/vboehm/Datasets/MassiveNusNumpy/z05'#'/Users/malikagolshan/Desktop/z1'

TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor()])

train_data   = MassiveNus(DATA_DIR,TRANSFORM_IMG, split='train')
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
valid_data   = MassiveNus(DATA_DIR,TRANSFORM_IMG, split='valid')
valid_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_data   = MassiveNus(DATA_DIR,TRANSFORM_IMG, split='train')
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
# test_data         = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=TRANSFORM_IMG)
# test_data_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

## Adrains 

class model_p(nn.Module):
    def __init__(self,  hidden=16, dr= 0.2):
        super(model_p, self).__init__()
        
        kernel_size = 4  # 4
        padding = 1
        
        # input: 1x512x512 ---------------> output: hiddenx256x256 (the stride of 2 with pad of 1 halves dim)
        self.C1 = nn.Conv2d(1, hidden, kernel_size=kernel_size, stride=2, padding=padding, bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx256x256 ----------> output: 2*hiddenx128x128
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=kernel_size, stride=2, padding=padding, bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx128x128 --------> output: 4*hiddenx64x64
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=kernel_size, stride=2, padding=padding, bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=kernel_size, stride=2, padding=padding, bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx32x32 --------> output: 16*hiddenx16x16
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=kernel_size, stride=2, padding=padding, bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx16x16 --------> output: 32*hiddenx8x8
        self.C6 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=kernel_size, stride=2, padding=padding, bias=True)
        self.B6 = nn.BatchNorm2d(32*hidden)
        
        # input: 32*hiddenx8x8 ----------> output: 64*hiddenx4x4
        self.C7 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=kernel_size, stride=2, padding=padding, bias=True)
        self.B7 = nn.BatchNorm2d(64*hidden)
        
        # input: 64*hiddenx4x4 ----------> output: 50x4x4
        self.C8 = nn.Conv2d(64*hidden, 50, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        self.B8 = nn.BatchNorm2d(50)

        self.FC1  = nn.Linear(50*3*3, 400)  
        self.FC2  = nn.Linear(400,   100)    
        self.FC3  = nn.Linear(100,   1)    


        self.Dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        #self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):# or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = self.LeakyReLU(self.B7(self.C7(x)))
        x = self.LeakyReLU(self.B8(self.C8(x)))
        x = x.view(image.shape[0],-1)#x = x.view(x.size(0), -1)#x = x.view(image.shape[0],-1)
        # # #print(x.data.shape)
        x = self.LeakyReLU(self.FC1(x))
        x = self.LeakyReLU(self.FC2(x))
        x = self.Dropout(x)
        x = self.FC3(x)
        #INCLUDE SIGMOID OUTSIDE OF NEWTORK (torch.nn.BCEWithLogitsLoss is more stable)

        return x


# In[ ]:


from torchsummary import summary


# In[ ]:


torch.manual_seed(SEED)
model = model_p().cuda()



#summary(model,(1,512,512))


# In[32]:


# def binary_acc(y_pred, y_test):
#     y_pred_tag = torch.round(torch.sigmoid(y_pred))

#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     acc = correct_results_sum/y_test.shape[0]
#     acc = torch.round(acc * 100)

#     return acc


# In[33]:


#loss_function = nn.BCELoss()  

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)  


# In[34]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[35]:


criterion = nn.BCEWithLogitsLoss()

epochs = 100
    


# In[36]:


np.random.seed(SEED)
torch.manual_seed(SEED)
losses = []
accuracies =[]
for epoch in range(epochs):
 #    if 0:# epoch % 30 == 0 and epoch > 0:
 #        lr /= 10
    # training
    train_loss, train_acc, num_points = 0.0, 0.0, 0
    vlad_loss, train_vlad_acc, num_vlad_points = 0.0, 0.0, 0
    model.train()
    for n, (x,y) in enumerate(train_data_loader):
        y = torch.reshape(y.cuda(),(-1,1))
        y_NN = model(x.cuda())
        loss = criterion(y_NN, y.float())
        binary_pred = (torch.sigmoid(y_NN)>0.5).long()
        acc = torch.mean((y == binary_pred).float())
        train_loss +=  loss.item()
        train_acc  +=  acc.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        if n%20==0:
#            print(train_loss/(n+1))
    train_loss = train_loss/(n+1)
    train_acc = train_acc/(n+1)
    losses.append(train_loss)
    accuracies.append(train_acc)
    print(train_loss, 'the loss')
    print(train_acc, 'the accuracy')
    
    
        for n, (x,y) in enumerate(valid_data_loader):
        y = torch.reshape(y.cuda(),(-1,1))
        y_NN = model(x.cuda())
        loss = criterion(y_NN, y.float())
        binary_pred = (torch.sigmoid(y_NN)>0.5).long()
        acc = torch.mean((y == binary_pred).float())
        train_vlad_loss +=  loss.item()
        train_vlad_acc  +=  acc.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        if n%20==0:
#            print(train_loss/(n+1))
    train_vlad_loss = train_vlad_loss/(n+1)
    train_vlad_acc = train_vlad_acc/(n+1)
    losses.append(train_vlad_loss)
    accuracies.append(train_vlad_acc)
    print(train_vlad_loss, 'the loss')
    print(train_vlad_acc, 'the accuracy')
    
    


# In[ ]:



 


# In[ ]:


# In[ ]:




