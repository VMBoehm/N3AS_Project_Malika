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



from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os


# fixed settings
SEED       = 1234
image_size = 512
BATCH_SIZE = 32
EPOCHS     = 10
DATA_DIR   = '/global/cscratch1/sd/vboehm/Datasets/MassiveNusNumpy/z05'#'/Users/malikagolshan/Desktop/z1'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


## computes output shape for both convolutions and pooling layers
def compute_size(in_dim,stride,padding,kernel,dilation=1):
    out_dim = np.floor((in_dim + 2*padding - dilation*(kernel-1)-1)/stride+1).astype(int)
    return out_dim


## generalized model class
class model_p(nn.Module):
    def __init__(self,  input_shape, #input shape of single datum, for us (1,512,512)
                        hidden=16,   #hidden size
                        dr= 0.2,    # dropout rate
                        kernel_size=4, #kenrel size
                        padding=1,  # padding, is not used currently
                        downsample_strategy = 'stride', #whether to downsample with stride=2 or pooling, one of 'stride', 'maxpool', 'meanpool'
                        CNN_depth=5, # how many CNN layers
                        activation='LeakyReLU', #activation function
                        activation_params = {'negative_slope':0.2}, #arguments to activation function
                        FC_params=[400,100]): #list of neurons in each fully conneected layer (controls both number and size of FC layers)
        
        super(model_p, self).__init__()
        
        # keep track of current size
        current_size = input_shape[1]
        
        if downsample_strategy == 'stride':
            stride  = 2
            padding = 1
        else:
            stride  = 1
            padding = 0
            
        # keep track of depth
        current_hidden = input_shape[0]
            
        self.activation = getattr(torch.nn,activation)(**activation_params)
            
        self.CNN_layers = []
        self.FC_layers  = []
        # create all CNN layers
        for ii in range(CNN_depth):
        # input: 1x512x512 ---------------> output: hiddenx256x256 (the stride of 2 with pad of 1 halves dim)
        
            self.CNN_layers.append(nn.Conv2d(current_hidden, hidden, kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
            if padding=='same':
                pass
            else:
                current_size = compute_size(current_size,stride,padding,kernel_size)
            self.CNN_layers.append(nn.BatchNorm2d(hidden))
            
            self.CNN_layers.append(self.activation)
        
            if downsample_strategy=='maxpool':
                self.CNN_layers.append(torch.nn.MaxPool2d(kernel_size=2,stride=2))
                current_size = compute_size(current_size,stride=2,padding=0,kernel=2)
            elif downsample_strategy=='meanpool':
                self.CNN_layers.append(torch.nn.AvgPool2d(kernel_size=2,stride=2))
                current_size = compute_size(current_size,stride=2,padding=0,kernel=2)
            else:
                self.CNN_layers.append(torch.nn.Identity())
            current_hidden = hidden
            hidden         = hidden*2
        
        #print(self.CNN_layers)
        self.CNN_layers = torch.nn.ModuleList(self.CNN_layers)
        
        #print(self.CNN_layers)

        # create all FC_layers
        current_flattened_size = current_size**2*current_hidden
        for jj, neurons in enumerate(FC_params):
            self.FC_layers.append(nn.Linear(current_flattened_size, neurons)) 
            self.FC_layers.append(self.activation)
            current_flattened_size = neurons
            
        self.FC_layers.append(nn.Dropout(p=dr)) 
            
        self.FC_layers.append(nn.Linear(current_flattened_size,   1))  
        
        self.FC_layers = torch.nn.ModuleList(self.FC_layers)


    def forward(self, image):
        x = image
        for layer in self.CNN_layers:
            x = layer(x) 
        x = x.view(image.shape[0],-1)
        for layer in self.FC_layers:
            x = layer(x)

        return x







# dataloader as before
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






import optuna

criterion = nn.BCEWithLogitsLoss()




def objective(trial):

    # define parameter ranges
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden        = trial.suggest_categorical('hidden', [4,8,16,32])
    dr            = trial.suggest_float('dropout_rate', 0,0.95)
    kernel_size   = trial.suggest_int('kernel_size', 3,9)
    downsample    = trial.suggest_categorical('downsample_strategy',['stride', 'meanpool', 'maxpool'])
    CNN_depth     = trial.suggest_int('CNN_depth', 2,5)
    FC_depth      = trial.suggest_int('FC_depth', 1,3)
    
    # vary only FC-depth not number of neurons; could be changed
    FC_params = [256,128,32]
    FC_params = [FC_params[ii] for ii in range(FC_depth)]
           
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    
    TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor()])

    train_data   = MassiveNus(DATA_DIR,TRANSFORM_IMG, split='train')
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    valid_data   = MassiveNus(DATA_DIR,TRANSFORM_IMG, split='valid')
    valid_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_data   = MassiveNus(DATA_DIR,TRANSFORM_IMG, split='train')
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    
    
    
    model = model_p([1,512,512],hidden=hidden, dr=dr,kernel_size=kernel_size, downsample_strategy = downsample, 
                    CNN_depth=CNN_depth, activation='LeakyReLU',activation_params = {'negative_slope':0.2},
                    FC_params=FC_params).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)  
    
    
    losses = []
    valid_losses =[]
    accuracies =[]
    valid_accuracies = []
    for epoch in range(EPOCHS):

        train_loss, train_acc, num_points = 0.0, 0.0, 0
        vlad_loss, vlad_acc, vlad_points = 0.0, 0.0, 0
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
        print(train_loss, 'loss')
        print(train_acc, 'accuracy')


        model.eval()
        for n, (x,y) in enumerate(valid_data_loader):
            y = torch.reshape(y.cuda(),(-1,1))
            # VB: turn off gradients
            with torch.no_grad():
                y_NN = model(x.cuda())
                loss = criterion(y_NN, y.float())
            binary_pred = (torch.sigmoid(y_NN)>0.5).long()
            acc = torch.mean((y == binary_pred).float())
            vlad_loss +=  loss.item()
            vlad_acc  +=  acc.item()

        vlad_loss = vlad_loss/(n+1)
        vlad_acc = vlad_acc/(n+1)
        valid_losses.append(vlad_loss)
        valid_accuracies.append(vlad_acc)
        print(vlad_loss, 'validation loss')
        print(vlad_acc, 'validation accuracy')
    
    
    return vlad_acc


study_folder  = '/global/cscratch1/sd/vboehm/OptunaStudies/'
study_name    = "MassiveNus_test"  # Unique identifier of the study.
#study_name    = "Optuna_Tutorial_multi_obj"  # Unique identifier of the study.
study_name    = os.path.join(study_folder, study_name)
storage_name  = "sqlite:///{}.db".format(study_name)

study = optuna.create_study(direction='minimize',study_name=study_name, storage=storage_name,load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=SEED),pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
study.optimize(objective, n_trials=1,timeout=4*60*60-600)

print("Study statistics: ")
print("Number of finished trials: ", len(study.trials))

trial = study.best_trial

print('accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
