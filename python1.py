#!C:/Users/Shahnawaz/Anaconda3/python.exe
import sys

print("Content-Type: text/html\n")
from datetime import date
import torch
import numpy as np
from torchvision import datasets,transforms
import torch.nn.functional as F
from torch import nn
from torch import optim
classes=['cardboard','fabrics','plastic','poly','wet']


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 128x128x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.bn1=nn.BatchNorm2d(16)
        # convolutional layer (sees 64x64x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2=nn.BatchNorm2d(32)
        # convolutional layer (sees 32x32x32 tensor)
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
                                #sees 16*16*48
        self.bn3=nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 24, 3, padding=1)
                                #sees 8*8*24
        self.bn4=nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(24, 12, 8, padding=0)
        
        #output 1*1*6
        
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(12,8)
        self.fc2=nn.Linear(8,5)
        self.drop=nn.Dropout(p=0.30)
        

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        #print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x=self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x=self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x=self.bn3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x=self.bn4(x)
        x = self.conv5(x)
        #print(x.shape)
        x = x.view(-1,1*1*12)
        x = self.drop(self.fc1(x))
        x=self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x
model=Net()
optimizer=optim.Adam(model.parameters(),lr=0.003)
criterion=nn.NLLLoss()

state=torch.load('copy2.pt')
model.load_state_dict(state)
print("up")

model.eval()
transform = transforms.Compose([transforms.Resize(128),transforms.ToTensor()])
dataset = datasets.ImageFolder('C:/xampp/htdocs/gpp/images/', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
for data,target in test_loader:
    output = model(data)
    #break;
output=torch.exp(output)
#print(output.shape)
print("raw output is ",torch.exp(output))
_,output=torch.max(output,dim=1)

print(dataset.class_to_idx)
print("ground truth is ",target)
print("prediction is   ",output)

import os

print(classes)
today = date.today()
print("Today's date:", today)
print ("Hello Python Web Browser!! This is cool!!")
print(sys.path())