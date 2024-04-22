# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        
        # Layer 1: Convolutional. Input = 3x224x224. Output = 112x112x64.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3])
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
        
        # Layer 2: Convolutional. Output = 56x56x64.
        self.conv2 = nn.Conv2d(64, 64, kernel_size=[3, 3], padding=[1, 1])
        
        # global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(64, 10)        
        
    def forward(self, x):
        # Activation 1
        x = F.relu(self.conv1(x))
        
        # Pooling 1. Input = 220x220x64. Output = 110x110x64.
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
                
        x = self.avgpool(x)
         
        # Flatten. Input = 54x54x128. Output = 128 * 53 * 53.
        x = torch.flatten(x, 1)
       
        # Fully connected layers
        x = F.relu(self.fc1(x))
        
        return x