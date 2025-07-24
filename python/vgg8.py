import h5py
import torch

from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn

import torch.optim as optim
from torchvision import models

import os, sys
import pandas as pd

class VGG8(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnnlayer00 = nn.Conv1d(in_channels=2, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.act00 = nn.ReLU(inplace=True)
        self.cnnlayer01 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.act01 = nn.ReLU(inplace=True)
        self.maxpool0 = nn.MaxPool1d(kernel_size=2,stride=2) #12x512
        
        self.cnnlayer10 = nn.Conv1d(in_channels=12,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.act10 = nn.ReLU(inplace=True)
        self.cnnlayer11 = nn.Conv1d(in_channels=24,out_channels=24,kernel_size=3,stride=1,padding=1)
        self.act11 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2,stride=2) #24x254
        
        self.cnnlayer20 = nn.Conv1d(in_channels=24,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.act20 = nn.ReLU(inplace=True)
        self.cnnlayer21 = nn.Conv1d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.act21 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2,stride=2) #32x128


        self.fullyconnected0 = nn.Linear(in_features=4096, out_features=100)
        self.act7 = nn.ReLU(inplace=True)
        self.fullyconnected1 = nn.Linear(in_features=100, out_features=100)
        self.act8 = nn.ReLU(inplace=True)
        self.fullyconnected2 = nn.Linear(in_features=100, out_features=24)
        


        self.classes = ['32PSK',
            '16APSK',
            '32QAM',
            'FM',
            'GMSK',
            '32APSK',
            'OQPSK',
            '8ASK',
            'BPSK',
            '8PSK',
            'AM-SSB-SC',
            '4ASK',
            '16PSK',
            '64APSK',
            '128QAM',
            '128APSK',
            'AM-DSB-SC',
            'AM-SSB-WC',
            '64QAM',
            'QPSK',
            '256QAM',
            'AM-DSB-WC',
            'OOK',
            '16QAM']


        self.HyperClassesMapping = {
            '4ASK' : 'ASK',
            '8ASK' : 'ASK',
            'OOK'  : 'ASK',
            'BPSK' : 'PSK',
            '8PSK' : 'PSK',
            '16PSK': 'PSK',
            '32PSK': 'PSK',
            'OQPSK': 'PSK',
            'QPSK' : 'PSK',
            '16APSK':'APSK',
            '32APSK': 'APSK',
            '64APSK' : 'APSK',
            '128APSK': 'APSK',
            '16QAM'  :  'QAM',
            '32QAM'  :  'QAM',
            '64QAM'  :  'QAM',
            '128QAM' :  'QAM',
            '256QAM' :  'QAM',
            'FM'     :  'fM',
            'GMSK'   :  'fM',
            'AM-DSB-SC': 'aM',
            'AM-SSB-WC': 'aM',
            'AM-SSB-SC': 'aM',
            'AM-DSB-WC':  'aM'
        }

        self.HyperClassesIndex = ['ASK', 'PSK', 'APSK', 'QAM', 'fM', 'aM' ]

        self.ClassesHyperClassIndex = {
            '4ASK' : 0,
            '8ASK' : 1,
            'OOK'  : 2,
            'BPSK' : 0,
            '8PSK' : 1,
            '16PSK': 2,
            '32PSK': 3,
            'OQPSK': 4,
            'QPSK' : 5,
            '16APSK': 0,
            '32APSK': 1,
            '64APSK' : 2,
            '128APSK': 3,
            '16QAM'  :  0,
            '32QAM'  :  1,
            '64QAM'  :  2,
            '128QAM' :  3,
            '256QAM' :  4,
            'FM'     :  0,
            'GMSK'   :  1,
            'AM-DSB-SC': 0,
            'AM-SSB-WC': 1,
            'AM-SSB-SC': 2,
            'AM-DSB-WC':  3
        }


    def forward(self, x):
        x = self.cnnlayer00(x)
        x = self.act00(x)
        x = self.cnnlayer01(x)
        x = self.act01(x)
        x = self.maxpool0(x)

        x = self.cnnlayer10(x)
        x = self.act10(x)
        x = self.cnnlayer11(x)
        x = self.act11(x)
        x = self.maxpool1(x)

        x = self.cnnlayer20(x)
        x = self.act20(x)
        x = self.cnnlayer21(x)
        x = self.act21(x)
        x = self.maxpool2(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.fullyconnected0(x)
        x = self.act7(x)
        x = self.fullyconnected1(x)
        x = self.act8(x)
        x = self.fullyconnected2(x)

        return x
    
    def PartialForward(self, x, layerindex):
        x = self.cnnlayer00(x) # layerindex >=0, these two layers are always computed
        
        if layerindex >= 1 :
            x = self.act00(x)
        
        if layerindex >= 2 :
            x = self.cnnlayer01(x)
        if layerindex >= 3 :
            x = self.act01(x)
        if layerindex >= 4 : 
            x = self.maxpool0(x)
        if layerindex >= 5 : 
            x = self.cnnlayer10(x)
        if layerindex >= 6 :
            x = self.act10(x)
        if layerindex >= 7 : 
            x = self.cnnlayer11(x)
        if layerindex >= 8 :    
            x = self.act11(x)
        if layerindex >= 9 : 
            x = self.maxpool1(x)
        if layerindex >= 10 :#Target for first dynamic layer
            x = self.cnnlayer20(x)
        if layerindex >= 11 :    
            x = self.act20(x)
        if layerindex >= 12 : 
            x = self.cnnlayer21(x)
        if layerindex >= 13 :    
            x = self.act21(x)
        if layerindex >= 14 : 
            x = self.maxpool2(x)

        x = x.reshape(x.size(0), -1) #Flattening partial result to train fully connected layer
        
        return x
    

    def LabelMapping(self,labels, HyperClassIdx):
        n_output = self.NumberClassesInHyperClass(HyperClassIdx)
        b = labels.shape
        o = torch.zeros(b[0],n_output)
        for b_ in range(int(b[0])):
            o[b_][
                self.ClassesHyperClassIndex[
                    self.classes[int(labels[b_].item())]]
            ] = 1.0
        
        return o 

    def NumberClassesInHyperClass(self,HyperClassIdx):
        counter = 0 
        for cls in self.classes:
            if self.HyperClassesMapping[cls] == self.HyperClassesIndex[HyperClassIdx]:
                counter += 1
        return counter

    def NumFiltersLayer(self, layerindex):
        if layerindex == 1 :
            return 12
        elif layerindex == 2 :
            return 12
        elif layerindex == 3 :
            return 12
        elif layerindex == 4 : 
            return 12
        elif layerindex == 5 : 
            return 24
        elif layerindex == 6 :
            return 24
        elif layerindex == 7 : 
            return 24
        elif layerindex == 8 :    
            return 24
        elif layerindex == 9 : 
            return 24
        elif layerindex == 10 :
            return 32
        elif layerindex == 11 :    
            return 32
        elif layerindex == 12 : 
            return 32 
        elif layerindex == 13 :    
            return 32
        elif layerindex == 14 : 
            return 32

        else:
            print('Layer Index not in the dynamic range, got:' + str(layerindex))
            sys.exit()



