# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:07:14 2019

@author: Diaa Elsayed
"""


from torch import nn 

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
    
class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=2,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
            
        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)
        

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
                            
        x+=skip
        
        return x
    
    
class Feature_Extractor(nn.Module):
    
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        
        self.con2d_1 = nn.Conv2d(in_channels=3,out_channels=8,
                                  kernel_size=(3, 3), stride=(1, 1), padding=0, bias=False)
        self.bn_1 = nn.BatchNorm2d(8)
        
        self.con2d_2 = nn.Conv2d(in_channels=8,out_channels=8,
                                  kernel_size=(3, 3), stride=(1, 1), padding=0, bias=False)
        self.bn_2 = nn.BatchNorm2d(8)
        self.block_1 = Block(in_filters=8, out_filters=16,reps=2)
        self.block_2 = Block(in_filters=16, out_filters=32,reps=2)
        self.block_3 = Block(in_filters=32, out_filters=64,reps=2)
        self.block_4 = Block(in_filters=64, out_filters=128,reps=2)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.con2d_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        
        x = self.con2d_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        
        return x
    
    
class Classifier(nn.Module):
    
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        
        self.conv2d_f = nn.Conv2d(in_channels=128, out_channels=num_classes,
                                  kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.glob_avg_bool = nn.AvgPool2d(kernel_size=(13, 13))
        
    def forward(self, x):
        x = self.conv2d_f(x)
        x = self.glob_avg_bool(x)
        
        return x
    