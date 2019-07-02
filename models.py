# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:09:59 2019

@author: Diaa Elsayed
"""
from torch import nn
from blocks import Feature_Extractor, Classifier

class SingleModel(nn.Module):
    
    def __init__(self,num_classes):
        super(SingleModel, self).__init__()
        
        self.feature_extractor = Feature_Extractor()
        self.classifier = Classifier(num_classes)
    
    def forward(self,x):
        
        x = self.feature_extractor(x)
        x = self.classifier(x)
        
        return x
    
class MultipleModel(nn.Module):
    
    def __init__(self):
        super(MultipleModel, self).__init__()
        
        self.feature_extractor = Feature_Extractor()
        self.illumination = Classifier(3)
        self.pose = Classifier(5)
        self.occlusion = Classifier(7)
        self.age = Classifier(3)
        self.makeup = Classifier(2)
    
    def forward(self,x):
        
        x = self.feature_extractor(x)
        
#         feature_extractor = self.feature_extractor(x)
        illumination = self.illumination(x)
        pose = self.pose(x)
        occlusion = self.occlusion(x) 
        age = self.age(x)
        makeup = self.makeup(x) 

        return illumination, pose, occlusion, age, makeup