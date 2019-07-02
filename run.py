# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:13:16 2019

@author: Diaa Elsayed
"""
import argparse
import os 
import torch

from models import SingleModel, MultipleModel
from util import video,image




def manage():
    
    parser = argparse.ArgumentParser(description='Faces is project for mutilple \
                                     models detection from faces such as gender,\
                                     expression, age etc')

    parser.add_argument('--cuda', type=bool, default=True, help='set this\
                        parameter to True value if you want to use cuda gpu,\
                        default is True')      
    parser.add_argument('--show', type=bool, default=True, help='set this \
                        parameter to True value if you want display \
                        images/videos while processing, default is True')      
    parser.add_argument('--delay', type=float, default=1, help='amount of \
                        seconds to wait to switch between images while show \
                        the precess')      

    parser.add_argument('--inputs_path', type=str,default='', help='path for \
                        directory contains images/videos to process, if\
                        you don\'t use it webcam will open to start the record')
    parser.add_argument('--outputs_path', type=str,default='outputs', help='path\
                        for directory to add the precesses images/videos on it,\
                        if you don\'t use it output directory will created and \
                        add the precesses images/videos on it')
    parser.add_argument('--models_path', type=str, default='models', help='path \
                        for directory contains pytorch model')

    parser.add_argument('--models', type=int, nargs='+',default=[1,1,1], help='\
                        first index refers to gender model, second index refers\
                        to expression model, and third index refers to multiple models\
                        ')
    
    return parser.parse_args()

args = manage()

cuda = args.cuda and torch.cuda.is_available()
show = args.show

inputs_path = args.inputs_path
outputs_path = args.outputs_path
models_path = args.models_path

delay = args.delay
bool_gender, bool_expression, bool_multiple = args.models





models = []
labels = []

if bool_gender:
    gender = SingleModel(2)
    if cuda:    
        gender.load_state_dict(torch.load(os.path.join(models_path,'gender.pth')))
        gender.cuda()
    else:
        gender.load_state_dict(torch.load(os.path.join(models_path,'gender.pth'), map_location='cpu'))
    models.append(gender)
    labels.append(["Female", "Male"])
    
if bool_expression:
    expression = SingleModel(7)
    if cuda:    
        expression.load_state_dict(torch.load(os.path.join(models_path,'expression.pth')))
        expression.cuda()
    else:
        expression.load_state_dict(torch.load(os.path.join(models_path,'expression.pth'), map_location='cpu'))
    models.append(expression)
    labels.append(['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'NEUTRAL', 'SADNESS', 'SURPRISE'])
    
if bool_multiple:
    multiple = MultipleModel()
    if cuda:    
        multiple.load_state_dict(torch.load(os.path.join(models_path,'multiple.pth')))
        multiple.cuda()
    else:
        multiple.load_state_dict(torch.load(os.path.join(models_path,'multiple.pth'), map_location='cpu'))
    models.append(multiple)
    labels.append([['BAD','HIGH','MEDIUM'],
                ['DOWN','FRONTAL','LEFT','RIGHT','UP'],
                ['BEARD','GLASSES','HAIR','HAND','NONE','ORNAMENTS','OTHERS',],
                ['Middle', 'Old', 'Young'],
                ['OVER','PARTIAL']])
    

    
image_extensions = ['jpg','png','webp','tiff','psd','raw','bmp','heif','indd']
video_extensions = ['mp4', 'm4a', 'm4v', 'f4v', 'f4a', 'm4b', 'm4r', 'f4b', 'mov','3gp', '3gp2',
 '3g2', '3gpp', '3gpp2','ogg', 'oga', 'ogv', 'ogx','wmv', 'wma', 'asf*','webm',
 'flv','avi','hdv','hdv','mxf']


if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

if inputs_path == '':
    video(inputs_path, 'output.avi', outputs_path, models, labels, show)

else:
    for file in os.listdir(inputs_path):
        print('\r')
        if file.split('.')[-1] in image_extensions:
            image(inputs_path, file, outputs_path, models, labels, delay,show)
        elif file.split('.')[-1] in video_extensions:
            video(inputs_path, file, outputs_path, models, labels, show)
