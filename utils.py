import os
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from gender import Gender
from expression import Expression
from multiple import Multiple
from face_detection import FaceDetection

from centroid_tracker import CentroidTracker


gender, expression, multiple, face_detection = [None]*4
ct = CentroidTracker()

transforms = transforms.Compose([transforms.ToTensor()])

image_extensions = ['jpg','png','webp','tiff','psd','raw','bmp','heif','indd']
video_extensions = ['mp4', 'm4a', 'm4v', 'f4v', 'f4a', 'm4b', 'm4r', 'f4b', 'mov','3gp', '3gp2',
 '3g2', '3gpp', '3gpp2','ogg', 'oga', 'ogv', 'ogx','wmv', 'wma', 'asf*','webm',
 'flv','avi','hdv','hdv','mxf']

def load_models(path):
    global gender, expression, multiple, face_detection
    gender = Gender(os.path.join(path,"gender.zip"))
    expression = Expression(os.path.join(path,"expression.zip"))
    multiple = Multiple(os.path.join(path,"multiple"))
    face_detection = FaceDetection(os.path.join(path,"face_detection"))

def models_predictions(face, expression_bool, gender_bool, multiple_bool):
    face = cv2.resize(face, (64,64))
    face = transforms(face)
    outputs = []

    if gender_bool:
        output = gender(face[None])
        output = output.view(-1)
        output = F.softmax(output, 0)
        pred = torch.argmax(output)
        acc = int(output[pred] * 100.)
        label = f"{['Female', 'Male'][pred]} {str(acc)}%"
        outputs.append([label, acc])

    if expression_bool:
        output = expression(face[None])  
        output = output.view(-1)
        output = F.softmax(output, 0)
        pred = torch.argmax(output)
        acc = int(output[pred] * 100.)
        label = f"{['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'NEUTRAL', 'SADNESS', 'SURPRISE'][pred]} {str(acc)} %"
        outputs.append([label, acc])

    if multiple_bool:
        output = multiple(face[None])
        labels = [['BAD','HIGH','MEDIUM'],
                ['DOWN','FRONTAL','LEFT','RIGHT','UP'],
                ['BEARD','GLASSES','HAIR','HAND','NONE','ORNAMENTS','OTHERS',],
                ['Middle', 'Old', 'Young'],
                ['OVER','PARTIAL']]
        for i in range(len(output)):
            out = output[i]
            out = out.view(-1)
            out = F.softmax(out, 0)
            pred = torch.argmax(out)
            acc = int(out[pred] * 100.)
            label = f"{labels[i][pred]} {str(acc)}%"
            outputs.append([label, acc])      

    return outputs
        
def write_predictions(labels, image, box):
    (_, startY, endX, _) = box
    for i in range(len(labels)):
        cv2.putText(image, labels[i][0], (endX + 10, startY + 25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),2)
                   
def draw_tracker(image, tracking_bool, rects):

    if not tracking_bool:
        return

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
   
def detect(image, expression_bool, gender_bool, multiple_bool, tracking_bool):

    detections = face_detection(image)
    rects = []
    for i in range(detections.shape[0]):
        
        box = detections[i]
        rects.append(box)
        (startX, startY, endX, endY) = box
        face = image[startY:endY, startX:endX]
        if 0 in face.shape:
            continue
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        predictions = models_predictions(face, expression_bool, gender_bool, multiple_bool)
        write_predictions(predictions, image, box)
    draw_tracker(image, tracking_bool, rects)

def process_image(dir_path, img_name, save_path, expression_bool, gender_bool, multiple_bool, seconds, show, tracking_bool):
    
    image = cv2.imread(os.path.join(dir_path, img_name))
    
    image = imutils.resize(image, width=500)
    
    detect(image, expression_bool, gender_bool, multiple_bool, tracking_bool)
    
    img = f'output_{img_name}'
    save_path = os.path.join(save_path, img)
    
    cv2.imwrite(save_path, image)
    if show:
        cv2.imshow("Faces", image)
        cv2.waitKey(int(seconds * 1000))
    cv2.destroyAllWindows()

def video(dir_path, video_name, save_path, expression_bool, gender_bool, multiple_bool , show, tracking_bool):
    
    camera = cv2.VideoCapture(os.path.join(dir_path,video_name))
    video_name = video_name.split('.')[0]
    video = f'output_{video_name}.avi'
    save_path = os.path.join(save_path, video)
  

    writer = None
    while True:
    
        _, frame = camera.read()
        frame = imutils.resize(frame, width=800)

        if writer is None:
            (h, w) = frame.shape[:2]
            writer =  cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))

        detect(frame, expression_bool, gender_bool, multiple_bool, tracking_bool)   
        
        writer.write(frame)
    
        if show:
            cv2.imshow("ClearFace", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    writer.release()
    cv2.destroyAllWindows()
    

    
def camera(camera_num=0, save_path="output", video_name="/output.avi",models_path='./models', expression_bool=True, gender_bool=True, multiple_bool=True, show=True, tracking_bool=True):
    load_models(models_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    camera = cv2.VideoCapture(camera_num)
    
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))
    save_path = os.path.join(save_path, video_name)
    writer = None 

    while True:
    
        (ret, frame) = camera.read()
        
        if not ret: 
            break

        frame = imutils.resize(frame, width=400)

        if writer is None:
            (h, w) = frame.shape[:2]
            writer =  cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))

        detect(frame, expression_bool, gender_bool, multiple_bool, tracking_bool)   
        
        writer.write(frame)    
        if show:
            cv2.imshow("Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    writer.release()
    cv2.destroyAllWindows()
    
def input_dir(dir_path="inputs", save_path="outputs",models_path='./models', expression_bool=True, gender_bool=True, multiple_bool=True, seconds=3, show=True, tracking_bool=True):

    load_models(models_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    files = os.listdir(dir_path)

    for file in files:

        if file.split('.')[-1] in image_extensions:
            process_image(dir_path, file, save_path, expression_bool, gender_bool, multiple_bool, seconds, show, False)
        elif file.split('.')[-1] in video_extensions:
            video(dir_path, file, save_path, expression_bool, gender_bool, multiple_bool , show, tracking_bool)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break        
