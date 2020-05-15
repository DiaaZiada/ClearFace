import os
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from gender import Gender
from expression import Expression
from multiple import Multiple
from face_detection import FaceDetection
from landmarks import LandMarks2D, LandMarks3D, to_orginal_image

from centroid_tracker import CentroidTracker

frame_num = 0
gender, expression, multiple, face_detection, landmarks2d, landmarks3d = [None]*6
ct = CentroidTracker()

transforms = transforms.Compose([transforms.ToTensor()])

image_extensions = ['jpg','png','webp','tiff','psd','raw','bmp','heif','indd']
video_extensions = ['mp4', 'm4a', 'm4v', 'f4v', 'f4a', 'm4b', 'm4r', 'f4b', 'mov','3gp', '3gp2',
 '3g2', '3gpp', '3gpp2','ogg', 'oga', 'ogv', 'ogx','wmv', 'wma', 'asf*','webm',
 'flv','avi','hdv','hdv','mxf']

def draw_2d(image, pts):
    for pt in pts:
        cv2.circle(image,(int(pt[0]),int(pt[1])), 1, (0,255,0), -1)

def plot_3d(num, arrs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    for arr in arrs:
        
        for i ,(x,y,z) in enumerate(arr):
            ax.scatter(x, y, z, color='g')
    
    elev = -85
    azim = -90
    ax.view_init(elev, azim)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(f'./images3d/{num}_3d.png'.format(num))
    # plt.show()

def load_models(path):
    global gender, expression, multiple, face_detection, landmarks2d, landmarks3d
    gender = Gender(os.path.join(path,"gender.zip"))
    expression = Expression(os.path.join(path,"expression.zip"))
    multiple = Multiple(os.path.join(path,"multiple"))
    face_detection = FaceDetection(os.path.join(path,"face_detection"))
    landmarks2d = LandMarks2D(path)
    landmarks3d = LandMarks3D(path)


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
   
def detect(image, expression_bool, gender_bool, multiple_bool, tracking_bool, _2d, _3d, _2d3d):
    global frame_num
    detections = face_detection(image)
    rects = []
    pts_3d = []
    for i in range(detections.shape[0]):
        
        box = detections[i]
        rects.append(box)
        (startX, startY, endX, endY) = box
        face = image[startY:endY, startX:endX]
        if 0 in face.shape:
            continue
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        predictions = models_predictions(face, expression_bool, gender_bool, multiple_bool)
        
        if _3d or _2d3d:
            _pts_3d = landmarks3d(face)
            _pts_3d = to_orginal_image(_pts_3d, startX, startY)

        if _2d3d:
            draw_2d(image, _pts_3d[:,:2])            
        
        if _3d:
            pts_3d.append(_pts_3d)

        if _2d:
            pts_2d = landmarks2d(face)
            pts_2d = to_orginal_image(pts_2d, startX, startY)
            draw_2d(image, pts_2d)
        

        write_predictions(predictions, image, box)
    if _3d:
        plot_3d(frame_num, pts_3d)
        
    draw_tracker(image, tracking_bool, rects)

def process_image(dir_path, img_name, save_path, expression_bool, gender_bool, multiple_bool, seconds, show, tracking_bool, _2d, _3d, _2d3d):
    
    image = cv2.imread(os.path.join(dir_path, img_name))
    
    image = imutils.resize(image, width=500)
    
    detect(image, expression_bool, gender_bool, multiple_bool, tracking_bool, _2d, _3d, _2d3d)
    
    img = f'output_{img_name}'
    save_path = os.path.join(save_path, img)
    
    cv2.imwrite(save_path, image)
    if show:
        cv2.imshow("Faces", image)
        cv2.waitKey(int(seconds * 1000))
    cv2.destroyAllWindows()

def video(dir_path, video_name, save_path, expression_bool, gender_bool, multiple_bool , show, tracking_bool, _2d, _3d, _2d3d):
    global frame_num
    camera = cv2.VideoCapture(os.path.join(dir_path,video_name))
    video_name = video_name.split('.')[0]
    video = f'output_{video_name}.avi'
    save_path = os.path.join(save_path, video)
  

    writer = None
    while True:
    
        _, frame = camera.read()
        frame = imutils.resize(frame, width=800)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if writer is None:
            (h, w) = frame.shape[:2]
            writer =  cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))

        detect(frame, expression_bool, gender_bool, multiple_bool, tracking_bool, _2d, _3d, _2d3d)   
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        writer.write(frame)
    
        if show:
            cv2.imshow("ClearFace", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # cv2.imwrite(f'./images2d/{frame_num}_2d.png', frame)
        frame_num += 1

    camera.release()
    writer.release()
    cv2.destroyAllWindows()
    

    
def camera(camera_num, save_path, video_name, models_path, expression_bool, gender_bool, multiple_bool, show, tracking_bool, _2d, _3d, _2d3d):
    global frame_num

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

        detect(frame, expression_bool, gender_bool, multiple_bool, tracking_bool, _2d, _3d, _2d3d)   
        
        writer.write(frame)    
        if show:
            cv2.imshow("Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_num += 1
    camera.release()
    writer.release()
    cv2.destroyAllWindows()
    
def input_dir(dir_path, save_path, models_path, expression_bool, gender_bool, multiple_bool, seconds, show, tracking_bool, _2d, _3d, _2d3d):

    load_models(models_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    files = os.listdir(dir_path)

    for file in files:

        if file.split('.')[-1] in image_extensions:
            process_image(dir_path, file, save_path, expression_bool, gender_bool, multiple_bool, seconds, show, False,  _2d, _3d, _2d3d)
        elif file.split('.')[-1] in video_extensions:
            video(dir_path, file, save_path, expression_bool, gender_bool, multiple_bool , show, tracking_bool, _2d, _3d, _2d3d)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break        
