# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:09:54 2019

@author: Diaa Elsayed
"""
import os 
import imutils
import cv2
import torch
from torch.nn import functional as F
from torchvision import transforms

transforms = transforms.Compose([transforms.ToTensor()])

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


def detect(frame,models,labels,frameClone):
    
    rects = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    
    # loop over the face bounding boxes
    for (fX, fY, fW, fH) in rects:
    
       
        face = frame[fY:fY + fH, fX:fX + fW]
        face = cv2.resize(face,(64,64))
        face = transforms(face)
    
        # detect models predictions
        for i in range(len(models)):
            outputs = models[i](face[None])
            if type(outputs) != tuple:
                output = F.softmax(outputs)
                pred = torch.argmax(output)
                acc = output[0][pred]#torch.exp(output[j][pred]) / torch.sum(torch.exp(output[j]))
                label = labels[i][pred]+" "+str(int(acc * 100.))+" %"
                cv2.putText(frameClone, label, (fX + fW + 10, fY + 25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),2)
                
            
            else:
                for j in range(len(outputs)):
                    output = F.softmax(outputs[j])
                    pred = torch.argmax(output)
                    acc = output[0][pred]#torch.exp(output[j][pred]) / torch.sum(torch.exp(output[j]))
                    label = labels[i][j][pred]+" "+str(int(acc * 100.))+" %"
                    cv2.putText(frameClone, label, (fX + fW + 10, fY + 25*i+25*(j) ), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)



def video(path, video, save_path, models, labels, show):
    
    video_bool =  path != ''
    if video_bool:
        camera = cv2.VideoCapture(os.path.join(path,video))
    else:
        camera = cv2.VideoCapture(0)
       
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))
    
    save_path = os.path.join(save_path,'output_'+video)
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
#    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    # keep looping
    while True:
    
        # grab the current frame
        (grabbed, frame) = camera.read()
    
        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if video_bool and not grabbed:
            break
    
        # resize the frame, convert it to framescale, and then clone the
        # original frame so we can draw on it later in the program
#        frame = imutils.resize(frame, wid/th=500)
        frameClone = frame.copy()
        detect(frame,models,labels,frameClone)    
        
        out.write(frameClone)    
        if show:
            cv2.imshow("Faces", frameClone)

        # if the ’q’ key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # cleanup the camera and close any open windows
    camera.release()
    out.release()
    cv2.destroyAllWindows()
    
def image(path, img, save_path, models, labels, seconds, show):
    
    frame = cv2.imread(os.path.join(path,img),1)
    
    frame = imutils.resize(frame, width=500)
    
    frameClone = frame.copy()
    detect(frame,models,labels,frameClone)   
    img = 'output_'+img
    save_path = os.path.join(save_path, img)
    
    cv2.imwrite(save_path, frameClone)
    if show:
        cv2.imshow("Faces", frameClone)
        cv2.waitKey(int(seconds * 1000))
    cv2.destroyAllWindows()
    