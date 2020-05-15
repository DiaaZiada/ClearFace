import os
import numpy as np
import cv2

class FaceDetection:
    def __init__(self, path, confidence=0.5, w=None ,h=None):
        
        prototxt = os.path.join(path, "deploy.prototxt")
        model = os.path.join(path, "res10_300x300_ssd_iter_140000.caffemodel")
        self.confidence = confidence
        self.W, self.H = w, h
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

    def __call__(self, image):
        if not self.W or not self.H :
            (self.H, self.W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (self.W, self.H), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        output = detections[0,0,:,:][detections[0,0,:,2] > self.confidence][:,3:7] * np.array([self.W, self.H, self.W, self.H])
        output =  output.astype("int")
        return output

