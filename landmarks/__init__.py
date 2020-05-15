import os
import cv2
import torch
from torchvision import transforms

from .models import LandMark2D as _2d, LandMark3D as _3d


def unnormalize(points, face_shape):
    points = points.view(68, -1)
    if points.shape[1] == 3:
        points = points* (256. / 1.5) + 256. / 3.
    else:
        points = points* (256. / 4.) + 256. / 2.
    scalar = [face_shape[1]/256,  face_shape[0]/256]
    points = points.detach().numpy()
    points[:,:2] = points[:,:2] * scalar
    return points

def to_orginal_image(points, startX, startY):
    points[:,0] += startX
    points[:,1] += startY
    return points

class LandMarks2D:
    def __init__(self, path):
        self.model = _2d(1)
        self.model.load_state_dict(torch.load(os.path.join(path,"lm2d.pt"), map_location="cpu"))
        self.model.eval()

    def __call__(self, image):
        image_shape = image.shape
        image = cv2.resize(image, (256,256))
        points = self.model(transforms.ToTensor()(image)[None])[0]
        return unnormalize(points, image_shape)

class LandMarks3D:
    def __init__(self, path):
        self.model = _3d(1)
        self.model.load_state_dict(torch.load(os.path.join(path,"lm3d.pt"), map_location="cpu"))
        self.model.eval()

    def __call__(self, image):
        image_shape = image.shape
        image = cv2.resize(image, (256,256))
        points = self.model(transforms.ToTensor()(image)[None])[0]
        return unnormalize(points, image_shape)

