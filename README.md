# Clear Face
Clear Face is a python project with C++ library for tracking faces and multiple models detections from faces such as:
 - Gender, Expressions, Illumination, Pose, Occlusion, Age, and Makeup.
	

![Ryan Reynolds & Jake Gyllenhaal Answer the Web's Most Searched Questions _ WIRED](https://github.com/DiaaZiada/ClearFace/blob/master/images/resutl.gif) 

## Content Tabel

 - [Requerments](#requerments)
 - [Download and Setup](#download-and-setup)
 - [Running Options](#running-options)
 - [Gender Expressions and other models](#gender-expressions-and-other-models)
	 * [Datasets](#datasets)
	 * [Models](#models)
	 * [Train](#train)
	 * [Convert to TorchScript](#convert-to-torchscript)
 - [Face Tracking](#face-tracking)
 - [Credits](#credits)
## Requerments
 - [Python](https://www.python.org/) 3.*
 - [Numpy](http://www.numpy.org/)
 - [OpenCV](https://opencv.org/)
 - [Pytorch](https://pytorch.org/)
 - [Imutils](https://pypi.org/project/imutils/)
 - [Cmake](https://cmake.org/)
 - C++
 - make commad 
 
## Download and setup
**Download Repo**
`$git clone https://github.com/DiaaZiada/ClearFace.git`
`$cd ClearFace`

**Download libtorch for c++ (CPU)**
`$./download.sh` note: this command will download and extract the libarary
**Build all c++ files**
`$./build.sh`

## Running Options
to use ClearFace execute `run.py` file with  various options
```
Clear Faces is project for multiple models detection from faces such as
gender, expression, age, etc, and Tracking

optional arguments:
  -h, --help            show this help message and exit
  --show                set this parameter to True value if you want to display
                        images/videos while processing, default is False
  --tracking            set this parameter to True value if you want to track
                        faces in images/videos, default is False
  --delay DELAY         amount of seconds to wait to switch between images
                        while show the precess
  --inputs_path INPUTS_PATH
                        the path for directory contains images/videos to process,
                        if you don't use it webcam will open to start the
                        record
  --video_name VIDEO_NAME
                        name of recorded video from the camera
  --outputs_path OUTPUTS_PATH
                        the path for directory to add the precesses images/videos
                        on it, if you don't use it output directory will
                        created and add the precesses images/videos on it
  --models_path MODELS_PATH
                        the path for directory contains PyTorch model
  --cam_num CAM_NUM     number of camera to use it
  --models MODELS [MODELS ...]
                        the first index refers to the gender model, the second index
                        refers to the expression model, and the third index refers to
                        multiple models
```
**Examples**
for process images/videos in the directory
`$python run.py --inputs_path /path/to/inputs_dir --tracking  --show`  
to use webcam 
`$python run.py --tracking  --show`   
## Gender Expressions and other models
1.  Gender: _Male, Female_
2.  Expressions: _Anger, Happiness, Sadness, Surprise, Fear, Disgust_
3.  Illumination: _Bad, Medium, High_
4.  Pose: _Frontal, Left, Right, Up, Down_
5.  Occlusion: _Glasses, Beard, Ornaments, Hair, Hand, None, Others_
6.  Age: _Child, Young, Middle, and Old_
7.  Makeup: _Partial makeup, Over-makeup_
### Datasets
* [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) ~ 500K image
* [WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) ~ 60K image
* [FER](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) ~ 35K image
* [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/) ~ 3K image
* [JAFFE](http://www.kasrl.org/jaffe.html) ~ 250 image


### Models
models in this project are based on [Real-time Convolutional Neural Networks for Emotion and Gender Classification](https://arxiv.org/pdf/1710.07557.pdf) paper
model architecture: 

![mini exception cnn model](https://github.com/DiaaZiada/ClearFace/blob/master/images/mini_exception_cnn_model.png)

this part of the project consists of 3 models:
	

 1. Gender
 2. Expression
 3. Multiple

### Gender Model
trained done by using [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) , [WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/),  [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/)  datasets. consist of ~ 600 K image, and by using ten crop data augmentation dataset increased to be ~ 6 M image 
Accuracy ~ 78% using only 6 epochs and it will reach higher accuracy expected to be ~ 96 %

### Expression Model
trained done by using [FER](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) , [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/), [JAFFE](http://www.kasrl.org/jaffe.html)  image datasets. consist of ~ 40 K
