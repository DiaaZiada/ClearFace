

# Clear Face
Clear Face is python project with C++/Python libraries for facial landmark detection(in 2d and 3d), tracking faces and multiple models detection from faces such as:
 - Gender, Expressions, Illumination, Pose, Occlusion, Age, and Makeup.
	

![Ryan Reynolds & Jake Gyllenhaal Answer the Web's Most Searched Questions _ WIRED](https://github.com/DiaaZiada/ClearFace/blob/master/images/ClearFace.gif)

## Content Tabel

 - [Download and Setup](#download-and-setup)

 - [Running Options](#running-options)

 - [Gender Expressions and other models](#gender-expressions-and-other-models)

 - [Face Tracking](#face-tracking)

 - [Facial Landmark Detection](#facial-landmark-detection)

 - [Credits](#credits)
## Requerments
 - [Python](https://www.python.org/) 3.*
 - [Cmake](https://cmake.org/)
 - C++
 - make commad 
 
## Download and setup
**Download Repo**
```bash 
git clone https://github.com/DiaaZiada/ClearFace.git
```
```bash
cd ClearFace
```
**Virtual Environment**
```bash
# create virtual environment
python -m venv ./venv
```
```bash
# activate virtual environment
source venv/bin/activate
```

**Install Requirements**
```bash
pip install -r requirements.txt
```
**Download libtorch for c++ (CPU)**
```bash
./download.sh
``` 
note: this command will download and extract the libarary

**Build all c++ files**
```bash
./build.sh
```

## Running Options
to use ClearFace execute `run.py` file with  various options
```
usage: run.py [-h] [--show] [--_2d] [--_3d] [--_2d3d] [--tracking]
              [--delay DELAY] [--inputs_path INPUTS_PATH]
              [--video_name VIDEO_NAME] [--outputs_path OUTPUTS_PATH]
              [--models_path MODELS_PATH] [--cam_num CAM_NUM]
              [--models MODELS [MODELS ...]]

Clear Face is python project with C++/Python libraries for facial landmark detection(in 2d and 3d), tracking faces and multiple models detection from faces such as:
 - Gender, Expressions, Illumination, Pose, Occlusion, Age, and Makeup.

optional arguments:
  -h, --help            show this help message and exit
  --show                set this parameter to True value if you want display
                        images/videos while processing, default is False
  --_2d                 set this parameter to True value if you 2d landmarks
                        from from the 2d model, default is False
  --_3d                 set this parameter to True value if you 3d landmarks
                        from from the 3d model, default is False
  --_2d3d               set this parameter to True value if you 2d landmarks
                        from from the 3d model, default is False
  --tracking            set this parameter to True value if you want tracking
                        faces in images/videos, default is False
  --delay DELAY         amount of seconds to wait to switch between images
                        while show the precess
  --inputs_path INPUTS_PATH
                        path for directory contains images/videos to process,
                        if you don't use it webcam will open to start the
                        record
  --video_name VIDEO_NAME
                        name of recorded video from camera
  --outputs_path OUTPUTS_PATH
                        path for directory to add the precesses images/videos
                        on it, if you don't use it output directory will
                        created and add the precesses images/videos on it
  --models_path MODELS_PATH
                        path for directory contains pytorch model
  --cam_num CAM_NUM     number of camera to use it
  --models MODELS [MODELS ...]
                        first index refers to gender model, second index
                        refers to expression model, and third index refers to
                        multiple models

```
**Examples**
for process images/videos in directory
`$python run.py --inputs_path /path/to/inputs_dir --tracking  --show --_2d3d --_3d`  
to use webcam 
`$python run.py ---tracking  --show --_2d3d --_3d`   

## Gender Expressions and other models
1.  Gender : _Male, Female_
2.  Expressions : _Anger, Happiness, Sadness, Surprise, Fear, Disgust_
3.  Illumination : _Bad, Medium, High_
4.  Pose : _Frontal, Left, Right, Up, Down_
5.  Occlusion : _Glasses, Beard, Ornaments, Hair, Hand, None, Others_
6.  Age : _Child, Young, Middle and Old_
7.  Makeup : _Partial makeup, Over-makeup_
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

this part of the project consist of 3 models:
	

 1. Gender
 2. Expression
 3. Multiple

### Gender Model
trained done by using [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) , [WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/),  [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/)  datasets. consist of ~ 600 K image, and by using tencrop data augmentation dataset increased to be ~ 6 M image 
Accuracy ~ 78% using only 6 epochs and it will reach higher accuracy expected to be ~ 96 %

### Expression Model
trained done by using [FER](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) , [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/), [JAFFE](http://www.kasrl.org/jaffe.html)  image datasets. consist of ~ 40 K image, and by using tencrop data augmentation dataset increased to be ~ 400 K image 
Accuracy ~ 60% using 15 epochs and it will reach higher accuracy expected to be ~ 66 %

### Multiple Models
this model is little bit different form Gender & Expression models 
in this model we use one feature extractor model and 5 different classifiers each classifier predict specific features form faces and they are:
* Illumination : _Bad, Medium, High_
*  Pose : _Frontal, Left, Right, Up, Down_
*  Occlusion : _Glasses, Beard, Ornaments, Hair, Hand, None, Others_
*  Age : _Child, Young, Middle and Old_
*  Makeup : _Partial makeup, Over-makeup_

trained done by using [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/) image datasets consist of ~ 3 K image, and by using tencrop data augmentation dataset increased to be ~ 30 K image 
Accuracy ~ 77% using 15 epochs
### Train
all training process done on [notebook](https://github.com/DiaaZiada/ClearFace/blob/master/notebooks/Gender%2C%20Expressions%2C%20Illumination%2C%20Pose%2C%20Occlusion%2C%20Age%2C%20and%20Makeup.ipynb) using [Google Colab](https://colab.research.google.com) cloud 
### Convert to TorchScript

we convert ower models to torchscript to load them in c++
and use them as extention of ClearFace
all convertions about this part will found in this [notebook](https://github.com/DiaaZiada/ClearFace/blob/master/notebooks/ConvToTorchScript.ipynb)

## Face Tracking
we used the centroid tracking algorithm
## Facial Landmark Detection
### DataSet
[kaggle youtube faces with facial keypoints](https://www.kaggle.com/selfishgene/youtube-faces-with-facial-keypoints) ~ 150K image

### Train
all training process for the 2d and 3d model done in this [notebook](https://github.com/DiaaZiada/ClearFace/blob/master/notebooks/Facial_Keypoints_Detection.ipynb) using [Google Colab](https://colab.research.google.com) cloud 

### Pre-Trained Models
You can email me on **diaa.elsayedziada@gmail.com** to get the pre-trained models

## Credits

* [Real-time Convolutional Neural Networks for Emotion and Gender Classification](https://arxiv.org/pdf/1710.07557.pdf) paper
* [Simple object tracking with OpenCV by  pyimagesearch](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)
* [Extending TorchScript with Custom C++ Operators](https://github.com/pytorch/tutorials/blob/master/advanced_source/torch_script_custom_ops.rst)
* [CMake Tutorial](https://medium.com/@onur.dundar1/cmake-tutorial-585dd180109b)
* [Ryan Reynolds and Jake Gyllenhaal interview for LIFE, DEADPOOL - UNCENSORED](https://www.youtube.com/watch?v=_kB5K33bd6Y&t=3s) Youtube
* [1adrianb face alignment](https://github.com/1adrianb/face-alignment) repo

