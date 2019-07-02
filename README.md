# Faces
Faces is project for multiple models detection from faces such as:
1.  Gender : _Male, Female_
2.  Expressions : _Anger, Happiness, Sadness, Surprise, Fear, Disgust_
3.  Illumination : _Bad, Medium, High_
4.  Pose : _Frontal, Left, Right, Up, Down_
5.  Occlusion : _Glasses, Beard, Ornaments, Hair, Hand, None, Others_
6.  Age : _Child, Young, Middle and Old_
7.  Makeup : _Partial makeup, Over-makeup_

[this video test Faces project on part of video of End Game trailer](https://youtu.be/dmqlNalkFUw), and here some examples of prediction on images

![Faces](https://github.com/DiaaZiada/Faces/blob/master/outputs/output_64247356_2248496652127382_3579464820719159220_n.jpg)

![Faces](https://github.com/DiaaZiada/Faces/blob/master/outputs/output_59144945_138097180690991_3236186259228474426_n.jpg)

![Faces](https://github.com/DiaaZiada/Faces/blob/master/outputs/output_55823662_2165766590171098_5989420157986120831_n.jpg)


![Faces](https://github.com/DiaaZiada/Faces/blob/master/outputs/output_56587697_135756350875054_112859982193896992_n.jpg)

![Faces](https://github.com/DiaaZiada/Faces/blob/master/outputs/output_56962151_376686736513461_5038117583971613094_n.jpg)


****Requirements****
 - [Python](https://www.python.org/) 3.*
 - [Imutils](https://pypi.org/project/imutils/)
 - [Numpy](http://www.numpy.org/)
 - [OpenCV](https://opencv.org/)
 - [Pytorch](https://pytorch.org/)

## Run
to use Faces execute `run.py` file with  various options
`
  usage: run.py [-h] [--cuda CUDA] [--show SHOW] [--delay DELAY]
                  [--inputs_path INPUTS_PATH] [--outputs_path OUTPUTS_PATH]
                  [--models_path MODELS_PATH] [--models MODELS [MODELS ...]]
 `                 
    
    optional arguments:
    * -h, --help            show this help message and exit
    
    * --cuda CUDA           set this parameter to True value if you want to use cuda gpu, default is True
    * --show SHOW           set this parameter to True value if you want display images/videos while processing, default is True
    * --delay DELAY         amount of seconds to wait to switch between images while show the precess
    * --inputs_path INPUTS_PATH path for directory contains images/videos to process, if you don't use it web-cam will open to start the record
    * --outputs_path OUTPUTS_PATH path for directory to add the precesses images/videos on it, if you don't use it output directory will created and add the precesses images/videos on it
    * --models_path MODELS_PATH path for directory contains pytorch model
    * --models MODELS [MODELS ...] first index refers to gender model, second index refers to expression model, and third index refers to multiple models Ex: gender, multiple ->> 1,0,1 we set expression to 0 to not use it, default 1,1,1
`

## Datasets
* [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) ~ 500K image
* [WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) ~ 60K image
* [FER](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) ~ 35K image
* [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/) ~ 3K image
* [JAFFE](http://www.kasrl.org/jaffe.html) ~ 250 image


## Models
models in this project are based on [Real-time Convolutional Neural Networks for Emotion and Gender Classification](https://arxiv.org/pdf/1710.07557.pdf) paper
model architecture: 

![mini exception cnn model](https://github.com/DiaaZiada/Faces/blob/master/images/mini_exception_cnn_model.png)

project consist of 3 models:
	

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
## Train
all training process done on [Faces notebook](https://github.com/DiaaZiada/Faces/blob/master/Faces.ipynb) using [Google Colab](https://colab.research.google.com) cloud 
## Credits

[Real-time Convolutional Neural Networks for Emotion and Gender Classification](https://arxiv.org/pdf/1710.07557.pdf) paper


