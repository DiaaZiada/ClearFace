import argparse
from utils import input_dir, camera



def args():
    
    parser = argparse.ArgumentParser(description='Clear Face is python project with C++/Python libraries \
    for facial landmark detection(in 2d and 3d), tracking faces and multiple models detection from faces such as:\
 - Gender, Expressions, Illumination, Pose, Occlusion, Age, and Makeup.')
    
    parser.add_argument('--show',action='store_true', default=False,help='set this \
                        parameter to True value if you want display \
                        images/videos while processing, default is False')

    parser.add_argument('--_2d',action='store_true', default=False,help='set this \
                        parameter to True value if you 2d landmarks from \
                        from the 2d model, default is False')   
    
    parser.add_argument('--_3d',action='store_true', default=False,help='set this \
                        parameter to True value if you 3d landmarks from \
                        from the 3d model, default is False')   
    
    parser.add_argument('--_2d3d',action='store_true', default=False,help='set this \
                        parameter to True value if you 2d landmarks from \
                        from the 3d model, default is False')   


    
    parser.add_argument('--tracking', action='store_true', default=False, help='set this \
                        parameter to True value if you want tracking faces in  \
                        images/videos, default is False')  
    
    parser.add_argument('--delay', type=float, default=1, help='amount of \
                        seconds to wait to switch between images while show \
                        the precess')      

    parser.add_argument('--inputs_path', type=str,default='', help='path for \
                        directory contains images/videos to process, if\
                        you don\'t use it webcam will open to start the record')
    
    parser.add_argument('--video_name', type=str,default='output.avi', help='name \
                        of recorded video from camera')
    
    parser.add_argument('--outputs_path', type=str,default='outputs', help='path\
                        for directory to add the precesses images/videos on it,\
                        if you don\'t use it output directory will created and \
                        add the precesses images/videos on it')
    
    parser.add_argument('--models_path', type=str, default='models', help='path \
                        for directory contains pytorch model') 

    parser.add_argument('--cam_num', type=int, default=0, help='number of \
                        camera to use it')

    parser.add_argument('--models', type=int, nargs='+',default=[1,1,1], help='\
                        first index refers to gender model, second index refers\
                        to expression model, and third index refers to multiple models')
    
    return parser.parse_args()

args = args()


show = args.show
tracking = args.tracking

inputs_path = args.inputs_path
outputs_path = args.outputs_path
models_path = args.models_path

_2d = args._2d
_3d = args._3d
_2d3d = args._2d3d


delay = args.delay
bool_gender, bool_expression, bool_multiple = args.models

cam_num = args.cam_num

video_name = args.video_name




if inputs_path == "":
    camera(cam_num, outputs_path, video_name, models_path, bool_expression, bool_gender, bool_multiple, show, tracking, _2d, _3d, _2d3d)
else:
    input_dir(inputs_path, outputs_path, models_path, bool_expression, bool_gender, bool_multiple, delay, show, tracking, _2d, _3d, _2d3d)
