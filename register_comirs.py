import os
import sys
import csv
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
import time
from skimage.metrics import structural_similarity as ssim



def register_folder(dirA, dirB, config_path, out_path):
    filenames = os.listdir(dirA)
    filenames.sort()
    filenames = [x for x in filenames if x.endswith(".tif") or
                 x.endswith(".png") or
                 x.endswith(".jpg")]
    image_extension = filenames[0][-4:]
    
    filenames = [x.replace(".png", "").replace(".tif", "") for x in filenames]
    N = len(filenames)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i, filename in enumerate(filenames):
        print("Registering image  {} {}/{}".format(filename, i+1,N))

        tforward_name = "tforward_" + filename + ".txt"
        treverse_name = "treverse_" + filename + ".txt"
        
        tforward_path = os.path.join(out_path, tforward_name)
        treverse_path = os.path.join(out_path, treverse_name)
        pathA = os.path.join(dirA, filename + image_extension)
        pathB = os.path.join(dirB, filename +  image_extension)

        process = subprocess.Popen(['../inspire-build/InspireRegister', '2', '-ref',
                                    pathB, '-flo', pathA, '-deform_cfg',
                                    config_path, '-out_path_deform_forward', tforward_path,
                                    '-out_path_deform_reverse', treverse_path],
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
def apply_registration(dirA, dirB, transform_dir, out_path, landmarks=False):
    filenames = os.listdir(dirA)
    filenames.sort()
    filenames = [x for x in filenames if x.endswith(".tif") or
                 x.endswith(".png") or
                 x.endswith(".jpg")]
    image_extension = filenames[0][-4:]
    
    filenames = [x.replace(".png", "").replace(".tif", "") for x in filenames]
    N = len(filenames)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i, filename in enumerate(filenames):
        #print("Registering image  {} {}/{}".format(filename, i+1,N))

        tforward_name = "tforward_" + filename + ".txt"
        treverse_name = "treverse_" + filename + ".txt"
        
        tforward_path = os.path.join(transform_dir, tforward_name)
        treverse_path = os.path.join(transform_dir, treverse_name)
        pathA = os.path.join(dirA, filename + image_extension)
        pathB = os.path.join(dirB, filename +  image_extension)
        
        
        registered_path = os.path.join(out_path, filename + image_extension)
        comir_registered_path = os.path.join(out_path, "comir_" + filename + image_extension)

        
        process = subprocess.Popen(['../inspire-build/InspireTransform', '-dim', '2', '-16bit', '1', 'interpolation', 'linear', '-transform', tforward_path, '-ref', pathB, '-in', pathA, '-out', registered_path, '-bg', "min"],
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if landmarks:
            landmark_filename = filename + ".csv"
            landmarkA_path = os.path.join(dirA, landmark_filename)
            landmarkB_path = os.path.join(dirB, landmark_filename)
            landmark_registered_path = os.path.join(out_path, landmark_filename)
            
            process = subprocess.Popen(['../itkAlphaAMD-build/ACTransformLandmarks', '-dim',
                                        '2', '-transform', tforward_path, '-in', landmarkB_path,
                                        '-out', landmark_registered_path],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()



if __name__ == "__main__":
    modA = sys.argv[1] 
    modB = sys.argv[2]
    modA_comir = sys.argv[3]
    modB_comir = sys.argv[4]
    transforms_dir = sys.argv[5]
    out_path = sys.argv[6]
    out_path_comir = sys.argv[7]
    config_path = sys.argv[8]
    landmarks = sys.argv[9]


    register_folder(modA_comir, modB_comir, config_path, transforms_dir)
    
    
    apply_registration(modA, modB, transforms_dir, out_path, landmarks=landmarks)
    apply_registration(modA_comir, modB_comir, transforms_dir, out_path_comir, landmarks=False)
