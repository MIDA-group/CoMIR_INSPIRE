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



def register_deformed_comirs(modA_path, modB_path, modA_comir_path, modB_comir_path, config_path, out_path):
    
    print("registering comirs")

    filenames = os.listdir(modA_path)
    filenames.sort()
    filenames = [x for x in filenames if x.endswith(".tif") or x.endswith(".png")]
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
        pathA = os.path.join(modA_path, filename + image_extension)
        pathB = os.path.join(modB_path, filename +  image_extension)
        
        comir_pathA = os.path.join(modA_comir_path, filename + image_extension)
        comir_pathB = os.path.join(modB_comir_path, filename + image_extension)
        landmark_filename = filename + ".csv"
        landmarkA_path = os.path.join(modA_path, landmark_filename)
        landmarkB_path = os.path.join(modB_path, landmark_filename)

        landmark_registered_path = os.path.join(out_path, landmark_filename)
        
        registered_path = os.path.join(out_path, filename + image_extension)
        comir_registered_path = os.path.join(out_path, "comir_" + filename + image_extension)
    
        
        #Perform registration
        t = time.time()
        process = subprocess.Popen(['../inspire-build/InspireRegister', '2', '-ref', comir_pathB, '-flo', comir_pathA, '-deform_cfg', config_path, '-out_path_deform_forward', tforward_path, '-out_path_deform_reverse', treverse_path],
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print("comir reg time: ", time.time()-t)
        process = subprocess.Popen(['../inspire-build/InspireTransform', '-dim', '2', '-16bit', '1', 'interpolation', 'linear', '-transform', tforward_path, '-ref', comir_pathB, '-in', comir_pathA, '-out', comir_registered_path],
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        

        process = subprocess.Popen(['../inspire-build/InspireTransform', '-dim', '2', '-16bit', '1', 'interpolation', 'linear', '-transform', tforward_path, '-ref', pathB, '-in', pathA, '-out', registered_path, '-bg', "min"],
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        process = subprocess.Popen(['../itkAlphaAMD-build/ACTransformLandmarks', '-dim', '2', '-transform', tforward_path, '-in', landmarkB_path, '-out', landmark_registered_path],
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        

        #if i > 3:
        #    break


def register_deformed(modA_path, modB_path, modA_original_path, config_path, out_path):
    
    print("registering original")

    filenames = os.listdir(modA_path)
    filenames.sort()
    filenames = [x for x in filenames if x.endswith(".tif") or x.endswith(".png")]
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
        pathA = os.path.join(modA_path, filename + image_extension)
        pathA_original = os.path.join(modA_original_path, filename +  image_extension)
        
        landmark_filename = filename + ".csv"
        landmarkA_path = os.path.join(modA_path, landmark_filename)
        landmarkB_path = os.path.join(modB_path, landmark_filename)
        landmark_registered_path = os.path.join(out_path, landmark_filename)
        
        registered_path = os.path.join(out_path, filename + image_extension)
        comir_registered_path = os.path.join(out_path, "comir_" + filename + image_extension)
    
        
        #Perform registration
        t = time.time()
        process = subprocess.Popen(['../inspire-build/InspireRegister', '2', '-ref', pathA_original, '-flo', pathA, '-deform_cfg', config_path, '-out_path_deform_forward', tforward_path, '-out_path_deform_reverse', treverse_path],
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print("comir reg time: ", time.time()-t)
        process = subprocess.Popen(['../inspire-build/InspireTransform', '-dim', '2', '-16bit', '1', 'interpolation', 'linear', '-transform', tforward_path, '-ref', pathA_original, '-in', pathA, '-out', registered_path],
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    
        
        process = subprocess.Popen(['../itkAlphaAMD-build/ACTransformLandmarks', '-dim', '2', '-transform', tforward_path, '-in', landmarkB_path, '-out', landmark_registered_path],
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()



        

def register_elastix(modA_path, modB_path, elastix_output_dir, gridspacing):
    print("registering elastix")
    if not os.path.exists(elastix_out_dir):
        os.makedirs(elastix_out_dir)
    running_mse_before = []
    running_mse_after = []

    filenames = os.listdir(modA_path)
    filenames_images = [x for x in filenames if  x.endswith(".tif") or x.endswith(".png")]
    filenames_landmarks = [x for x in filenames if x.endswith(".csv")]
    filenames_images.sort()
    filenames_landmarks.sort()
    N = len(filenames_images)
    for i, names in enumerate(zip(filenames_images, filenames_landmarks)):
        
        (image_name, landmarks_name) = names
        print("Registering image  {} {}/{}".format(image_name, i+1,N))
        landmarkA_path = os.path.join(modA_path, landmarks_name)
        landmarkB_path = os.path.join(modB_path, landmarks_name)
        
        pathA = os.path.join(modA_path, image_name)
        pathB = os.path.join(modB_path, image_name)

        registered_landmarks_path = os.path.join(elastix_out_dir, landmarks_name)

        registered_path = os.path.join(elastix_out_dir, image_name)
        
        fixedImage = sitk.ReadImage(pathB, sitk.sitkInt8)
        movingImage = sitk.ReadImage(pathA, sitk.sitkInt8)
        fixedImage = sitk.GetArrayFromImage(fixedImage)
        if len(fixedImage.shape) == 3:
            fixedImage = cv2.cvtColor(fixedImage, cv2.COLOR_BGR2GRAY)
        fixedImage = sitk.GetImageFromArray(fixedImage)


        movingImage = sitk.GetArrayFromImage(movingImage)
        if len(movingImage.shape) == 3:
            movingImage = cv2.cvtColor(movingImage, cv2.COLOR_BGR2GRAY)
        movingImage = sitk.GetImageFromArray(movingImage)
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)
        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMap = sitk.GetDefaultParameterMap("bspline")

        spacings = ('3.5', '2.803221', '1.988100', '1.410000', '1.000000')
        n_res = len(spacings)
        parameterMap['NumberOfResolutions'] = [str(n_res)]
        parameterMap['GridSpacingSchedule'] = spacings[(len(spacings)-n_res)::]
        parameterMap['MaxumNumberOfIterations'] = ['1024']
        parameterMap['FinalGridSpacingInPhysicalUnits']=[str(gridspacing)]
        
        parameterMapVector.append(parameterMap)
        elastixImageFilter.SetParameterMap(parameterMapVector)

        elastixImageFilter.LogToConsoleOff()
        t = time.time()
        elastixImageFilter.Execute()
        print("elastix inference time: ", time.time() - t)
        transformParameterMap = elastixImageFilter.GetTransformParameterMap()

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.Execute()
        deformationField = transformixImageFilter.GetDeformationField()
        deformationField = sitk.GetArrayFromImage(deformationField)
        deformationField = deformationField.astype(np.float64)
        deformationField = sitk.GetImageFromArray(deformationField, True)

        transform = sitk.DisplacementFieldTransform(deformationField)
        pointsA = np.genfromtxt(landmarkA_path, delimiter=',')
        pointsB = np.genfromtxt(landmarkB_path, delimiter=',')
        if len(pointsB.shape) < 2:
            pointsB = np.expand_dims(pointsB, 0)
        registeredPointsB = []
        #if image_name == "LNCaP_do_2_f15_01_02_R.png":
        
        for i in range(pointsB.shape[0]):
            p = pointsB[i,:]
            newp = transform.TransformPoint(p)
            registeredPointsB.append(newp)

        registeredPointsB = np.array(registeredPointsB)
        
        pd.DataFrame(registeredPointsB).to_csv(registered_landmarks_path,index=False,header=False)
        resultsImage = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()).astype(np.uint8)
        cv2.imwrite(registered_path, resultsImage)
        cv2.waitKey(0)


        
def evaluate_registration(path1, path2):

    #running_mse_before = []
    #running_mse_after = []
    distances = []

    filenames = os.listdir(path1)
    filenames.sort()
    filenames = [x for x in filenames if x.endswith(".csv")]
    N = len(filenames)
    for filename in filenames:
        landmark1_path = os.path.join(path1, filename)
        landmark2_path = os.path.join(path2, filename)

        landmarks1 = np.genfromtxt(landmark1_path, delimiter=',')
        landmarks2 = np.genfromtxt(landmark2_path, delimiter=',')
        distance = np.sqrt(np.sum((landmarks1-landmarks2)**2, axis=-1))
        
        distances.append(distance)

        
        #print("MSE A & B: {} --- MSE registered A & B: {}".format(mse_before, mse_after))

    #running_mse_before = [item for sublist in running_mse_before for item in sublist]
    #running_mse_after = [item for sublist in running_mse_after for item in sublist]

    #distances = [item for sublist in distances for item in sublist]
    #print(distances)
    distances = [x.mean() for x in distances]
    #running_mse_before = [np.amax(x) for x in running_mse_before]
    #running_mse_after = [np.amax(x) for x in running_mse_after]
    

    distances = np.array(distances)

    success_rates = []
    thresholds = []
    threshold = 0
    step_size = 0.1
    

    #print(np.mean(running_mse_after < running_mse_before) )
    
    for i in range(200):#200 for zurich
        success_rate = np.sum(distances <= threshold)/len(distances)
        success_rates.append(success_rate)
        thresholds.append(threshold)
        threshold = threshold + step_size
        
        #successes = np.sum(running_mse_before <= threshold)
        #accuracy = successes/len(running_mse_before)
        #accuracies_before.append(accuracy)


    return thresholds, distances, success_rates


def evaluate_MSE(original_dir, registered_dir):
    filenames = os.listdir(original_dir)
    filenames.sort()
    extensions = tuple([".png", ".jpg", "tif"])
    filenames = [x for x in filenames if x.endswith(extensions)]
    diffs = []
    for filename in filenames:
        original_path = os.path.join(original_dir, filename)
        registered_path = os.path.join(registered_dir, filename)
        original = cv2.imread(original_path).astype(np.float32)/255
        registered = cv2.imread(registered_path).astype(np.float32)/255
        h, w, c = original.shape
        cv2.waitKey(0)
        pad = 75
        
        diffs.append(np.mean(np.square(original[pad:w-pad, pad:h-pad,:]-registered[pad:w-pad, pad:h-pad,:])))
        
        
    #diff = np.mean(np.array(diffs))
    diffs = np.array(diffs)

    success_rates = []
    thresholds = []
    threshold = 0
    step_size = 0.0002


    for i in range(200):#200 for zurich
        success_rate = np.sum(diffs <= threshold)/len(diffs)
        success_rates.append(success_rate)
        thresholds.append(threshold)
        threshold = threshold + step_size
    
    return thresholds, diffs, success_rates


print(len(sys.argv), sys.argv)
if len(sys.argv) < 3:
    print('Use: inference_comir.py model_path mod_a_path mod_b_path mod_a_out_path mod_b_out_path')
    sys.exit(-1)

if __name__ == "__main__":
    
    root = sys.argv[1]
    config_path = sys.argv[2]
    voxelmorph = int(sys.argv[3]) == 1
    modA_path = os.path.join(root, "A")
    modB_path = os.path.join(root, "B")
    modA_path_original = os.path.join(root, "A_original")
    modA_comir_path = os.path.join(root, "A_comir")
    modB_comir_path = os.path.join(root, "B_comir")
    out_path = os.path.join(root, "registered")
    elastix_out_dir = os.path.join(root, "elastix")
    if voxelmorph:
        voxelmorph_dir = os.path.join(root, "voxelmorph")
        voxelmorph_MI_dir = os.path.join(root, "voxelmorph_MI")
    gridspacing = sys.argv[4] #Good number is 16 for zuirch or 32 for eliceiri

    out_path_mono = os.path.join(root, "registered_mono")
    
    register_deformed_comirs(modA_path, modB_path, modA_comir_path, modB_comir_path, config_path, out_path)
    register_deformed(modA_path, modB_path, modA_path_original, config_path, out_path_mono)
    register_elastix(modA_path, modB_path, elastix_out_dir, gridspacing)



                                  
    """
    thresholds, distances, success_rate_noreg = evaluate_MSE(modA_path_original, modA_path)
    thresholds, distances_mono, success_rate_inspire = evaluate_MSE(modA_path_original, out_path_mono)
    thresholds, distances_comir, success_rate_comir_inspire = evaluate_MSE(modA_path_original, out_path)
    thresholds, distances_elastix, success_rate_elastix = evaluate_MSE(modA_path_original, elastix_out_dir)
    thresholds, distances_voxelmorph, success_rate_voxelmorph = evaluate_MSE(modA_path_original, voxelmorph_dir)
    if voxelmorph:
        thresholds, distances_voxelmorph, success_rate_voxelmorph = evaluate_MSE(modA_path, voxelmorph_dir)
        thresholds, distances_voxelmorph_MI, success_rate_voxelmorph_MI = evaluate_MSE(modA_path, voxelmorph_MI_dir)
    """
    
    thresholds, distances, success_rate_noreg = evaluate_registration(modA_path, modB_path)
    thresholds, distances_mono, success_rate_inspire = evaluate_registration(modA_path, out_path_mono)
    thresholds, distances_comir, success_rate_comir_inspire = evaluate_registration(modA_path, out_path)
    thresholds, distances_elastix, success_rate_elastix = evaluate_registration(modA_path, elastix_out_dir)
    if voxelmorph:
        thresholds, distances_voxelmorph, success_rate_voxelmorph = evaluate_registration(modA_path, voxelmorph_dir)
        thresholds, distances_voxelmorph_MI, success_rate_voxelmorph_MI = evaluate_registration(modA_path, voxelmorph_MI_dir)
        print(distances_voxelmorph_MI)
        print(distances_voxelmorph)                                  
    
    
    plt.plot(thresholds, success_rate_inspire, label="INSPIRE monomodal")
    plt.plot(thresholds, success_rate_noreg, label="No registration")
    plt.plot(thresholds, success_rate_comir_inspire, label="INSPIRE CoMIR")
    plt.plot(thresholds, success_rate_elastix, label="Elastix")
    if voxelmorph:
        plt.plot(thresholds, success_rate_voxelmorph, label="voxelmorph")
        plt.plot(thresholds, success_rate_voxelmorph_MI, label="voxelmorph MI")
    plt.ylabel('Accuracy')
    plt.xlabel('Landmark distance threshold')
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(os.path.join(out_path, 'accuracies.png'))
    

    no_regisration_result_path = os.path.join(out_path, "success_rate_no_registration.csv")
    comir_inspire_result_path = os.path.join(out_path, "success_rate_comir_inspire.csv")
    inspire_result_path = os.path.join(out_path, "success_rate_inspire.csv")
    elastix_result_path = os.path.join(out_path, "success_rate_elastix.csv")
    if voxelmorph:
        voxelmorph_result_path = os.path.join(out_path, "success_rate_voxelmorph.csv")
        voxelmorph_MI_result_path = os.path.join(out_path, "success_rate_voxelmorph_MI.csv")
    
    
    pd.DataFrame(np.array([thresholds,success_rate_noreg]).T).to_csv(no_regisration_result_path,index=False,header=False)
    pd.DataFrame(np.array([thresholds,success_rate_comir_inspire]).T).to_csv(comir_inspire_result_path,index=False,header=False)
    pd.DataFrame(np.array([thresholds,success_rate_inspire]).T).to_csv(inspire_result_path,index=False,header=False)
    pd.DataFrame(np.array([thresholds,success_rate_elastix]).T).to_csv(elastix_result_path,index=False,header=False)
    if voxelmorph:
        pd.DataFrame(np.array([thresholds,success_rate_voxelmorph]).T).to_csv(voxelmorph_result_path,index=False,header=False)
        pd.DataFrame(np.array([thresholds,success_rate_voxelmorph_MI]).T).to_csv(voxelmorph_MI_result_path,index=False,header=False)
                                  
    
    plt.figure()
    plt.hist(distances_comir, 100, facecolor='blue', alpha=0.5)
    plt.xlim(0,60)
    plt.ylim(0,50)
    plt.savefig(os.path.join(out_path, 'success_rate_histogram.png'))
    plt.figure()
    plt.hist(distances_mono, 100, facecolor='blue', alpha=0.5)
    plt.xlim(0,60)
    plt.ylim(0,50)
    plt.savefig(os.path.join(out_path_mono, 'success_rate_histogram.png'))
    plt.figure()
    plt.hist(distances, 100, facecolor='blue', alpha=0.5)
    plt.xlim(0,60)
    plt.ylim(0,50)
    plt.savefig(os.path.join(out_path, 'success_rate_histogram_no_reg.png'))
    plt.figure()
    plt.scatter(distances, distances_mono)
    plt.savefig(os.path.join(out_path, 'scatter_distances_mono.png'))
    
    filenames = os.listdir(modA_path)
    filenames.sort()
    filenames = [x for x in filenames if x.endswith(".csv")]
    filenames = np.array(filenames)
    print(filenames[(distances_comir > 7)])

        
    

    #print("Total landmark mse and median before registration: {}, {}".format(np.mean(running_mse_before), np.median(running_mse_before)))
    #print("Total landmark mse and median after registration: {}, {}".format(np.mean(running_mse_after), np.median(running_mse_after)))    
    
    
