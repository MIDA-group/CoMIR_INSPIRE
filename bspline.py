import SimpleITK as sitk
import numpy as np
import cv2





def create_transform(array, max_displacement=20):
    ctrl_pts =  9, 9
    fix_edges = 3

    ctrl_pts = np.array(ctrl_pts, np.uint32)
    SPLINE_ORDER = 3
    mesh_size = ctrl_pts - SPLINE_ORDER

    image = sitk.GetImageFromArray(array)
    transform = sitk.BSplineTransformInitializer(image, mesh_size.tolist())
    grid_shape = *ctrl_pts, 2

    #max_displacement = 200
    uv = np.random.rand(*grid_shape) - 0.5  # [-0.5, 0.5)

    uv *= 2  # [-1, 1)

    uv *= max_displacement
    

    for i in range(fix_edges):
        uv[i, :] = 0
        uv[-1 - i, :] = 0
        uv[:, i] = 0
        uv[:, -1 - i] = 0
 
    transform.SetParameters(uv.flatten(order='F').tolist())
    return transform

def create_transform_train(array, max_displacement=20):
    ctrl_pts =  5, 5
    fix_edges = 1

    ctrl_pts = np.array(ctrl_pts, np.uint32)
    SPLINE_ORDER = 3
    mesh_size = ctrl_pts - SPLINE_ORDER

    image = sitk.GetImageFromArray(array)
    transform = sitk.BSplineTransformInitializer(image, mesh_size.tolist())
    grid_shape = *ctrl_pts, 2

    #max_displacement = 200
    uv = np.random.rand(*grid_shape) - 0.5  # [-0.5, 0.5)

    uv *= 2  # [-1, 1)

    uv *= max_displacement
    

    for i in range(fix_edges):
        uv[i, :] = 0
        uv[-1 - i, :] = 0
        uv[:, i] = 0
        uv[:, -1 - i] = 0
 
    transform.SetParameters(uv.flatten(order='F').tolist())
    return transform



def transform_image(array, transform):
    image = sitk.GetImageFromArray(array)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.5)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampled = resampler.Execute(image)
    array = sitk.GetArrayViewFromImage(resampled)
    return np.copy(array)


def read_image(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute();
    return image


    

if __name__ == '__main__':

    image1 = read_image("/data2/jiahao/Registration/Datasets/Eliceiri_patches/patch_tlevel3/A/test/1B_A1_R.tif")
    image2 = read_image("/data2/jiahao/Registration/Datasets/Eliceiri_patches/patch_tlevel3/A/test/1B_A3_R.tif")
    
    array1 = sitk.GetArrayViewFromImage(image1)
    array2 = sitk.GetArrayViewFromImage(image2)

    #array1 = array1[0,:,:]
    #array2 = array2[:,:,0]
    transform = create_transform(array1)
    array1_deformed = transform_image(array1, transform) 
    array2_deformed = transform_image(array2, transform) 
    
    cv2.imshow("before", array1/255)
    cv2.imshow("after", array1_deformed/255)
    cv2.waitKey(0)
    cv2.imshow("before", array2/255)
    cv2.imshow("after", array2_deformed/255)
    cv2.waitKey(0)



