
import skimage
import skimage.io as skio

import cv2
import csv

import os
import sys
import os
from tqdm import tqdm
import numpy as np


def save_landmarks(landmarks, path):
    with open(path, 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for landmark in landmarks:
            #point = landmark.pt
            writer.writerow(landmark)

def drawKeypoints(image, keypoints, colors):
    points = [(keypoints[i,0], keypoints[i,1]) for i in range(keypoints.shape[0])]
    points = [cv2.KeyPoint(p[0], p[1],5) for p in points]
    newimg = np.zeros(image.shape)
    newimg = image.copy()
    for keypoint, color in zip(points, colors):
        newimg = cv2.drawKeypoints(newimg, [keypoint], newimg, color)
    return newimg


def draw_matches(img1, img2, points1, points2, pairs):
    points1_cv = make_keypoints(points1)
    points2_cv = make_keypoints(points2)
    matches = make_matches(pairs)
    return cv2.drawMatches(img1, points1_cv, img2, points2_cv, matches, img2, flags=2)

def make_keypoints(points):
    points = [(points[i,0], points[i,1]) for i in range(points.shape[0])]
    points = [cv2.KeyPoint(p[0], p[1],5) for p in points]
    return points

def make_matches(pairs):
    matches = []
    for pair in pairs:
        
        match = cv2.DMatch(pair[0], pair[1], cv2.norm(np.array(1), np.array(1), cv2.NORM_L2))
        matches.append(match)
    return matches
        

def normalize(im, p):
    low = np.percentile(im, 100.0*p)
    hi = np.percentile(im, 100.0*(1.0-p))
    im = (im-low)/(hi-low+1e-15)
    im = np.clip(im, 0.0, 1.0)
    im = im * 255.0
    im = im.astype('uint8')
    return im 


def find_keypoints(img):
    erosion_kernel = kernel = np.ones((5, 5), np.uint8)
    erosion_kernel1 = np.ones((9, 9), np.uint8)
    erosion_kernel2 = np.ones((3, 3), np.uint8)
    erosion_kernel3 = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
    dilation_kernel = np.ones((3, 3), np.uint8)
    img = cv2.GaussianBlur(img,(21,21),1)
    #img =cv2.AdaptiveThreshold(img, img, maxValue, adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, thresholdType=CV_THRESH_BINARY, blockSize=3, param1=5)
    #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret, img = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)
    #img = cv2.erode(img, erosion_kernel)
    img = cv2.erode(img, erosion_kernel2)
    img = cv2.erode(img, erosion_kernel2)
    img = cv2.erode(img, erosion_kernel2)
    img = cv2.erode(img, erosion_kernel2)
    img = cv2.erode(img, erosion_kernel2)
    img = cv2.erode(img, erosion_kernel3)
    img = cv2.dilate(img, dilation_kernel)
    stats = cv2.connectedComponentsWithStats(img, connectivity=4)
    masks = stats[1]
    components = []


    for i in range(np.amax(masks)+1):
        mask = np.zeros(masks.shape, dtype=np.uint8)
        mask[masks==i] = 255
        components.append(mask)
        

    components = np.array(components[1:])
    centroids = stats[3][1:]
    stats = stats[2][1:]
    return centroids, components

def find_matches(keypoints_1, descriptors_1, keypoints_2, descriptors_2, threshold):

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    filtered_matches = []
    
    for match in matches:

        p1, p2 = keypoints_1[match.queryIdx].pt, keypoints_2[match.trainIdx].pt
        if  np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < threshold:
  
            filtered_matches.append(match)
        
    matches = filtered_matches
    

    return matches


def match_keypoints(keypoints1, keypoints2):
    distance_matrix = np.zeros((keypoints1.shape[0], keypoints2.shape[0]))

    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            p1 = keypoints1[i,:]
            p2 = keypoints2[j,:]
            distance_matrix[i,j] = np.linalg.norm(p1-p2)

    pairs = []
    distance_matrix_inv = -distance_matrix + np.amax(distance_matrix)
    
    for k in range(distance_matrix_inv.shape[0]):
        i,j = np.unravel_index(np.argmax(distance_matrix_inv, axis=None), distance_matrix_inv.shape)
        if distance_matrix_inv[i,j] == 0:
            break
        distance_matrix_inv[i,:] = 0
        distance_matrix_inv[:,j] = 0
        if distance_matrix[i,j] < 20:
            pairs.append((i,j))

    
        
    pairs = np.array(pairs)
    return pairs


def sort_keypoints(points1, points2, stats1, stats2, pairs):
    points1_sorted = []
    points2_sorted = []
    stats1_sorted = []
    stats2_sorted = []
    pairs_sorted = []
    for i, pair in enumerate(pairs):
        points1_sorted.append(points1[pair[0]])
        points2_sorted.append(points2[pair[1]])
        stats1_sorted.append(stats1[pair[0]])
        stats2_sorted.append(stats2[pair[1]])
        pairs_sorted.append((i, i))
    points1 = np.array(points1_sorted)
    points2 = np.array(points2_sorted)
    stats1 = np.array(stats1_sorted)
    stats2 = np.array(stats2_sorted)
    pairs = np.array(pairs_sorted)
    return points1, points2, stats1, stats2, pairs




def expand_keypoints(points1, points2, stats1, stats2, pairs):
    points1_expanded = []
    stats1_expanded = []
    points2_expanded = []
    stats2_expanded = []
    pairs_expanded = [(x,x) for x in range(5*len(points1))]
    dilation_kernel = np.ones((3, 3), np.uint8)
    
    for i, data in enumerate(zip(points1, points2, stats1, stats2)):
        p1, p2, s1, s2 = data
        for i in range(5):
            s1 = cv2.dilate(s1, dilation_kernel)
            s2 = cv2.dilate(s2, dilation_kernel)
            
        x, y, w, h = cv2.boundingRect(s1)
        l1 = (x, p1[1])
        r1 = (x+w, p1[1])
        t1 = (p1[0], y)
        b1 = (p1[0], y+h)
        
        x, y, w, h = cv2.boundingRect(s2)
        l2 = (x, p2[1])
        r2 = (x+w, p2[1])
        t2 = (p2[0], y)
        b2 = (p2[0], y+h)


        points1_expanded.extend([p1, l1, r1, t1, b1])
        points2_expanded.extend([p2, l2, r2, t2, b2])
        

    points1 = np.array(points1_expanded)
    points2 = np.array(points2_expanded)
    pairs = np.array(pairs_expanded)

    return points1, points2, pairs
    
path1 = sys.argv[1]
path2 = sys.argv[2]
outpath = sys.argv[3]
framesize = int(sys.argv[4])


name1 = os.path.basename(path1).split('.')[0]
name2 = os.path.basename(path2).split('.')[0]
extension = os.path.basename(path1).split('.')[1]
name = '_'.join(name1.split('_')[1:])

vid1 = skio.imread(path1)
vid1 = skimage.img_as_float(vid1)

vid2 = skio.imread(path2)
vid2 = skimage.img_as_float(vid2)

sift = cv2.xfeatures2d.SIFT_create()

iterative_threshold = 20


for j in range(5):

    f1 = j*2*framesize
    f2 = f1+framesize
    
    img0 = vid1[f1,:,:]
    img1 = vid1[f1+1,:,:]
    img0 = normalize(img0, 0.001)
    img1 = normalize(img1, 0.001)
    img2 = vid1[f2,:,:]
    img2 = normalize(img2, 0.001)
    img0b = vid2[f1,:,:]
    img0b = normalize(img0b, 0.001)

    keypoints0, stats0 = find_keypoints(img0)
    keypoints1, stats1 = find_keypoints(img1)
    pairs1 = match_keypoints(keypoints0, keypoints1)

    keypoints0_cv = make_keypoints(keypoints0)
    keypoints1_cv = make_keypoints(keypoints1)
    matches1 = make_matches(pairs1)
    img3 = cv2.drawMatches(img0, keypoints0_cv, img1, keypoints1_cv, matches1, img1, flags=2)
    #cv2.imshow("matches", img3)



    for i in tqdm(range(f1+2, f2)):
        img2 = vid1[i,:,:]
        img2 = normalize(img2, 0.001)

        img2b = vid2[i,:,:]
        img2b = normalize(img2b, 0.001)
        
        keypoints2, stats2 = find_keypoints(img2)
        pairs2 = match_keypoints(keypoints1, keypoints2)
        filtered_pairs = []

        for pair in pairs1:
            location = np.array(np.where(pairs2[:,0] == pair[1]))
            if location.shape[1] == 1:
                filtered_pairs.append((pair[0], pairs2[:,1][location[0,0]]))

        pairs1 = np.array(filtered_pairs)
        img1 = img2
        keypoints1 = keypoints2


    keypoints0, keypoints2, stats0, stats2, pairs1 = sort_keypoints(keypoints0, keypoints2, stats0, stats2, pairs1)
    keypoints0, keypoints2, pairs1 = expand_keypoints(keypoints0, keypoints2, stats0, stats2, pairs1)

    img3 = draw_matches(img0, img2, keypoints0, keypoints2, pairs1)
    #cv2.imshow("running matches", img3)

    colors = np.random.random(size=(len(keypoints0), 3))*256
    colors = [colors[i,:] for i in range(len(keypoints0))]

    
    
    landmarks_image_1 = drawKeypoints(img0, keypoints0, colors)
    landmarks_image_2 = drawKeypoints(img2, keypoints2, colors)
    #cv2.imshow("landmarks " + str(f1), landmarks_image_1)
    #cv2.imshow("landmarks " + str(f2), landmarks_image_2)
    #cv2.waitKey(0)

    distance = np.mean(np.sqrt(np.sum((keypoints0-keypoints2)**2, axis=-1)))
    print(distance)
    if distance > 3:
    
        outname = "{}_f{}_f{}".format(name, f1, f2)
        cv2.imwrite(os.path.join(outpath, "A_original/", outname + "." + extension ), img0)
        cv2.imwrite(os.path.join(outpath, "A/", outname + "." + extension), img2)
        cv2.imwrite(os.path.join(outpath, "B/", outname + "." + extension), img0b)
        cv2.imwrite(os.path.join(outpath, "B_after/", outname + "." + extension), img2b)
        
        save_landmarks(keypoints2, os.path.join(outpath, "A", outname + ".csv"))
        save_landmarks(keypoints0, os.path.join(outpath, "B", outname + ".csv"))
    else:
        print("Too small distance. Skipped")
