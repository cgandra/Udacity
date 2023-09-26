# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## MP.1 Data Buffer Optimization

Implemented class RingBuffer with stl deque of fixed size. 
New elements are inserted at the back of the queue. Elements are removed from the front of the queue when queue is full

## MP.2 Keypoint Detection

detectKeypoints function added to handle the required detectors selectable via detectorType string.
Harris Corner detector from class exercise was adopted. Rest were added from the approp opencv Feature2D classes.
Default parameters for each detector were used. Octave unpacking was handled for SIFT detector
so that ORB etc detectors that rely on octave field work correctly

## MP.3 Keypoint Removal

getKeypointsInROI function was added that checks for a point in the roi to retain it, ow erases it from vector

## MP.4 Keypoint Descriptors

descKeypoints function added to handle the required descriptors selectable via descriptorType string.
OpenCV DescriptorExtractor wrappers used for this with default parameters

## MP.5 Descriptor Matching

DescriptorMatcher::FLANNBASED matcher added. Output data convered to float as required by nn/knn functions.
BFMatcher norm type modified for SIFT (NORM_L2)

## MP.6 K-Nearest-Neighbor matching

KNN match added along with distance ratio filtering with threshold of 0.8

## MP.7 Performance Evaluation 1

Performance Logs added to code. And summarized using the parseLog python script in scripts folder
Sum of keypoints across all 10 images shown below

Detector | #Total Keypoints | Per Img KPs
------------ | ------------- | -------------
SHITOMASI | 13423 | [1370, 1301, 1361, 1358, 1333, 1284, 1322, 1366, 1389, 1339]
HARRIS | 5073 | [490, 499, 508, 509, 522, 510, 491, 511, 539, 494]
FAST | 49204 | [5063, 4952, 4863, 4840, 4856, 4899, 4870, 4868, 4996, 4997]
BRISK | 27116 | [2757, 2777, 2741, 2735, 2757, 2695, 2715, 2628, 2639, 2672]
ORB | 5000 | [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
AKAZE | 13429 | [1351, 1327, 1311, 1351, 1360, 1347, 1363, 1331, 1357, 1331]
SIFT | 13862 | [1438, 1371, 1380, 1335, 1305, 1370, 1396, 1382, 1463, 1422]

Note of the distribution of their neighborhood size

Detector | Neighborhood Size(used)
------------ | -------------
SHITOMASI | 4
HARRIS | adaptive non-maximal suppression for each interest point we compare the corner strength to all other interest points and we keep track of the minimum distance to a larger magnitude interest point) overlap in neighborhood of size 3
FAST | circle of radius 3, FAST 9-16
BRISK | same as Agast(FAST 9-16) for spatial and at different scales, FAST 5-8 on octave level 0
ORB | same as Fast for spatial and at different scales
AKAZE | same as KAZE which uses 3×3 window or spatial and at different scales
SIFT | Local extrema w 3x3 window at current scale (8 pixels) & in prev/next scales (9 pixels). Taylor series expansion of scale space to refine extrema & contrast thresholding at this location


## MP.8 Performance Evaluation 2

Number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors was added. 
In the matching step, the BF approach used with the descriptor distance ratio set to 0.8
Sum of matched keypoints across all 10 images shown below

Det-Desc-Mat-Sel                    | #Total Matches | Per Img Matches
------------ | ------------- | -------------
SHITOMASI-BRISK-MAT_BF-SEL_KNN      |  767 | [95, 88, 80, 90, 82, 79, 85, 86, 82]
SHITOMASI-BRIEF-MAT_BF-SEL_KNN      |  944 | [115, 111, 104, 101, 102, 102, 100, 109, 100]
SHITOMASI-ORB-MAT_BF-SEL_KNN        |  908 | [106, 102, 99, 102, 103, 97, 98, 104, 97]
SHITOMASI-FREAK-MAT_BF-SEL_KNN      |  768 | [86, 90, 86, 88, 86, 80, 81, 86, 85]
SHITOMASI-SIFT-MAT_BF-SEL_KNN       |  927 | [112, 109, 104, 103, 99, 101, 96, 106, 97]
HARRIS-BRISK-MAT_BF-SEL_KNN         |  365 | [40, 36, 36, 38, 40, 48, 42, 45, 40]
HARRIS-BRIEF-MAT_BF-SEL_KNN         |  424 | [48, 46, 42, 45, 45, 54, 48, 49, 47]
HARRIS-ORB-MAT_BF-SEL_KNN           |  422 | [48, 41, 43, 46, 44, 53, 49, 52, 46]
HARRIS-FREAK-MAT_BF-SEL_KNN         |  372 | [44, 39, 37, 39, 39, 51, 37, 44, 42]
HARRIS-SIFT-MAT_BF-SEL_KNN          |  412 | [45, 46, 41, 44, 43, 51, 45, 51, 46]
FAST-BRISK-MAT_BF-SEL_KNN           | 2183 | [256, 243, 241, 239, 215, 251, 248, 243, 247]
FAST-BRIEF-MAT_BF-SEL_KNN           | 2831 | [320, 332, 299, 331, 276, 327, 324, 315, 307]
FAST-ORB-MAT_BF-SEL_KNN             | 2768 | [307, 308, 298, 321, 283, 315, 323, 302, 311]
FAST-FREAK-MAT_BF-SEL_KNN           | 2233 | [251, 247, 233, 255, 231, 265, 251, 253, 247]
FAST-SIFT-MAT_BF-SEL_KNN            | 2782 | [316, 325, 297, 311, 291, 326, 315, 300, 301]
BRISK-BRISK-MAT_BF-SEL_KNN          | 1570 | [171, 176, 157, 176, 174, 188, 173, 171, 184]
BRISK-BRIEF-MAT_BF-SEL_KNN          | 1704 | [178, 205, 185, 179, 183, 195, 207, 189, 183]
BRISK-ORB-MAT_BF-SEL_KNN            | 1514 | [162, 175, 158, 167, 160, 182, 167, 171, 172]
BRISK-FREAK-MAT_BF-SEL_KNN          | 1524 | [160, 177, 155, 173, 161, 183, 169, 178, 168]
BRISK-SIFT-MAT_BF-SEL_KNN           | 1646 | [182, 193, 169, 183, 171, 195, 194, 176, 183]
ORB-BRISK-MAT_BF-SEL_KNN            |  751 | [73, 74, 79, 85, 79, 92, 90, 88, 91]
ORB-BRIEF-MAT_BF-SEL_KNN            |  545 | [49, 43, 45, 59, 53, 78, 68, 84, 66]
ORB-ORB-MAT_BF-SEL_KNN              |  763 | [67, 70, 72, 84, 91, 101, 92, 93, 93]
ORB-FREAK-MAT_BF-SEL_KNN            |  420 | [42, 36, 44, 47, 44, 51, 52, 48, 56]
ORB-SIFT-MAT_BF-SEL_KNN             |  763 | [67, 79, 78, 79, 82, 95, 95, 94, 94]
AKAZE-BRISK-MAT_BF-SEL_KNN          | 1215 | [137, 125, 129, 129, 131, 132, 142, 146, 144]
AKAZE-BRIEF-MAT_BF-SEL_KNN          | 1266 | [141, 134, 131, 130, 134, 146, 150, 148, 152]
AKAZE-ORB-MAT_BF-SEL_KNN            | 1182 | [131, 129, 127, 117, 130, 131, 137, 135, 145]
AKAZE-FREAK-MAT_BF-SEL_KNN          | 1187 | [126, 129, 127, 121, 122, 133, 144, 147, 138]
AKAZE-AKAZE-MAT_BF-SEL_KNN          | 1259 | [138, 138, 133, 127, 129, 146, 147, 151, 150]
AKAZE-SIFT-MAT_BF-SEL_KNN           | 1270 | [134, 134, 130, 136, 137, 147, 147, 154, 151]
SIFT-BRISK-MAT_BF-SEL_KNN           |  592 | [64, 66, 62, 66, 59, 64, 64, 67, 80]
SIFT-BRIEF-MAT_BF-SEL_KNN           |  702 | [86, 78, 76, 85, 69, 74, 76, 70, 88]
SIFT-ORB-MAT_BF-SEL_KNN             |  810 | [91, 84, 84, 90, 88, 84, 86, 105, 98]
SIFT-FREAK-MAT_BF-SEL_KNN           |  593 | [65, 72, 64, 66, 59, 59, 64, 65, 79]
SIFT-SIFT-MAT_BF-SEL_KNN            |  800 | [82, 81, 85, 93, 90, 81, 82, 102, 104]

## MP.9 Performance Evaluation 3

CSV log of the time it takes for keypoint detection and descriptor extraction can be found [here](./results/Performance.csv)

Pls note the detector parameters are not tuned to get best performance from each. Default parameters have been used

Based of perf data TOP3 detector / descriptor combinations for the purpose of detecting keypoints on vehicles are shown below. 

Detector | Descriptor
------------ | -------------
FAST | ORB
FAST | BRIEF
FAST | BRISK