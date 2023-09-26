# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

[//]: # (Image References)

[image1]: ./images/lidar_ttc_dist.png
[image2]: ./images/lidar_constant_velocity.png
[image3]: ./images/lp_0.png
[image4]: ./images/lp_1.png
[image5]: ./images/cluster_kps.jpg
[image6]: ./images/camera_ttc_dist.png
[image7]: ./images/camera_constant_velocity.png
[image8]: ./images/camera_bb_prob.jpg
[image9]: ./images/camera_kps.jpg
[image10]: ./images/cluster_kps_orb_orb_frm7.jpg
[image11]: ./images/cluster_kps_orb_freak_frm2.jpg
[image12]: ./images/cluster_kps_orb_sift_frm2.jpg

## FP.1 Match 3D Objects
Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

1. Find each keypoint match between current and previous frame, gather the bounding box id's containing the keypoint. Pls see function matchBoundingBoxesForKeyPt()
    ```
    matchBoundingBoxesForKeyPt(prevFrame.keypoints[match.queryIdx], prevFrame.boundingBoxes, prevBBoxIds);
    matchBoundingBoxesForKeyPt(currFrame.keypoints[match.trainIdx], currFrame.boundingBoxes, curBBoxIds);
    ```
2.  Keep track on count of combinations of current and previous frame bounding box ids
    ```
	for (const auto pBB : prevBBoxIds)
	{
		for (const auto cBB : curBBoxIds)
		{
			bbMatches[pBB][cBB]++;
		}
	}
    ```
3.  Select the combination of previous & current frame bounding box ids with maximum count
    ```
	int maxValIdx;
	for (const auto pBB : prevFrame.boundingBoxes)
	{
		maxValIdx = std::max_element(bbMatches[pBB.boxID].begin(), bbMatches[pBB.boxID].end()) - bbMatches[pBB.boxID].begin();
		bbBestMatches.insert(std::make_pair(pBB.boxID, maxValIdx));
	}
    ```

## FP.2 Compute Lidar-based TTC
Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

1. Pls refer to below on ttc estimation via constant-velocity model from course notes
![alt text][image1] and ![alt text][image2]

Instead of selecting the closest points after outlier removal, median points are selected based of x value. In figures below, the filled circle is the median points
![alt text][image3]
![alt text][image4]
 

## FP.3 Associate Keypoint Correspondences with Bounding Boxes
Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

For all matching keypoints in bounding box, nearest neighbor clustering is performed. The largest cluster keypoints are associated with the bounding box. 
Other mechanisms of using all the keypoints till the median point and using all keypoints in bounding box were experimented with. Nearest neighbor results proved to be the best amongst the 3 methods

![alt text][image5]

## FP.4 Compute Camera-based TTC
Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

The relations between the distance of the vehicle d to the heights of the preceding vehicle H, h and focal length of the pinhole camera f is shown in image below

![alt text][image6]

For the constant velocity motion model, above geometrical relation is exploited to compute ttc as shown in equations in fig below

![alt text][image7]

TTC can be estimated by observing changes in relative height directly in the image.
However using bounding box height or width for TTC computation leads to significant estimation errors, as the bounding boxes do not always reflect the true vehicle dimensions and the aspect ratio differs between images

![alt text][image8]

Instead with uniquely identifiable keypoints that could be tracked from one frame to the next, the distance between all keypoints on the vehicle relative to each other can be used to compute a robust estimate of the height ratio in the TTC equation

![alt text][image9]
In (a), a set of keypoints has been detected and the relative distances between keypoints 1-7 have been computed. 

In (b), 4 keypoints have been matched between successive images (with keypoint 3 being a mismatch) using a higher-dimensional similarity measure called descriptor. 
The ratio of all relative distances between each other can be used to compute a reliable TTC estimate by replacing the height ratio h_1 / h_0
with the mean or median of all distance ratios d_k/d_k'

Median distance ration was used

## FP.5 Performance Evaluation 1
Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

Using median, ttc values seemed resonable. However on using minimum value even with outlier removal, ttc values vary significantly

FrameNum | TTC_Lidar | TTC_Camera | TTC_Diff
-------- | --------- | ---------- |---------
1 | 12.5156 | 13.5353 | 1.01965
2 | 12.6142 | 13.3068 | 0.692557
3 | 14.091 | 12.1752 | -1.91579
4 | 16.6894 | 12.1046 | -4.58479
5 | 15.7465 | 12.5715 | -3.17501
6 | 12.7835 | 14.5902 | 1.80666
7 | 11.9844 | 12.4165 | 0.43218
8 | 13.1241 | 13.2599 | 0.135778
9 | 13.0241 | 11.208 | -1.81614
10 | 11.1746 | 13.9459 | 2.77127
11 | 12.8086 | 11.2431 | -1.56548
12 | 8.95978 | 11.7391 | 2.77927
13 | 9.96439 | 11.26 | 1.29564
14 | 9.59863 | 11.801 | 2.20235
15 | 8.52157 | 9.51866 | 0.997098
16 | 9.51552 | 12.0398 | 2.52427
17 | 9.61241 | 11.1382 | 1.52576
18 | 8.3988 | 9.33977 | 0.940967

## FP.6 Performance Evaluation 2
Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

#### Average TTC in seconds:
**Descriptor_Match_Sel** | **SHITOMASI** | **HARRIS** | **FAST** | **BRISK** | **ORB** | **AKAZE** | **SIFT**
-------- | --------- | ------ | ---- | ----- | --- | ----- | -----
BRISK_BF_NN | 12.06630722 | 14.00035611 | 12.03034944 | 14.01688833 | 17.56457294 | 12.33823667 | 11.78609778
BRISK_BF_KNN | 12.17373278 | 13.30347222 | 12.14517222 | 14.69747278 | 20.51923765 | 12.41237778 | 12.16121333
BRISK_FLANN_NN | 11.99708222 | 13.76188 | 12.08190556 | 14.49003833 | 18.31238118 | 12.23824722 | 11.81776667
BRISK_FLANN_KNN | 12.11105778 | 13.94284722 | 12.27182222 | 14.33271889 | 18.80996706 | 12.51483444 | 12.37280278
BRIEF_BF_NN | 12.12162167 | 12.61435111 | 12.13256111 | 15.06838111 | 18.50514222 | 12.52324056 | 12.07303056
BRIEF_BF_KNN | 12.31758667 | 13.27809111 | 12.231155 | 14.35088333 | 33.65796333 | 12.33496667 | 12.24849111
BRIEF_FLANN_NN | 12.28928056 | 12.70366944 | 12.04731444 | 13.96770222 | 38.15689278 | 12.49095444 | 12.38055833
BRIEF_FLANN_KNN | 12.17010222 | 14.01622222 | 12.17330167 | 13.20674611 | 41.51209778 | 12.32478667 | 12.51800111
ORB_BF_NN | 11.81624333 | 13.04028944 | **11.719425** | 13.70898667 | 19.72410588 | 12.437415 | 11.77854556
ORB_BF_KNN | 11.73511944 | 13.98896722 | 11.77398333 | 14.2947 | **1164.945498** | 12.33061444 | 11.76114333
ORB_FLANN_NN | 11.85340278 | 15.85413556 | 11.82756889 | 14.29717333 | 18.99565556 | 12.29548333 | 11.83205278
ORB_FLANN_KNN | 11.98675611 | 16.57724722 | 12.15541111 | 14.24904833 | 18.68861846 | 12.31990444 | 11.831555
FREAK_BF_NN | 11.94539444 | 13.03802278 | 12.13906167 | 14.54565722 | **11.35724941** | 12.19365944 | 11.57152056
FREAK_BF_KNN | 12.04863722 | 12.627 | 12.01992222 | 14.37355389 | 26.93437118 | 12.34037222 | 12.11525667
FREAK_FLANN_NN | 11.93022667 | 12.42061167 | 11.888905 | 14.43552944 | 46.365455 | 12.46262222 | **11.47099167** 
FREAK_FLANN_KNN | 11.74744611 | 13.745375 | 11.90828333 | 14.32780389 | 12.84806067 | 12.58875278 | 12.76545889
SIFT_BF_NN | 11.90632611 | 13.229045 | 12.20024333 | 15.28112222 | 33.38900412 | 12.46082222 | 11.81421
SIFT_BF_KNN | 12.09119944 | 12.85517778 | 12.06100556 | 14.63280389 | **87.20578118** | 12.40734333 | 11.77790333
SIFT_FLANN_NN | 11.94467389 | 12.58298944 | 12.20658667 | 15.75711722 | 31.53337765 | 12.46188111 | 11.79988889
SIFT_FLANN_KNN | 12.15799389 | 12.6237 | 12.01847167 | 15.29769444 | **83.53146353** | 12.42460667 | 11.73326056
AKAZE_BF_NN |  |  |  |  |  | 12.37848667 | 
AKAZE_BF_KNN |  |  |  |  |  | 12.32669944 | 
AKAZE_FLANN_NN |  |  |  |  |  | 12.29760667 | 
AKAZE_FLANN_KNN |  |  |  |  |  | 12.21880556 | 


1. Top 3 Detector/Descriptor/Matcher/Selector combinations:
    * ORB + FREAK + BF + NN
    * SIFT + FREAK + FLANN + NN
    * FAST + ORB + BF + NN
    
2. Top 3 Detectors across combinations of Descriptor/Matcher/Selector
    * SIFT
    * SHITOMASI
    * FAST
   
3. Bottom 3 Detector/Descriptor/Matcher/Selector combinations:
    * ORB + ORB + BF + KNN
    * ORB + SIFT + BF + KNN
    * ORB + SIFT + FLANN + KNN
    
    
#### Standard Deviation TTC in seconds:
**Descriptor_Match_Sel** | **SHITOMASI** | **HARRIS** | **FAST** | **BRISK** | **ORB** | **AKAZE** | **SIFT**
-------- | --------- | ------ | ---- | ----- | --- | ----- | -----
BRISK_BF_NN | 1.381475145 | 4.439062642 | 1.125669249 | 4.42372815 | 13.73789921 | 2.18389632 | 2.584145571
BRISK_BF_KNN | 1.357998277 | 3.063026253 | **0.866268661** | 4.707500781 | 25.56557613 | 2.267303277 | 2.7380311
BRISK_FLANN_NN | 1.545483703 | 4.26964174 | 1.075262461 | 4.492841576 | 9.943709668 | 2.273076044 | 2.573978426
BRISK_FLANN_KNN | 1.264352664 | 6.398575585 | **0.944886466** | 3.542128125 | 19.90816971 | 2.3392172 | 2.983670235
BRIEF_BF_NN | 2.006861345 | 1.927407424 | 1.179087509 | 4.674724957 | 8.49499961 | 2.236343508 | 2.941279464
BRIEF_BF_KNN | 1.758402881 | 2.842922064 | 1.336579388 | 3.307545576 | 37.03686876 | 2.206910958 | 3.034103872
BRIEF_FLANN_NN | 1.49817026 | 1.992407079 | 1.502274628 | 3.290617215 | 80.02286957 | 2.363184753 | 3.132132056
BRIEF_FLANN_KNN | 1.387917801 | 4.313541376 | 1.298149858 | 2.028577521 | 77.34831657 | 2.23666534 | 3.451201999
ORB_BF_NN | 1.278650433 | 2.820158216 | 1.370377668 | 3.282131273 | 16.33094234 | 2.21122575 | 2.641125637
ORB_BF_KNN | 1.406532164 | 5.235260088 | 1.075811535 | 3.193606524 | 4727.009247 | 2.147814452 | 2.46704793
ORB_FLANN_NN | 1.396296595 | 7.436134165 | 1.275908852 | 3.550695751 | 8.901781021 | 1.970754778 | 2.477952812
ORB_FLANN_KNN | 1.489110279 | 10.772572 | 1.290029105 | 3.843270057 | 9.939839947 | 2.069661277 | 2.434237502
FREAK_BF_NN | 1.175698637 | 3.481478236 | 1.232427575 | 4.169887176 | 172.0973716 | 2.101725252 | 2.502354419
FREAK_BF_KNN | 1.414553046 | 2.540373967 | **0.911583892** | 3.993557497 | **229.8563703** | 2.221390341 | 2.711186174
FREAK_FLANN_NN | 1.445315396 | 1.330716055 | 1.008904692 | 4.06682604 | 115.7208473 | 2.144491444 | 2.235807373
FREAK_FLANN_KNN | 1.625613521 | 6.75917678 | 0.9764919 | 4.782033185 | 8.220927085 | 2.397878446 | 3.038289614
SIFT_BF_NN | 1.579403745 | 3.518092729 | 1.611930796 | 5.096776698 | 67.78436026 | 2.243113955 | 2.562991794
SIFT_BF_KNN | 1.111451061 | 2.797129662 | 1.292385276 | 4.229909802 | **238.0871445** | 2.254884582 | 2.404321306
SIFT_FLANN_NN | 1.411730185 | 2.31196881 | 1.730585288 | 7.599504007 | 67.85990455 | 2.287587378 | 2.501882293
SIFT_FLANN_KNN | 1.100721092 | 1.9625345 | 1.299873435 | 7.521192348 | **238.2887571** | 2.2541946 | 2.436703395
AKAZE_BF_NN |  |  |  |  |  | 2.199156838 | 
AKAZE_BF_KNN |  |  |  |  |  | 2.185084297 | 
AKAZE_FLANN_NN |  |  |  |  |  | 2.06117048 | 
AKAZE_FLANN_KNN |  |  |  |  |  | 1.946231047 | 


#### Examples where camera-based TTC estimation is way off:

1. 3 Detector/Descriptor/Matcher/Selector examples:
    * ORB + ORB + BF + KNN
    ![alt text][image10]
    * ORB + SIFT + BF + KNN
    ![alt text][image11]
    * ORB + FREAK + FLANN + KNN
    ![alt text][image12]

Couple of reasons for very high or invalid ttc
* Insufficient keypoint matches in the frames resulting in empty distRatios list 
* Median distance ratio is very close to 1, again resulting in very high ttc