#include <iostream>
#include <algorithm>
#include <numeric>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include "kdTree.h"

using namespace std;
#define KP_CLUSTERING 1


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto &lidarPoint : lidarPoints)
    {
		// assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = lidarPoint.x;
        X.at<double>(1, 0) = lidarPoint.y;
        X.at<double>(2, 0) = lidarPoint.z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = static_cast<int>(Y.at<double>(0, 0) / Y.at<double>(2, 0)); // pixel coordinates
        pt.y = static_cast<int>(Y.at<double>(1, 0) / Y.at<double>(2, 0));

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (auto bBox = boundingBoxes.begin(); bBox != boundingBoxes.end(); ++bBox)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = static_cast<int>(bBox->roi.x + shrinkFactor * bBox->roi.width / 2.0);
            smallerBox.y = static_cast<int>(bBox->roi.y + shrinkFactor * bBox->roi.height / 2.0);
            smallerBox.width = static_cast<int>(bBox->roi.width * (1 - shrinkFactor));
            smallerBox.height = static_cast<int>(bBox->roi.height * (1 - shrinkFactor));

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(bBox);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (1 == enclosingBoxes.size())
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(lidarPoint);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));

	// display image
	for(const auto& bBox : boundingBoxes)
    {
        // create randomized color for current 3D object
        cv::RNG rng(bBox.boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));

        // plot Lidar points into top view image
		int top=1e8, left=1e8, bottom=0, right=0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (const auto & lidarPoint : bBox.lidarPoints)
        {
            // world coordinates
            float xw = lidarPoint.x; // world position in m with x facing forward from sensor
            float yw = lidarPoint.y; // world position in m with y facing left from sensor

            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
			int x = static_cast<int>((-yw * imageSize.width / worldSize.width) + imageSize.width /2.0f);
			int y = static_cast<int>((-xw * imageSize.height / worldSize.height) + imageSize.height);

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(int(left), int(top)), cv::Point(int(right), int(bottom)),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", bBox.boxID, (int)bBox.lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 1, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 1, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = int(floor(worldSize.height / lineSpacing));
    for (int i = 0; i < nMarkers; ++i)
    {
        int y = int((-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height);
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


template<typename PointT>
void euclideanCluster(int i, typename std::vector<PointT> &dataPoints, std::vector<bool> &processed, KdTree<PointT>* tree, float distanceTol[4], std::vector<int> &cluster)
{
	processed[i] = true;
	cluster.push_back(i);

	std::vector<int> nearby = tree->search(dataPoints[i], distanceTol);

	for (int id : nearby)
	{
		if (!processed[id])
			euclideanCluster(id, dataPoints, processed, tree, distanceTol, cluster);
	}
}

template<typename PointT>
std::vector<std::vector<int>> euclideanCluster(typename std::vector<PointT> &dataPoints, KdTree<PointT>* tree, float distanceTol[4])
{

	// TODO: Fill out this function to return list of indices for each cluster
	std::vector<std::vector<int>> clusters;
	std::vector<bool> processed(dataPoints.size(), false);
	for (int i = 0; i < dataPoints.size(); ++i)
	{
		if (processed[i])
			continue;

		std::vector<int> cluster;
		euclideanCluster(i, dataPoints, processed, tree, distanceTol, cluster);
		if (cluster.size() > 0)
			clusters.push_back(cluster);
	}

	return clusters;
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
	// ...
	struct data_pt
	{
		float dist;
		cv::KeyPoint kpt;
		cv::DMatch kptm;
	};

	struct compare {
		bool operator()(struct data_pt lhs, struct data_pt rhs) const {
			return lhs.dist < rhs.dist;
		}
	};

	struct data_pt datapt;
	std::multiset<struct data_pt, compare> data;

	for (const auto& it : kptMatches)
	{ // outer kpt. loop

		// get current keypoint and its matched partner in the prev. frame
		const auto &kpCur = kptsCurr.at(it.trainIdx);
		if (boundingBox.roi.contains(kpCur.pt))
		{
			const auto &kpPrev = kptsPrev.at(it.queryIdx);
			datapt.dist = cv::norm(kpPrev.pt - kpCur.pt);
			datapt.kpt = kpCur;
			datapt.kptm = it;
			data.insert(datapt);
		}
	}

#if KP_CLUSTERING==0 //MEDIAN_CLUSTERING
	int medIndex = int(floor(data.size() / 2.0));
	auto medDist = std::next(data.begin(), medIndex);
	std::multiset<struct data_pt, compare>::iterator maxDistPt = data.upper_bound(*medDist);

	for (auto it = data.begin(); it != maxDistPt; ++it)
	{
		boundingBox.keypoints.push_back(it->kpt);
		boundingBox.kptMatches.push_back(it->kptm);
	}
#elif KP_CLUSTERING==1 //NN_CLUSTERING
	union data2
	{
		float dist;
		float data[1];
	};
	std::vector<data2> distances;

	KdTree<data2> *pTree = new KdTree<data2>(1);
	float distTol[2] = { 3.0, 3.0 };
	for (auto &it : data)
	{
		data2 dist;
		dist.dist = it.dist;
		distances.push_back(dist);
	}

	pTree->setInputData(distances);
	std::vector<std::vector<int>> clusterIndices = euclideanCluster(distances, pTree, distTol);

	vector<size_t> sizes(clusterIndices.size());
	std::transform(clusterIndices.begin(), clusterIndices.end(), sizes.begin(), mem_fun_ref(&vector<int>::size));
	vector<size_t>::iterator max_iter = max_element(sizes.begin(), sizes.end());
	int loc = max_iter - sizes.begin();

	for (auto it : clusterIndices[loc])
	{
		datapt = *std::next(data.begin(), it);
		boundingBox.keypoints.push_back(datapt.kpt);
		boundingBox.kptMatches.push_back(datapt.kptm);
	}
#else
	for (auto &match : kptMatches) {
	    if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
	        boundingBox.kptMatches.push_back(match);
	    }
	}	
#endif	
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
	// compute distance ratios between all matched keypoints
	vector<float> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
	for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
	{ // outer kpt. loop

		// get current keypoint and its matched partner in the prev. frame
		cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
		cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

		for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
		{ // inner kpt.-loop

			double minDist = 100.0; // min. required distance

			// get next keypoint and its matched partner in the prev. frame
			cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
			cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

			// compute distances and distance ratios
			double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
			double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

			if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
			{ // avoid division by zero

				float distRatio = distCurr / distPrev;
				distRatios.push_back(distRatio);
			}
		} // eof inner loop over all matched kpts
	}     // eof outer loop over all matched kpts

	// only continue if list of distance ratios is not empty
	if (distRatios.size() == 0)
	{
		TTC = NAN;
		return;
	}


	// STUDENT TASK (replacement for meanDistRatio)
	std::sort(distRatios.begin(), distRatios.end());
	int medIndex = int(floor(distRatios.size() / 2.0));
	double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

	double dT = 1 / frameRate;

	TTC = -dT / (1 - medDistRatio);
}

std::vector<LidarPoint> removeOutliers(std::vector<LidarPoint> &lidarPoints)
{
	float distTol[4] = { 0.09, 0.09, 0.09, 0.09 };
	int nDim = 3;
	KdTree<LidarPoint> *pTree = new KdTree<LidarPoint>(nDim);
	pTree->setInputData(lidarPoints);
	std::vector<std::vector<int>> clusterIndices = euclideanCluster(lidarPoints, pTree, distTol);

	std::vector<LidarPoint> lidarPointsCluster;
	for (auto cluster : clusterIndices)
	{
		if (cluster.size() > 5)
		{
			for (auto id : cluster)
			{
				lidarPointsCluster.push_back(lidarPoints[id]);
			}
		}
	}
	return lidarPointsCluster;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr,
                     double frameRate, double &TTC)
{
#ifdef REMOVE_OUTLIERS
	std::vector<LidarPoint> lidarPointsPrev_n = removeOutliers(lidarPointsPrev);
	std::vector<LidarPoint> lidarPointsCurr_n = removeOutliers(lidarPointsCurr);
#else
	std::vector<LidarPoint> lidarPointsPrev_n = lidarPointsPrev;
	std::vector<LidarPoint> lidarPointsCurr_n = lidarPointsCurr;
#endif
	auto cmpFunc = [](const LidarPoint& lp1, const LidarPoint& lp2) { return (lp1.x == lp2.x ? lp1.z < lp2.z : lp1.x < lp2.x); };
	std::sort(lidarPointsPrev_n.begin(), lidarPointsPrev_n.end(), cmpFunc);
	std::sort(lidarPointsCurr_n.begin(), lidarPointsCurr_n.end(), cmpFunc);

	double prevX, currX;
#ifdef MEAN_MIN_VALUE
	prevX = lidarPointsPrev_n.begin()->x;
	currX = lidarPointsCurr_n.begin()->x;
	//auto sumOp = [](const double sum, const LidarPoint& lp) { return sum + lp.x; };
	//std::vector<LidarPoint>::iterator endIt = lidarPointsPrev_n.begin() + cv::min((size_t)6, lidarPointsPrev_n.size());
	//int numElem = endIt - lidarPointsPrev_n.begin();
	//prevX = std::accumulate(lidarPointsPrev_n.begin(), endIt, 0.0, sumOp) / numElem;
	//endIt = lidarPointsCurr_n.begin() + +cv::min((size_t)6, lidarPointsCurr_n.size());
	//numElem = endIt - lidarPointsCurr_n.begin();
	//currX = std::accumulate(lidarPointsCurr_n.begin(), endIt, 0.0, sumOp) / numElem;
#else
	int medIndex = int(floor(lidarPointsPrev_n.size() / 2.0));
	prevX = lidarPointsPrev_n.size() % 2 == 0 ? (lidarPointsPrev_n[medIndex - 1].x + lidarPointsPrev_n[medIndex].x) / 2.0 : lidarPointsPrev_n[medIndex].x; // compute median dist. ratio to remove outlier influence
	medIndex = int(floor(lidarPointsCurr_n.size() / 2.0));
	currX = lidarPointsCurr_n.size() % 2 == 0 ? (lidarPointsCurr_n[medIndex - 1].x + lidarPointsCurr_n[medIndex].x) / 2.0 : lidarPointsCurr_n[medIndex].x; // compute median dist. ratio to remove outlier influence
#endif

	double dT = 1 / frameRate; // time between two measurements in seconds
	// compute TTC from both measurements
	TTC = currX * dT / (prevX - currX);
}

void matchBoundingBoxesForKeyPt(const cv::KeyPoint &kpt, const std::vector<BoundingBox> &boundingBoxes, std::vector<int> &matchBBoxIds)
{
	for (int i = 0; i < boundingBoxes.size(); ++i) 
	{
		if (boundingBoxes[i].roi.contains(kpt.pt)) 
		{
			matchBBoxIds.push_back(boundingBoxes[i].boxID);
		}
	}
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
	std::vector<std::vector<int> > bbMatches(prevFrame.boundingBoxes.size(), std::vector<int>(currFrame.boundingBoxes.size(), 0));

	std::vector<int> prevBBoxIds;
	std::vector<int> curBBoxIds;
	for (const auto& match : matches)
	{
		matchBoundingBoxesForKeyPt(prevFrame.keypoints[match.queryIdx], prevFrame.boundingBoxes, prevBBoxIds);
		matchBoundingBoxesForKeyPt(currFrame.keypoints[match.trainIdx], currFrame.boundingBoxes, curBBoxIds);

		for (const auto pBB : prevBBoxIds)
		{
			for (const auto cBB : curBBoxIds)
			{
				bbMatches[pBB][cBB]++;
			}
		}
		prevBBoxIds.clear();
		curBBoxIds.clear();
	}

	int maxValIdx;
	for (const auto pBB : prevFrame.boundingBoxes)
	{
		maxValIdx = std::max_element(bbMatches[pBB.boxID].begin(), bbMatches[pBB.boxID].end()) - bbMatches[pBB.boxID].begin();
		bbBestMatches.insert(std::make_pair(pBB.boxID, maxValIdx));
	}
}
