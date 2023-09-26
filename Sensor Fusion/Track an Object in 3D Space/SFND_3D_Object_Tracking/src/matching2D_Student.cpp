
#include <numeric>
#include "matching2D.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
double matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
		//f ORB is using WTA_K == 3 or 4, cv2.NORM_HAMMING2 should be used. TBD
        int normType = (descriptorType.compare("SIFT") == 0)? cv::NORM_L2: cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
		if (descSource.type() != CV_32F) {
			descSource.convertTo(descSource, CV_32F);
		}

		if (descRef.type() != CV_32F) {
			descRef.convertTo(descRef, CV_32F);
		}
    }
	else
	{
		cout << matcherType << " is not supported" << endl;
		return 0;
	}

    // perform matching task
	double t = (double)cv::getTickCount();
	if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
		std::vector<std::vector<cv::DMatch>> knn_matches;
		matcher->knnMatch(descSource, descRef, knn_matches, 2);

		const float dist_ratio_thresh = 0.8f;
		for (size_t i = 0; i < knn_matches.size(); ++i) {
			if (knn_matches[i][0].distance < (knn_matches[i][1].distance * dist_ratio_thresh))
				matches.push_back(knn_matches[i][0]);
		}
	}
	else
	{
		cout << selectorType << " is not supported" << endl;
	}
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	t = 1000 * t / 1.0;
	return t;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        extractor = cv::BRISK::create();
    }
	else if (descriptorType.compare("BRIEF") == 0)
	{
		extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
	}
	else if (descriptorType.compare("ORB") == 0)
	{
		extractor = cv::ORB::create();
	}
	else if (descriptorType.compare("FREAK") == 0)
	{
		extractor = cv::xfeatures2d::FREAK::create();
	}
	else if (descriptorType.compare("AKAZE") == 0)
	{
		extractor = cv::AKAZE::create();
	}
	else if (descriptorType.compare("SIFT") == 0)
	{
		extractor = cv::xfeatures2d::SIFT::create();
	}
	else
	{
		cout << descriptorType << " is not supported" << endl;
		return 0;
	}

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	t = 1000 * t / 1.0;
	return t;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(cv::Mat &img, vector<cv::KeyPoint> &keypoints, bool useHarrisDetector, bool bVis)
{
	// compute detector parameters based on image size
	int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
	double maxOverlap = 0.0; // max. permissible overlap between two features in %
	double minDistance = (1.0 - maxOverlap) * blockSize;
	int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

	double qualityLevel = 0.01; // minimal accepted quality of image corners
	double k = 0.04;

	// Apply corner detection
	double t = (double)cv::getTickCount();
	vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

	// add corners to result vector
	for (auto it = corners.begin(); it != corners.end(); ++it)
	{

		cv::KeyPoint newKeyPoint;
		newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
		newKeyPoint.size = blockSize;
		keypoints.push_back(newKeyPoint);
	}
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	t = 1000 * t / 1.0;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
	return t;
}

double detKeypointsHarris(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, bool bVis = false)
{
	// Detector parameters
	int blockSize = 4;     // for every pixel, a blockSize × blockSize neighborhood is considered
	int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
	double qualityLevel = 0.01; // minimal accepted quality of image corners
	int minResponse = 0; // minimum value for a corner in the 8bit scaled response matrix
	double k = 0.04;       // Harris parameter (see equation for details)

	double t = (double)cv::getTickCount();

	// Detect Harris corners and normalize output
	cv::Mat dst, dstNorm, dstNormScaled;
	dst = cv::Mat::zeros(img.size(), CV_32FC1);
	cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
	double maxVal = 0;
	minMaxLoc(dst, 0, &maxVal, 0, 0);
	// Use quality based thresholding instead of fixed minResponse
	cv::threshold(dst, dst, maxVal*qualityLevel, 0, cv::THRESH_TOZERO);
	cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	cv::convertScaleAbs(dstNorm, dstNormScaled);


	// Look for prominent corners and instantiate keypoints
	// Adaptive Non-Maximal Suppression
	double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
	for (size_t j = 0; j < dstNorm.rows; j++)
	{
		for (size_t i = 0; i < dstNorm.cols; i++)
		{
			int response = (int)dstNorm.at<float>(j, i);
			if (response > minResponse)
			{ 
				cv::KeyPoint newKeyPoint;
				newKeyPoint.pt = cv::Point2f(i, j);
				newKeyPoint.size = 2 * apertureSize;
				newKeyPoint.response = response;

				// Non-maximum suppression (NMS) in local neighbourhood around new key point
				bool bOverlap = false;
				for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
				{
					double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
					if (kptOverlap > maxOverlap)
					{
						bOverlap = true;
						if (newKeyPoint.response > (*it).response)
						{
							// if response is higher for new kpt, replace old key
							*it = newKeyPoint;
							break;
						}
					}
				}
				if (!bOverlap)
					keypoints.push_back(newKeyPoint);
			}
		}
	}

	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	t = 1000 * t / 1.0;

	// visualize keypoints
	if (bVis)
	{
		cv::Mat visImage = dstNormScaled.clone();
		string windowName = "Harris Corner Detector Response Matrix";
		cv::namedWindow(windowName, 6);
		cv::imshow(windowName, visImage);
		cv::waitKey(0);

		windowName = "Harris Corner Detector Results";
		cv::namedWindow(windowName, 6);
		cv::drawKeypoints(dstNormScaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imshow(windowName, visImage);
		cv::waitKey(0);
	}

	return t;
}

static inline void
unpackOctave(cv::KeyPoint& kpt)
{
	int octave = kpt.octave & 255;
	octave = octave < 128 ? octave : (-128 | octave);
	octave = max(octave, 0);
	kpt.octave = octave;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double detectKeypoints(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, std::string detectorType, bool unpackSiftOctave, bool bVis)
{
	cv::Ptr<cv::Feature2D> detector;
	if (detectorType.compare("SHITOMASI") == 0)
	{
		return detKeypointsShiTomasi(img, keypoints, false, bVis);
	}
	else if (detectorType.compare("HARRIS") == 0)
	{
		//Can also use goodFeaturesToTrack with useHarrisDetector=True or GFTTDetector (equivalent). NMS algo used is different
		return detKeypointsHarris(img, keypoints, bVis);
	}
	else if (detectorType.compare("FAST") == 0)
	{
		detector = cv::FastFeatureDetector::create();
	}
	else if (detectorType.compare("BRISK") == 0)
	{
		detector = cv::BRISK::create();
	}
	else if (detectorType.compare("ORB") == 0)
	{
		detector = cv::ORB::create();
	}
	else if (detectorType.compare("AKAZE") == 0)
	{
		detector = cv::AKAZE::create();
	}
	else if (detectorType.compare("SIFT") == 0)
	{
		detector = cv::xfeatures2d::SIFT::create();
	}
	else
	{
		cout << detectorType << " is not supported" << endl;
		return 0;
	}

	// perform feature description
	double t = (double)cv::getTickCount();
	detector->detect(img, keypoints);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	t = 1000 * t / 1.0;

	if (unpackSiftOctave)
	{
		//Seems like an interoperability issue in implementation. Octave level needs to be decoded to use with other descriptors that use octave
		for (int i = 0; i < keypoints.size(); i++)
			unpackOctave(keypoints[i]);
	}
	return t;
}
