#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <deque>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

template<typename Data>
class RingBuffer {
	std::deque<Data> m_dataArr;
	size_t m_maxSize;
	size_t m_numel;

public:
	//constructor
	RingBuffer(size_t size) {
		m_maxSize = size;
		m_numel = 0;
	};

	//deconstructor
	~RingBuffer() { m_dataArr.clear(); };

	void push_back(Data data) {
		if (m_numel == m_maxSize) {
			m_dataArr.pop_front();
		}
		else
			++m_numel;
		m_dataArr.push_back(data);
	};

	typename std::deque<Data>::iterator end() {
		return m_dataArr.end();
	}

	typename std::deque<Data>::iterator begin() {
		return m_dataArr.begin();
	}

	std::size_t size() {
		return m_numel;
	}
};

typedef enum
{
	Detector=0,
	Descriptor,
	Matcher,
	Selector
}featTypes;

#endif /* dataStructures_h */