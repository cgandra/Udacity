// PCL lib Functions for processing point clouds 


#include <unordered_set>
#include "processPointClouds.h"
#include <pcl/segmentation/sac_segmentation.h>

//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: Fill in the function to do voxel grid point reduction and region based filtering
	// Create the filtering object
	typename pcl::PointCloud<PointT>::Ptr cloudFiltered(new pcl::PointCloud<PointT>);
	pcl::VoxelGrid<PointT> vg;
	vg.setInputCloud(cloud);
	vg.setLeafSize(filterRes, filterRes, filterRes);
	vg.filter(*cloudFiltered);

	typename pcl::PointCloud<PointT>::Ptr cloudRoi(new pcl::PointCloud<PointT>);
	pcl::CropBox<PointT> roi(true);
	roi.setMin(minPoint);
	roi.setMax(maxPoint);
	roi.setInputCloud(cloudFiltered);
	roi.filter(*cloudRoi);

	std::vector<int> indices;
	pcl::CropBox<PointT> roof(true);
	roi.setMin(Eigen::Vector4f(-1.5, -1.7, -1, 1));
	roi.setMax(Eigen::Vector4f(2.6, 1.7, -0.4, 1));
	roi.setInputCloud(cloudRoi);
	roi.filter(indices);


	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	for (int point : indices)
		inliers->indices.push_back(point);

	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud(cloudRoi);
	extract.setIndices(inliers);
	extract.setNegative(true);
	extract.filter(*cloudRoi);


	auto endTime = std::chrono::steady_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

	return cloudRoi;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud)
{
	// TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
	// Create the filtering object
	pcl::ExtractIndices<PointT> extract;

	pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>()), obstCloud(new pcl::PointCloud<PointT>());

	for (int index : inliers->indices)
		planeCloud->points.push_back(cloud->points[index]);

	extract.setInputCloud(cloud);
	extract.setIndices(inliers);
	extract.setNegative(true);
	extract.filter(*obstCloud);
	std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstCloud, planeCloud);

	return segResult;
}

template<typename PointT>
void ProcessPointClouds<PointT>::RansacPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol, pcl::PointIndices &inliers)
{
	std::unordered_set<int> inliersResult;
	std::unordered_set<int> inliersTmp;
	std::srand(125676);
	//std::srand(static_cast<unsigned int> (time(NULL)));

	// TODO: Fill in this function

	// For max iterations 
	PointT pts[3];
	float A, B, C, D, dist, denom;
	float bestDist = distanceTol * cloud->points.size();
	float sumDist = 0;
	for (int idx = 0; idx < maxIterations; idx++)
	{
		// Randomly sample subset and fit line
		inliersTmp.clear();
		while (inliersTmp.size() < 3)
			inliersTmp.insert(rand() % cloud->points.size());

		auto itr = inliersTmp.begin();
		for (int i = 0; i < 3; i++)
		{
			pts[i].x = cloud->points[*itr].x;
			pts[i].y = cloud->points[*itr].y;
			pts[i].z = cloud->points[*itr].z;
			itr++;
		}

		for (int i = 1; i < 3; i++)
		{
			pts[i].x -= pts[0].x;
			pts[i].y -= pts[0].y;
			pts[i].z -= pts[0].z;
		}

		// Calculate Normal
		A = pts[1].y*pts[2].z - pts[1].z*pts[2].y;
		B = pts[1].z*pts[2].x - pts[1].x*pts[2].z;
		C = pts[1].x*pts[2].y - pts[1].y*pts[2].x;
		denom = (A * A + B * B + C * C);

		if (denom < EPS)
		{
			std::cout << "Invalid Plane " << idx << ": " << pts[0].x << ", " << pts[0].y << ", " << pts[0].z << ", " << pts[1].x << ", " << pts[1].y << ", " << pts[1].z << ", " << pts[2].x << ", " << pts[2].y << ", " << pts[2].z << std::endl;
			continue;
		}

		denom = sqrt(denom);
		A = A / denom;
		B = B / denom;
		C = C / denom;
		D = -(A*pts[0].x + B * pts[0].y + C * pts[0].z);

		// Measure distance between every point and fitted line
		sumDist = 0;
		for (int cidx = 0; cidx < cloud->points.size(); cidx++)
		{
			if (inliersTmp.count(cidx) > 0)
				continue;

			dist = fabs(A * cloud->points[cidx].x + B * cloud->points[cidx].y + C * cloud->points[cidx].z + D);
			// If distance is smaller than threshold count it as inlier
			if (dist < distanceTol)
			{
				inliersTmp.insert(cidx);
			}
			else
			{
				dist = distanceTol;
			}
			sumDist += dist;			
		}

		if ((sumDist < bestDist) && (inliersTmp.size() > inliersResult.size()))
		{
			inliersResult = inliersTmp;
		}
	}

	// Return indicies of inliers from fitted line with most inliers
	inliers.indices.clear();
	inliers.indices.insert(inliers.indices.end(), inliersResult.begin(), inliersResult.end());
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
	pcl::PointIndices::Ptr inliers{ new pcl::PointIndices };
    // TODO:: Fill in this function to find inliers for the cloud.
#ifdef PCL_SEGMENT
	pcl::SACSegmentation<PointT> seg;
	pcl::ModelCoefficients::Ptr coefficients{ new pcl::ModelCoefficients() };
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(maxIterations);
	seg.setDistanceThreshold(distanceThreshold);
	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);
#else
	RansacPlane(cloud, maxIterations, distanceThreshold, *inliers);
#endif

	if (inliers->indices.size() == 0)
	{
		std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
	}

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);
    return segResult;
}

template<typename PointT>
void ProcessPointClouds<PointT>::euclideanCluster(int i, typename pcl::PointCloud<PointT>::Ptr cloud, std::vector<bool> &processed, KdTree<PointT>* tree, float distanceTol, pcl::PointIndices &cluster)
{
	processed[i] = true;
	cluster.indices.push_back(i);

	std::vector<int> nearby = tree->search(cloud->points[i], distanceTol);

	for (int id : nearby)
	{
		if (!processed[id])
			euclideanCluster(id, cloud, processed, tree, distanceTol, cluster);
	}
}

template<typename PointT>
std::vector<pcl::PointIndices> ProcessPointClouds<PointT>::euclideanCluster(typename pcl::PointCloud<PointT>::Ptr cloud, KdTree<PointT>* tree, float distanceTol, int minSize, int maxSize)
{

	// TODO: Fill out this function to return list of indices for each cluster
	std::vector<pcl::PointIndices> clusters;
	std::vector<bool> processed(cloud->points.size(), false);
	for (int i = 0; i < cloud->points.size(); i++)
	{
		if (processed[i])
			continue;

		pcl::PointIndices cluster;
		euclideanCluster(i, cloud, processed, tree, distanceTol, cluster);
		if ((cluster.indices.size() >= minSize) && (cluster.indices.size() <= maxSize))
			clusters.push_back(cluster);
		i++;
	}

	std::sort(clusters.begin(), clusters.end(), [](pcl::PointIndices &a, pcl::PointIndices &b) { return a.indices.size() > b.indices.size(); });

	return clusters;
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // TODO:: Fill in the function to perform euclidean clustering to group detected obstacles
    // Creating the KdTree object for the search method of the extraction
	std::vector<pcl::PointIndices> clusterIndices;
#ifdef PCL_CLUSTER
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	tree->setInputCloud(cloud);

	pcl::EuclideanClusterExtraction<PointT> ec;
	ec.setClusterTolerance(clusterTolerance);
	ec.setMinClusterSize(minSize);
	ec.setMaxClusterSize(maxSize);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud);
	ec.extract(clusterIndices);
#else
	typename KdTree<PointT> *pTree = new KdTree<PointT>;
	pTree->setInputCloud(cloud);
	clusterIndices = euclideanCluster(cloud, pTree, clusterTolerance, minSize, maxSize);

#endif

	for (pcl::PointIndices getIndices: clusterIndices)
	{
		typename pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
		for (int idx: getIndices.indices)
			cloud_cluster->points.push_back(cloud->points[idx]);

		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		clusters.push_back(cloud_cluster);
	}

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}