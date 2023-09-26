/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include "../../render/render.h"
#include <unordered_set>
#include "../../processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  	// Add inliers
  	float scatter = 0.6;
  	for(int i = -5; i < 5; i++)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = i+scatter*rx;
  		point.y = i+scatter*ry;
  		point.z = 0;

  		cloud->points.push_back(point);
  	}
  	// Add outliers
  	int numOutliers = 10;
  	while(numOutliers--)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = 5*rx;
  		point.y = 5*ry;
  		point.z = 0;

  		cloud->points.push_back(point);

  	}
  	cloud->width = cloud->points.size();
  	cloud->height = 1;

  	return cloud;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
	ProcessPointClouds<pcl::PointXYZ> pointProcessor;
	return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}


pcl::visualization::PCLVisualizer::Ptr initScene()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("2D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
  	viewer->initCameraParameters();
  	viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
  	viewer->addCoordinateSystem (1.0);
  	return viewer;
}

std::unordered_set<int> Ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	std::unordered_set<int> inliersResult;
	std::unordered_set<int> inliers;
	srand(time(NULL));
	
	// TODO: Fill in this function

	// For max iterations 
	for (int idx = 0; idx < maxIterations; idx++)
	{
		// Randomly sample subset and fit line
		inliers.clear();
		while (inliers.size() < 2)
			inliers.insert(rand() % cloud->points.size());

		auto itr = inliers.begin();
		float x1, y1, x2, y2;
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;

		float A, B, C, dist, denom;
		A = y1 - y2;
		B = x2 - x1;
		C = x1 * y2 - x2 * y1;
		denom = sqrt(A * A + B * B);

		// Measure distance between every point and fitted line
		for (int cidx = 0; cidx < cloud->points.size(); cidx++)
		{
			if (inliers.count(cidx) > 0)
				continue;

			dist = fabs(A*cloud->points[cidx].x + B * cloud->points[cidx].y + C) / denom;
			// If distance is smaller than threshold count it as inlier
			if (dist < distanceTol)
			{
				inliers.insert(cidx);
			}
		}

		if (inliers.size() > inliersResult.size())
		{
			inliersResult = inliers;
		}
	}

	// Return indicies of inliers from fitted line with most inliers
	return inliersResult;

}

std::unordered_set<int> RansacPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	std::unordered_set<int> inliersResult;
	std::unordered_set<int> inliers;
	srand(time(NULL));

	// TODO: Fill in this function

	// For max iterations 
	for (int idx = 0; idx < maxIterations; idx++)
	{
		// Randomly sample subset and fit line
		inliers.clear();
		while (inliers.size() < 3)
			inliers.insert(rand() % cloud->points.size());

		auto itr = inliers.begin();
		pcl::PointXYZ pts[3];
		for (int i = 0; i < 3; i++)
		{
			pts[i].x = cloud->points[*itr].x;
			pts[i].y = cloud->points[*itr].y;
			pts[i].z = cloud->points[*itr].z;
			itr++;
		}

		float A, B, C, D, dist, denom;
		A = (pts[1].y - pts[0].y)*(pts[2].z- pts[0].z) - (pts[1].z- pts[0].z)*(pts[2].y- pts[0].y);
		B = (pts[1].z - pts[0].z)*(pts[2].x - pts[0].x) - (pts[1].x - pts[0].x)*(pts[2].z - pts[0].z);
		C = (pts[1].x - pts[0].x)*(pts[2].y - pts[0].y) - (pts[1].y - pts[0].y)*(pts[2].x - pts[0].x);
		D = -(A*pts[0].x + B * pts[0].y + C * pts[0].z);
		denom = sqrt(A * A + B * B + C * C);

		// Measure distance between every point and fitted line
		for (int cidx = 0; cidx < cloud->points.size(); cidx++)
		{
			if (inliers.count(cidx) > 0)
				continue;

			dist = fabs(A * cloud->points[cidx].x + B * cloud->points[cidx].y + C * cloud->points[cidx].z + D) / denom;
			// If distance is smaller than threshold count it as inlier
			if (dist < distanceTol)
			{
				inliers.insert(cidx);
			}
		}

		if (inliers.size() > inliersResult.size())
		{
			inliersResult = inliers;
		}
	}

	// Return indicies of inliers from fitted line with most inliers
	return inliersResult;

}

int main ()
{

	// Create viewer
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene();
	std::unordered_set<int> inliers;

	// Create data
	bool b2d = false;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	if (b2d)
	{
		// Create data
		cloud = CreateData();
		// TODO: Change the max iteration and distance tolerance arguments for Ransac function
		inliers = Ransac(cloud, 10, 1.0);
	}
	else
	{
		cloud = CreateData3D();
		inliers = RansacPlane(cloud, 10, 1.0);
	}
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

	for(int index = 0; index < cloud->points.size(); index++)
	{
		pcl::PointXYZ point = cloud->points[index];
		if(inliers.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}


	// Render 2D point cloud with inliers and outliers
	if(inliers.size())
	{
		renderPointCloud(viewer,cloudInliers,"inliers",Color(0,1,0));
  		renderPointCloud(viewer,cloudOutliers,"outliers",Color(1,0,0));
	}
  	else
  	{
  		renderPointCloud(viewer,cloud,"data");
  	}
	
  	while (!viewer->wasStopped ())
  	{
  	  viewer->spinOnce ();
  	}
  	
}
