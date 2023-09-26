/* \author Aaron Brown */
// Create simple 3d highway enviroment using PCL
// for exploring self-driving car sensors

#include "sensors/lidar.h"
#include "render/render.h"
#include "processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "processPointClouds.cpp"

std::vector<Car> initHighway(bool renderScene, pcl::visualization::PCLVisualizer::Ptr& viewer)
{

    Car egoCar( Vect3(0,0,0), Vect3(4,2,2), Color(0,1,0), "egoCar");
    Car car1( Vect3(15,0,0), Vect3(4,2,2), Color(0,0,1), "car1");
    Car car2( Vect3(8,-4,0), Vect3(4,2,2), Color(0,0,1), "car2");	
    Car car3( Vect3(-12,4,0), Vect3(4,2,2), Color(0,0,1), "car3");
  
    std::vector<Car> cars;
    cars.push_back(egoCar);
    cars.push_back(car1);
    cars.push_back(car2);
    cars.push_back(car3);

    if(renderScene)
    {
        renderHighway(viewer);
        egoCar.render(viewer);
        car1.render(viewer);
        car2.render(viewer);
        car3.render(viewer);
    }

    return cars;
}

void cityBlock(pcl::visualization::PCLVisualizer::Ptr& viewer, ProcessPointClouds<pcl::PointXYZI>* pointProcessorI, pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud) 
{
	inputCloud = pointProcessorI->FilterCloud(inputCloud, 0.21, Eigen::Vector4f(-8, -5, -2, 1), Eigen::Vector4f(30, 6.5, -0.01, 1));


	// TODO:: Create point processor
	std::pair<typename pcl::PointCloud<pcl::PointXYZI>::Ptr, typename pcl::PointCloud<pcl::PointXYZI>::Ptr> segmentCloud;
	segmentCloud = pointProcessorI->SegmentPlane(inputCloud, 25, 0.2);
	renderPointCloud(viewer, segmentCloud.first, "obstCloud", Color(0, 1, 1));
	renderPointCloud(viewer, segmentCloud.second, "planeCloud", Color(0, 1, 0));

	std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> obstClusters = pointProcessorI->Clustering(segmentCloud.first, 0.4, 10, 800);

	int clusterId = 0;
	std::vector<Color> colors = { Color(1,0,0), Color(1,1,0), Color(0,0,1) };

	for (pcl::PointCloud<pcl::PointXYZI>::Ptr cluster : obstClusters)
	{
		Box box = pointProcessorI->BoundingBox(cluster);
		renderPointCloud(viewer, cluster, "obstCloud" + std::to_string(clusterId), colors[clusterId%colors.size()]);
		renderBox(viewer, box, clusterId);
		++clusterId;
	}

}


//setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
void initCamera(CameraAngle setAngle, pcl::visualization::PCLVisualizer::Ptr& viewer)
{

    viewer->setBackgroundColor (0, 0, 0);
    
    // set camera position and angle
    viewer->initCameraParameters();
    // distance away in meters
    int distance = 16;
    
    switch(setAngle)
    {
        case XY : viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
        case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
        case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
        case FPS : viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
    }

    if(setAngle!=FPS)
        viewer->addCoordinateSystem (1.0);
}


int main (int argc, char** argv)
{
    std::cout << "starting enviroment" << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    CameraAngle setAngle = FPS;
    initCamera(setAngle, viewer);

	ProcessPointClouds<pcl::PointXYZI>* pointProcessorI = new ProcessPointClouds<pcl::PointXYZI>();
	std::vector<boost::filesystem::path> stream = pointProcessorI->streamPcd("../src/sensors/data/pcd/data_1");
	auto streamIterator = stream.begin();
	pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI;


#ifdef DEBUG_SINGLE_FRAME
	viewer->removeAllPointClouds();
	viewer->removeAllShapes();
	for (int i=0; i< argv[1]; i++)
		streamIterator++;
	inputCloudI = pointProcessorI->loadPcd((*streamIterator).string());
	cityBlock(viewer, pointProcessorI, inputCloudI);
	viewer->spin();
#else
	std::string fileName;
	int frameNum = 0;
	bool bDump = false;

	while (!viewer->wasStopped())
	{
		//Clear viewer
		if (streamIterator == stream.end())
		{
			bDump = false;
			streamIterator = stream.begin();
		}

		viewer->removeAllPointClouds();
		viewer->removeAllShapes();

		//Load pcd and run obstacle detection process
		inputCloudI = pointProcessorI->loadPcd((*streamIterator).string());
		cityBlock(viewer, pointProcessorI, inputCloudI);

		streamIterator++;

		viewer->spinOnce();
		if (bDump)
		{
			fileName = "frame_" + std::to_string(frameNum) + ".png";
			viewer->saveScreenshot(fileName);
		}
		frameNum++;
	}
#endif
}