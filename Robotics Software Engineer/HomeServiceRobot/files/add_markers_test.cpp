#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

int main(int argc, char** argv)
{
	ros::init(argc, argv, "add_markers_test");
	ros::NodeHandle n;
	ros::Rate r(15);
	ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);

	uint32_t shape = visualization_msgs::Marker::CUBE;

	// Define multiple goals i.e positions and orientations for the robot to reach
	// The first goal should be your desired pickup goal and the second goal should be your desired drop off goal
	// The robot has to travel to the desired pickup zone, display a message that it reached its destination, wait 5 seconds,
	// travel to the desired drop off zone, and display a message that it reached the drop off zone
	float goals[2][3] = {
		{ -4, -5, 1 },
		{ -6,  6, 1 }
	};

	bool is_picked = false;

	visualization_msgs::Marker marker;
	// Set the frame ID and timestamp.  See the TF tutorials for information on these.
	marker.header.frame_id = "map";
	// Set the namespace and id for this marker.  This serves to create a unique ID
	// Any marker sent with the same namespace and id will overwrite the old one
	marker.ns = "add_markers_test";
	marker.id = 0;

	// Set the marker type
	marker.type = shape;

	// Set the scale of the marker -- 1x1x1 here means 1m on a side
	marker.scale.x = 0.4;
	marker.scale.y = 0.4;
	marker.scale.z = 0.4;

	// Set the color -- be sure to set alpha to something non-zero!
	marker.color.r = 1.0f;
	marker.color.g = 0.0f;
	marker.color.b = 0.0f;
	marker.color.a = 1.0;

	marker.pose.position.z = 0.5;
	marker.pose.orientation.x = 0.0;
	marker.pose.orientation.y = 0.0;
	marker.pose.orientation.z = 0.0;

	while (ros::ok())
	{
		while (marker_pub.getNumSubscribers() < 1)
		{
			if (!ros::ok())
			{
				return 0;
			}
			ROS_WARN_ONCE("Please create a subscriber to the marker");
			sleep(1);
		}

		// Set the frame ID and timestamp.  See the TF tutorials for information on these.
		marker.header.stamp = ros::Time::now();
		marker.lifetime = ros::Duration();

		if (!is_picked) {
			// Set the marker at pick up loc
			marker.action = visualization_msgs::Marker::ADD;

			// Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
			marker.pose.position.x = goals[0][0];
			marker.pose.position.y = goals[0][1];
			marker.pose.orientation.w = goals[0][2];

			// Publish the marker
			marker_pub.publish(marker);
			ros::Duration(5).sleep();

			// Clear the marker at pick up loc
			marker.action = visualization_msgs::Marker::DELETE;
			// Publish the marker
			marker_pub.publish(marker);
			is_picked = true;
			ROS_INFO("Hooray, object is picked up");
			ros::Duration(5).sleep();
		}
		else {
			// Set the marker at drop off loc
			marker.action = visualization_msgs::Marker::ADD;

			// Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
			marker.pose.position.x = goals[1][0];
			marker.pose.position.y = goals[1][1];
			marker.pose.orientation.w = goals[1][2];

			// Publish the marker
			marker_pub.publish(marker);
			ROS_INFO("Hooray, object is dropped off");
			ros::Duration(1).sleep();
			break;
		}

		ros::spinOnce();		
	}

	return 0;
}
