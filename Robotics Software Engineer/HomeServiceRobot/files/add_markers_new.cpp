#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include "nav_msgs/Odometry.h"
#include <std_msgs/UInt8.h>


float pose_x = 0.0, pose_y = 0.0;
bool b_odom = false;
uint8_t po_state = 0;

void get_cur_pose(const nav_msgs::Odometry::ConstPtr& msg) {
	pose_x = msg->pose.pose.position.x;
	pose_y = msg->pose.pose.position.y;
	b_odom = true;
}

void get_po_stat(const std_msgs::UInt8::ConstPtr& msg) {
	po_state = msg->data;
}

int main(int argc, char** argv) {
	ros::init(argc, argv, "add_markers_new");
	ros::NodeHandle n;
	ros::Rate r(15);
	ros::Subscriber odom_sub = n.subscribe("/odom", 5, get_cur_pose);
	ros::Subscriber po_stat = n.subscribe("pick_obj_stat", 5, get_po_stat);
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

	enum State { PICKUP = 1, CARRY, DROP_OFF, DONE };
	State state = State::PICKUP;
	float dist_thresh = 0.4;
	float dist = 0;

	visualization_msgs::Marker marker;
	// Set the frame ID and timestamp.  See the TF tutorials for information on these.
	marker.header.frame_id = "map";
	// Set the namespace and id for this marker.  This serves to create a unique ID
	// Any marker sent with the same namespace and id will overwrite the old one
	marker.ns = "add_markers_new";
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

	while (ros::ok()) {
		while (marker_pub.getNumSubscribers() < 1) {
			if (!ros::ok()) {
				return 0;
			}
			ROS_WARN("Please create a subscriber to the marker");
			sleep(1);
		}

		// Set the frame ID and timestamp.  See the TF tutorials for information on these.
		marker.header.stamp = ros::Time::now();
		marker.lifetime = ros::Duration();
		
		if ((state == State::PICKUP) && b_odom) {
			// Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
			marker.action = visualization_msgs::Marker::ADD;

			// Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
			// Publish the marker & send pick up goal
			marker.pose.position.x = goals[0][0];
			marker.pose.position.y = goals[0][1];
			marker.pose.orientation.w = goals[0][2];

			// Publish the marker
			marker_pub.publish(marker);
			// Got odom messeage, robot is moving. Move to next state
			ROS_INFO("Odom message: %1.2f, %1.2f", pose_x, pose_y);
			state = State::CARRY;
			ROS_INFO("Hooray, pick up goal is set");
		}

		if ((state == State::CARRY) && (po_state == 1)) {
			marker.action = visualization_msgs::Marker::DELETE;

			ROS_INFO("Hooray, object is picked up");
			state = State::DROP_OFF;
		}

		if ((state == State::DROP_OFF) && (po_state == 1)) {

			// Publish the marker & send drop off goal
			marker.action = visualization_msgs::Marker::DELETE;
			marker.pose.position.x = goals[1][0];
			marker.pose.position.y = goals[1][1];
			marker.pose.orientation.w = goals[1][2];
			marker_pub.publish(marker);
			po_state = 0;

			ROS_INFO("Hooray, drop off goal set");
		}

		if ((state == State::DROP_OFF) && (po_state == 2)) {
			//Dropped
			marker.action = visualization_msgs::Marker::ADD;

			// Publish the marker
			marker_pub.publish(marker);
			ROS_INFO("Hooray, object is dropped off");
			state = State::DONE;
			break;
		}

		ros::spinOnce();		
	}

	return 0;
}
