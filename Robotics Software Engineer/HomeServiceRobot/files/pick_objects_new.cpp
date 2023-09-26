#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/UInt8.h>

// Define a client for to send goal requests to the move_base server through a SimpleActionClient
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

float goal_x = 0.0, goal_y = 0.0;
float goal_orient[4] = { 0.0, 0.0, 0.0, 0.0 };
bool b_got_goal = false;

void get_goal(const visualization_msgs::Marker::ConstPtr& msg)
{
	goal_x = msg->pose.position.x;
	goal_y = msg->pose.position.y;
	goal_orient[0] = msg->pose.orientation.x;
	goal_orient[1] = msg->pose.orientation.y;
	goal_orient[2] = msg->pose.orientation.z;
	goal_orient[3] = msg->pose.orientation.w;
	b_got_goal = true;
}

int move_robot(MoveBaseClient & ac, float x, float y, float orient[4])
{
	move_base_msgs::MoveBaseGoal goal;

	// set up the frame parameters
	goal.target_pose.header.frame_id = "map";
	goal.target_pose.header.stamp = ros::Time::now();

	ROS_INFO("Robot's goal: %1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f", x, y, orient[0], orient[1], orient[2], orient[3]);

	goal.target_pose.pose.position.x = x;
	goal.target_pose.pose.position.y = y;
	goal.target_pose.pose.orientation.x = orient[0];
	goal.target_pose.pose.orientation.y = orient[1];
	goal.target_pose.pose.orientation.z = orient[2];
	goal.target_pose.pose.orientation.w = orient[3];

	// Send the goal position and orientation for the robot to reach
	ROS_INFO("Sending goal to robot");
	ac.sendGoal(goal);

	// Wait an infinite time for the results
	ac.waitForResult();

	// Check if the robot reached its goal
	if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED) {
		ROS_INFO("Hooray, reached goal");
		return 1;
	}
	else {
		ROS_INFO("Failed to reach goal for some reason");
		return 0;
	}
}

int main(int argc, char** argv) {
	// Initialize the pick_objects node
	ros::init(argc, argv, "pick_objects_new");

	ros::NodeHandle n;
	ros::Subscriber marker_sub = n.subscribe<visualization_msgs::Marker>("visualization_marker", 5, get_goal);
	ros::Publisher po_stat = n.advertise<std_msgs::UInt8>("pick_obj_stat", 1);


	//tell the action client that we want to spin a thread by default
	MoveBaseClient ac("move_base", true);

	// Wait 5 sec for move_base action server to come up
	while (!ac.waitForServer(ros::Duration(5.0))) {
		ROS_INFO("Waiting for the move_base action server to come up");
	}

	int ret;
	std_msgs::UInt8 msg;
	unsigned char val = 1;

	while (ros::ok()) {
		while (po_stat.getNumSubscribers() < 1) {
			if (!ros::ok()) {
				return 0;
			}
			ROS_WARN("Please create a subscriber to the pick_obj_stat");
			sleep(1);
		}

		if (b_got_goal) {
			b_got_goal = false;
			ret = move_robot(ac, goal_x, goal_y, goal_orient);
			if (ret == 0) {
				break;
			}
			msg.data = val;
			po_stat.publish(msg);
			val++;

			if (val == 3) {
				break;
			}
		}
		ros::spinOnce();
	}

	return 0;
}