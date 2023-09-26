#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

// Define a client for to send goal requests to the move_base server through a SimpleActionClient
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

int main(int argc, char** argv) {
	// Initialize the pick_objects node
	ros::init(argc, argv, "pick_objects");

	//tell the action client that we want to spin a thread by default
	MoveBaseClient ac("move_base", true);

	// Wait 5 sec for move_base action server to come up
	while (!ac.waitForServer(ros::Duration(5.0))) {
		ROS_INFO("Waiting for the move_base action server to come up");
	}

	move_base_msgs::MoveBaseGoal goal;

	// set up the frame parameters
	goal.target_pose.header.frame_id = "map";
	goal.target_pose.header.stamp = ros::Time::now();


	// Define multiple goals i.e positions and orientations for the robot to reach
	// The first goal should be your desired pickup goal and the second goal should be your desired drop off goal
	// The robot has to travel to the desired pickup zone, display a message that it reached its destination, wait 5 seconds,
	// travel to the desired drop off zone, and display a message that it reached the drop off zone
	float goals[2][3] = {
							{ -4, -5, 1},
							{ -6,  6, 1}
						};

	int no_goals = sizeof(goals)/sizeof(goals[0]);

	for (int i=0; i<no_goals; i++){
		goal.target_pose.pose.position.x = goals[i][0];
		goal.target_pose.pose.position.y = goals[i][1];
		goal.target_pose.pose.orientation.w = goals[i][2];

		// Send the goal position and orientation for the robot to reach
		ROS_INFO("Sending goal");
		ac.sendGoal(goal);

		// Wait an infinite time for the results
		ac.waitForResult();

		// Check if the robot reached its goal
		if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){
			ROS_INFO("Hooray, reached goal");
		}
		else{
			ROS_INFO("Failed to reach goal for some reason");
		}
		ros::Duration(5.0).sleep();
	}

	return 0;
}