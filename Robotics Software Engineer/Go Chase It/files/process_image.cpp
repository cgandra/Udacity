#include "ros/ros.h"
#include "ball_chaser/DriveToTarget.h"
#include <sensor_msgs/Image.h>

// Define a global client that can request services
ros::ServiceClient client;

// This function calls the command_robot service to drive the robot in the specified direction
void drive_robot(float lin_x, float ang_z)
{
    // TODO: Request a service and pass the velocities to it to drive the robot
    ball_chaser::DriveToTarget srv;

    srv.request.linear_x = lin_x;
    srv.request.angular_z = ang_z;

    // Call the command_robot service and pass the requested motor commands
    if (!client.call(srv)) {
        ROS_ERROR("Failed to call service command_robot");
    }

}

// This callback function continuously executes and reads the image data
void process_image_callback(const sensor_msgs::Image img)
{
	// TODO: Loop through each pixel in the image and check if there's a bright white one
	// Then, identify if this pixel falls in the left, mid, or right side of the image
	// Depending on the white ball position, call the drive_robot function and pass velocities to it
	// Request a stop when there's no white ball seen by the camera
	float lin_x = 0.0;
	float ang_z = 0.0;
	std::vector<int> ball_pos_x;
	int i;

	// Loop through each pixel in the image and check if there's a bright white one
	// Can be optimized to do early exit once ball is detected. Currently not implemented
	for (int r = 0; r < img.height; r++) {
		for (int c = 0; c < img.width; c++) {
			i = r * img.step + 3 * c;
			if ((img.data[i] == 255) && (img.data[i + 1] == 255) && (img.data[i + 2] == 255)) {
				ball_pos_x.push_back(c);
			}
		}
	}

	if (ball_pos_x.size()) {
		float avg_ball_pos_x = 0;
		avg_ball_pos_x = std::accumulate(ball_pos_x.begin(), ball_pos_x.end(), 0.0);
		avg_ball_pos_x /= ball_pos_x.size();

		float img_centre = img.width / 2;
		// Max angular velocity in left/right region is 0.3
		ang_z = (img_centre - avg_ball_pos_x)*0.3 / img_centre;
		// Forward region where angular velocity is set to zero is the middle 10% range of image
		if (fabs(ang_z) < 0.03) {
			ang_z = 0.0;
		}
		lin_x = 0.1;
		//ROS_INFO_STREAM(std::to_string(avg_ball_pos_x) + ", " + std::to_string(ang_z));
	}


	drive_robot(lin_x, ang_z);
}

int main(int argc, char** argv)
{
    // Initialize the process_image node and create a handle to it
    ros::init(argc, argv, "process_image");
    ros::NodeHandle n;

    // Define a client service capable of requesting services from command_robot
    client = n.serviceClient<ball_chaser::DriveToTarget>("/ball_chaser/command_robot");

    // Subscribe to /camera/rgb/image_raw topic to read the image data inside the process_image_callback function
    ros::Subscriber sub1 = n.subscribe("/camera/rgb/image_raw", 10, process_image_callback);

    // Handle ROS communication events
    ros::spin();

    return 0;
}
