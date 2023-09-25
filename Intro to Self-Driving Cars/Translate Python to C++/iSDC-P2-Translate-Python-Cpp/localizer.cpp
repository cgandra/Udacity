/**
	localizer.cpp

	Purpose: implements a 2-dimensional histogram filter
	for a robot living on a colored cyclical grid by 
	correctly implementing the "initialize_beliefs", 
	"sense", and "move" functions.

	This file is incomplete! Your job is to make these
	functions work. Feel free to look at localizer.py 
	for working implementations which are written in python.
*/

#include "localizer.h"
#include "helpers.cpp"
#include <stdlib.h>
#include "debugging_helpers.cpp"

using namespace std;
//ideally best to move this to common header to inlcude in all files
typedef vector< vector <float> > vec2Df;

/**
	TODO - implement this function 
    
    Initializes a grid of beliefs to a uniform distribution. 

    @param grid - a two dimensional grid map (vector of vectors 
    	   of chars) representing the robot's world. For example:
    	   
    	   g g g
    	   g r g
    	   g g g
		   
		   would be a 3x3 world where every cell is green except 
		   for the center, which is red.

    @return - a normalized two dimensional grid of floats. For 
           a 2x2 grid, for example, this would be:

           0.25 0.25
           0.25 0.25
*/
vec2Df initialize_beliefs(vector< vector <char> > grid) {
	// your code here
	float beliefPerCell = 1.0 / (grid.size() * grid[0].size());
	vec2Df newGrid(grid.size(), vector<float>(grid[0].size(), beliefPerCell));
	return newGrid;	
}

/**
  TODO - implement this function 
    
    Implements robot motion by updating beliefs based on the 
    intended dx and dy of the robot. 

    For example, if a localized robot with the following beliefs

    0.00  0.00  0.00
    0.00  1.00  0.00
    0.00  0.00  0.00 

    and dx and dy are both 1 and blurring is 0 (noiseless motion),
    than after calling this function the returned beliefs would be

    0.00  0.00  0.00
    0.00  0.00  0.00
    0.00  0.00  1.00 

  @param dy - the intended change in y position of the robot

  @param dx - the intended change in x position of the robot

    @param beliefs - a two dimensional grid of floats representing
         the robot's beliefs for each cell before sensing. For 
         example, a robot which has almost certainly localized 
         itself in a 2D world might have the following beliefs:

         0.01 0.98
         0.00 0.01

    @param blurring - A number representing how noisy robot motion
           is. If blurring = 0.0 then motion is noiseless.

    @return - a normalized two dimensional grid of floats 
         representing the updated beliefs for the robot. 
*/
vec2Df move(int dy, int dx, 
  vec2Df beliefs,
  float blurring) 
{
	// your code here
    const int h = beliefs.size(), w = beliefs[0].size();
	vec2Df newGrid(h, vector<float>(w, 0.));
  
	int tr, tc;
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			tr = (r + dy);
			tc = (c + dx);
			tr = ((tr%h) + h) % h;
			tc = ((tc%w) + w) % w;
			newGrid[tr][tc] = beliefs[r][c];
		}
	}
  
	return blur(newGrid, blurring);
}


/**
	TODO - implement this function 
    
    Implements robot sensing by updating beliefs based on the 
    color of a sensor measurement 

	@param color - the color the robot has sensed at its location

	@param grid - the current map of the world, stored as a grid
		   (vector of vectors of chars) where each char represents a 
		   color. For example:

		   g g g
    	   g r g
    	   g g g

   	@param beliefs - a two dimensional grid of floats representing
   		   the robot's beliefs for each cell before sensing. For 
   		   example, a robot which has almost certainly localized 
   		   itself in a 2D world might have the following beliefs:

   		   0.01 0.98
   		   0.00 0.01

    @param p_hit - the RELATIVE probability that any "sense" is 
    	   correct. The ratio of p_hit / p_miss indicates how many
    	   times MORE likely it is to have a correct "sense" than
    	   an incorrect one.

   	@param p_miss - the RELATIVE probability that any "sense" is 
    	   incorrect. The ratio of p_hit / p_miss indicates how many
    	   times MORE likely it is to have a correct "sense" than
    	   an incorrect one.

    @return - a normalized two dimensional grid of floats 
    	   representing the updated beliefs for the robot. 
*/
vec2Df sense(char color, 
	vector< vector <char> > grid, 
	vec2Df beliefs, 
	float p_hit,
	float p_miss) 
{

	// your code here
	const int h = grid.size(), w = grid[0].size();
	vec2Df newGrid(h, vector<float>(w, 0.));
	float hit;
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			hit = (grid[r][c] == color);
			newGrid[r][c] = beliefs[r][c] * (hit*p_hit + (1.0-hit)*p_miss);
		}
	}

	return normalize(newGrid);
}