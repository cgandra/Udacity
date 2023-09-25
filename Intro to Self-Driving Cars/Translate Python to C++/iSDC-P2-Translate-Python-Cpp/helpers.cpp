/**
	helpers.cpp

	Purpose: helper functions which are useful when
	implementing a 2-dimensional histogram filter.

	This file is incomplete! Your job is to make the
	normalize and blur functions work. Feel free to 
	look at helper.py for working implementations 
	which are written in python.
*/

#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include "helpers.h"
// #include "debugging_helpers.cpp"

using namespace std;
//ideally best to move this to common header to inlcude in all files
typedef vector< vector <float> > vec2Df;
/**
	TODO - implement this function

    Normalizes a grid of numbers. 

    @param grid - a two dimensional grid (vector of vectors of floats)
		   where each entry represents the unnormalized probability 
		   associated with that grid cell.

    @return - a new normalized two dimensional grid where the sum of 
    	   all probabilities is equal to one.
*/
vec2Df normalize(vec2Df grid) {

	// todo - your code here
	const int h = grid.size(), w = grid[0].size();
	vec2Df newGrid(h, vector<float>(w, 0.));

  //sum of all unnormalized probability in grid
	float tProb = 0;
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c++) {
			tProb += grid[r][c];
		}
	}

	//normalize grid
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c++) {
			newGrid[r][c] = grid[r][c]/ tProb;
		}
	}

	return newGrid;
}

/**
	TODO - implement this function.

    Blurs (and normalizes) a grid of probabilities by spreading 
    probability from each cell over a 3x3 "window" of cells. This 
    function assumes a cyclic world where probability "spills 
    over" from the right edge to the left and bottom to top. 

    EXAMPLE - After blurring (with blurring=0.12) a localized 
    distribution like this:

    0.00  0.00  0.00 
    0.00  1.00  0.00
    0.00  0.00  0.00 

    would look like this:
	
	0.01  0.02	0.01
	0.02  0.88	0.02
	0.01  0.02  0.01

    @param grid - a two dimensional grid (vector of vectors of floats)
		   where each entry represents the unnormalized probability 
		   associated with that grid cell.

	@param blurring - a floating point number between 0.0 and 1.0 
		   which represents how much probability from one cell 
		   "spills over" to it's neighbors. If it's 0.0, then no
		   blurring occurs. 

    @return - a new normalized two dimensional grid where probability 
    	   has been blurred.
*/
vec2Df blur(vec2Df grid, float blurring) {	
	// your code here
	const int h = grid.size(), w = grid[0].size();
	vec2Df newGrid(h, vector<float>(w, 0.));
	
	float centerProb = 1.0 - blurring;
	float cornerProb = blurring / 12.0;
	float adjacentProb = blurring / 6.0;

	//initialize blur kernel
	vec2Df window{	{cornerProb, adjacentProb, cornerProb},
					{adjacentProb, centerProb, adjacentProb},
					{cornerProb, adjacentProb, cornerProb} };

	//spread probability from each grid cell over a 3x3 window of cells
	int tr, tc;
	for (int r = 0; r < h; r++) { 
		for (int c = 0; c < w; c++) {
			for (int dx = -1; dx < 2; dx++) {
				for (int dy = -1; dy < 2; dy++) {
					tr = (r + dy);
					tc = (c + dx);
					tr = ((tr%h) + h) % h;
					tc = ((tc%w) + w) % w;
					newGrid[tr][tc] += (window[dy + 1][dx + 1] * grid[r][c]);
				}
			}
		}
	}

	return normalize(newGrid);
}

/** -----------------------------------------------
#
#
#	You do not need to modify any code below here.
#
#
# ------------------------------------------------- */


/**
    Determines when two grids of floating point numbers 
    are "close enough" that they should be considered 
    equal. Useful for battling "floating point errors".

    @param g1 - a grid of floats
    
    @param g2 - a grid of floats

    @return - A boolean (True or False) indicating whether
    these grids are (True) or are not (False) equal.
*/
bool close_enough(vec2Df g1, vec2Df g2) {
	int i, j;
	float v1, v2;
	if (g1.size() != g2.size()) {
		return false;
	}

	if (g1[0].size() != g2[0].size()) {
		return false;
	}
	for (i=0; i<g1.size(); i++) {
		for (j=0; j<g1[0].size(); j++) {
			v1 = g1[i][j];
			v2 = g2[i][j];
			if (abs(v2-v1) > 0.0001 ) {
				return false;
			}
		}
	}
	return true;
}

bool close_enough(float v1, float v2) { 
	if (abs(v2-v1) > 0.0001 ) {
		return false;
	} 
	return true;
}

/**
    Helper function for reading in map data

    @param s - a string representing one line of map data.

    @return - A row of chars, each of which represents the
    color of a cell in a grid world.
*/
vector <char> read_line(string s) {
	vector <char> row;

	size_t pos = 0;
	string token;
	string delimiter = " ";
	char cell;

	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		s.erase(0, pos + delimiter.length());

		cell = token.at(0);
		row.push_back(cell);
	}

	return row;
}

/**
    Helper function for reading in map data

    @param file_name - The filename where the map is stored.

    @return - A grid of chars representing a map.
*/
vector < vector <char> > read_map(string file_name) {
	ifstream infile(file_name);
	vector < vector <char> > map;
	if (infile.is_open()) {

		char color;
		vector <char> row;
		
		string line;

		while (std::getline(infile, line)) {
			row = read_line(line);
			map.push_back(row);
		}
	}
	return map;
}

/**
    Creates a grid of zeros

    For example:

    zeros(2, 3) would return

    0.0  0.0  0.0
    0.0  0.0  0.0

    @param height - the height of the desired grid

    @param width - the width of the desired grid.

    @return a grid of zeros (floats)
*/
vec2Df zeros(int height, int width) {
	int i, j;
	vec2Df newGrid(height, vector<float>(width, 0.0));
	return newGrid;
}

// int main() {
// 	vector < vector < char > > map = read_map("maps/m1.txt");
// 	show_grid(map);
// 	return 0;
// }
