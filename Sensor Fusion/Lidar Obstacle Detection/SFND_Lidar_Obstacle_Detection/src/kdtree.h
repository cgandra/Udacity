#ifndef KDTREE_H
#define KDTREE_H
#include <pcl/cloud_iterator.h>

/* \author Aaron Brown */
// Quiz on implementing kd tree

typedef unsigned int uint;

// Structure to represent node of kd tree
template<typename PointT>
struct Node
{
	PointT point;
	int id;
	Node<PointT>* left;
	Node<PointT>* right;

	Node(PointT pt, int setId)
	:	point(pt), id(setId), left(NULL), right(NULL)
	{}
};

template<typename PointT>
struct KdTree
{
	Node<PointT>* root;

	KdTree()
	: root(NULL)
	{}

	Node<PointT>* insert(typename pcl::PointCloud<PointT>::iterator start, typename pcl::PointCloud<PointT>::iterator end, int depth, std::vector<int>::iterator ids)
	{
		if (start >= end) return NULL;

		int axis = depth % 3;
		auto cmp = [axis](const PointT & p1, const PointT & p2) { return p1.data[axis] < p2.data[axis]; };

		std::size_t len = end- start;
		auto mid = start + len / 2;
		std::nth_element(start, mid, end, cmp);

		while (mid > start && (mid - 1)->data[axis] == mid->data[axis]) {
			--mid;
		}

		len = mid - start;
		auto mid_ids = ids + len;
		Node<PointT>* node = new Node<PointT>(*mid, *mid_ids);
		node->left = insert(start, mid, depth + 1, ids);
		node->right = insert(mid + 1, end, depth + 1, mid_ids + 1);
		return node;

	}

	void setInputCloud(typename pcl::PointCloud<PointT>::Ptr cloud)
	{
		// TODO: Fill in this function to insert a new point into the tree
		// the function should create a new node and place correctly with in the root
		std::vector<int> ids(cloud->points.size());
		std::iota(ids.begin(), ids.end(), 0);
		root = insert(cloud->points.begin(), cloud->points.end(), 0, ids.begin());
	}

	void search(Node<PointT> *&node, uint depth, PointT &target, float distanceTol, std::vector<int> &ids)
	{
		if (node == NULL)
			return;

		bool valid_data = true;
		for (int i = 0; i < 3; i++)
		{
			valid_data = valid_data && ((target.data[i] - distanceTol) <= node->point.data[i]);
			valid_data = valid_data && ((target.data[i] + distanceTol) >= node->point.data[i]);
		}

		if (valid_data)
		{
			float distance = 0;
			float diff;
			for (int i = 0; i < 3; i++)
			{
				diff = node->point.data[i] - target.data[i];
				distance += (diff*diff);
			}

			distance = sqrt(distance);
			if (distance <= distanceTol)
			{
				ids.push_back(node->id);
			}
		}

		int cd = depth % 3;
		if ((target.data[cd] - distanceTol) < node->point.data[cd])
		{
			search(node->left, depth+1, target, distanceTol, ids);
		}
	    
		if ((target.data[cd] + distanceTol) > node->point.data[cd])
		{
			search(node->right, depth + 1, target, distanceTol, ids);
		}
	}

	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(PointT &target, float distanceTol)
	{
		std::vector<int> ids;
		search(root, 0, target, distanceTol, ids);

		return ids;
	}

};

#endif