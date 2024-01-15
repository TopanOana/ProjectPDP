#pragma once
#include <string>
#include <opencv2/opencv.hpp>

class DistributedTransform
{
private:
	void sendMat(cv::Mat& mat, int destination);

	cv::Mat receiveMat(int source);

public:
	void masterWorker(std::string filename);

	void worker(int my_rank);

	void gogoBoys();
};

