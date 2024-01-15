#pragma once
#include <string>
#include <opencv2/opencv.hpp>

class DistributedTransform
{
private:
	

public:
	void masterWorker(std::string filename);

	void worker();

	void gogoBoys();
};

