#pragma once
#include <vector>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>

class NormalThreadsTransform
{
private:
	int threadsNumber;
	int chunksNumber;

	std::vector<cv::Mat> processChunks(cv::Mat grayscaleImage, cv::Mat colorImage);


public:
	NormalThreadsTransform(int thNr, int chNr);

	std::vector<std::vector<cv::Mat>> imageProcessing(std::vector<cv::Mat> grayscaleChunks, std::vector<cv::Mat> colorChunks);

	void gogoBoys(std::string filename);
};

