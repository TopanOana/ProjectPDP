#pragma once
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>

std::vector<cv::Mat> splitImageInChunks(cv::Mat sourceImage, int numberChunks);

cv::Mat concatenateChunksVertically(std::vector<cv::Mat> chunks);