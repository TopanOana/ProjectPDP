#pragma once
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>

cv::Mat houghLinesTransform(cv::Mat sourceImage);

std::vector<cv::Mat> splitImageInChunks(cv::Mat sourceImage, int numberChunks);

cv::Mat concatenateChunksHorizontally(std::vector<cv::Mat> chunks);

cv::Mat uniteMats(cv::Mat m1, cv::Mat m2);

void displayChunks(std::vector<cv::Mat> chunks);