#include "ImageProcessingFunctions.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

std::vector<cv::Mat> splitImageInChunks(cv::Mat sourceImage, int numberChunks)
{
	vector<Mat> chunks;
	int chunkHeight = sourceImage.rows / numberChunks;
	int remainingRows = sourceImage.rows % numberChunks;

	int maxChunkHeight = chunkHeight + (remainingRows > 0 ? 1 : 0);

	for (int i = 0; i < numberChunks; i++) {
		int currentChunkHeight = (i < remainingRows) ? maxChunkHeight : chunkHeight;

		Rect rect(0, i * chunkHeight, sourceImage.cols, currentChunkHeight);
		Mat chunk(sourceImage, rect);

		if (currentChunkHeight < maxChunkHeight) {
			// Resize the chunk to have the same number of rows as maxChunkHeight
			Mat resizedChunk;
			resize(chunk, resizedChunk, Size(sourceImage.cols, maxChunkHeight));
			chunks.push_back(resizedChunk);
		}
		else {
			chunks.push_back(chunk);
		}
	}

	return chunks;
}

cv::Mat concatenateChunksVertically(std::vector<cv::Mat> chunks)
{
	Mat result;
	vconcat(chunks, result);
	return result;
}
