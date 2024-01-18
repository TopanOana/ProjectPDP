#include "HoughTransformFunctions.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void polarToCartesian(double rho, int theta, Point& p1, Point& p2) {

	int x0 = cvRound(rho * cos(theta));
	int y0 = cvRound(rho * sin(theta));

	p1.x = cvRound(x0 + 1000 * (-sin(theta)));
	p1.y = cvRound(y0 + 1000 * (cos(theta)));

	p2.x = cvRound(x0 - 1000 * (-sin(theta)));
	p2.y = cvRound(y0 - 1000 * (cos(theta)));
}

cv::Mat houghLinesTransform(cv::Mat sourceImage)
{
	Mat imageWEdges;
	int lineThreshold = 150;

	//the edge detection
	Canny(sourceImage, imageWEdges, 50, 200, 3, false);

	//copies the edges to the images that display results in BGR (blue green red == colour)
	Mat imageWLines;
	//from grayscale to color (blue green red)
	cvtColor(imageWEdges, imageWLines, COLOR_GRAY2BGR);


	vector<Vec2f> lines;
	// 1 -> resolution of accumulator
	// CV_PI/180 -> angle resolution of accumulator
	// 150 -> threshold for line detection
	//changing times boy
	//HoughLines(imageWEdges, lines, 1, CV_PI / 180, 150);

	//de aici 
	int maxDistance = sqrt(imageWEdges.rows * imageWEdges.rows + imageWEdges.cols * imageWEdges.cols);

	vector<vector<int>> votes(2 * maxDistance, vector<int>(NUMBINS, 0));

	for (int i = 0; i < imageWEdges.rows; i++) {
		for (int j = 0; j < imageWEdges.cols; j++) {
			if (imageWEdges.at<uchar>(i, j) == 255) {
				//cout << "in theta turn i=" << i << " j=" << j << endl;
				for (int theta = 0; theta < 180; theta += BIN_WIDTH) {

					int rho = round(j * cos(theta - 90) + i * sin(theta - 90)) + maxDistance;

					//cout << "rho = " << rho << " theta = " << theta << endl;

					votes[rho][theta]++;
				}
			}


		}
	}

	//pana aici pt cuda kernel

	for (int i = 0; i < votes.size(); i++) {
		for (int j = 0; j < votes[i].size(); j++) {
			if (votes[i][j] >= lineThreshold) {
				//cout << "in line threshold i=" << i << " j=" << j << endl;
				int rho = i - maxDistance;
				int theta = j - 90;

				Point p1, p2;
				polarToCartesian(rho, theta, p1, p2);

				line(imageWLines, p1, p2, Scalar(0, 0, 255), 2, LINE_AA);
			}
		}
	}



	//for (int x = 0; x < lines.size(); x++) {
	//	double rho = lines[x][0];
	//	double theta = lines[x][1];

	//	double a = cos(theta);
	//	double b = sin(theta);
	//	double x0 = a * rho;
	//	double y0 = b * rho;

	//	Point point1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
	//	Point point2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

	//	line(imageWLines, point1, point2, Scalar(0, 0, 255), 3, LINE_AA, 0);

	//}

	return imageWLines;
}

//std::vector<cv::Mat> splitImageInChunks(cv::Mat sourceImage, int numberChunks)
//{
//	vector<Mat> chunks;
//	int chunkHeight = sourceImage.rows / numberChunks;
//	int remainingRows = sourceImage.rows % numberChunks;
//
//	int 
//
//	int startingRow = 0;
//	for (int i = 0; i < numberChunks; i++) {
//
//		int currentChunkHeight = chunkHeight ;
//		if (i < remainingRows)
//			currentChunkHeight++;
//
//
//		Rect rect(0, startingRow, sourceImage.cols, currentChunkHeight);
//		Mat chunk(sourceImage, rect);
//		chunks.push_back(chunk);
//
//		startingRow += currentChunkHeight;
//	}
//
//	return chunks;
//}

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

cv::Mat uniteMats(cv::Mat m1, cv::Mat m2)
{
	Mat result;
	bitwise_or(m1, m2, result);
	return result;
}

void displayChunks(std::vector<cv::Mat> chunks)
{
	for (int i = 0; i < chunks.size(); i++) {
		Mat currentChunk = chunks[i];
		imshow("Chunk " + to_string(i), currentChunk);
	}
	cv::waitKey(0);
}
