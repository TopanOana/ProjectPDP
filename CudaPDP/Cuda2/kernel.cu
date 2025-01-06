
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ImageProcessingFunctions.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>


#include <stdio.h>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <iostream>

#define BIN_WIDTH 1
#define NUMBINS 180 / BIN_WIDTH
#define KERNEL_SIZE 3
#define THRESHOLD 50
#define RATIO 3


void polarToCartesian(double rho, int theta, cv::Point& p1, cv::Point& p2) {

	int x0 = cvRound(rho * cos(theta));
	int y0 = cvRound(rho * sin(theta));

	p1.x = cvRound(x0 + 1000 * (-sin(theta)));
	p1.y = cvRound(y0 + 1000 * (cos(theta)));

	p2.x = cvRound(x0 - 1000 * (-sin(theta)));
	p2.y = cvRound(y0 - 1000 * (cos(theta)));
}


__global__ void houghTKernel(uchar* imageWEdges, int rows, int cols, int* votes, int chunkHeight, int maxChunkHeight, int remainingRows) {

	int initialRow = threadIdx.x * chunkHeight;
	int currentChunkHeight = (threadIdx.x < remainingRows) ? maxChunkHeight : chunkHeight;
	int lastRow = initialRow + currentChunkHeight;

	int currentMaxDistance = sqrtf((lastRow - initialRow + 1) * (lastRow - initialRow + 1) + cols * cols);


	for (int i = initialRow; i < lastRow; i++) {
		for (int j = 0; j < cols; j++) {
			if (imageWEdges[i * cols + j] == 255) {
				for (int theta = 0; theta < 180; theta += BIN_WIDTH) {
					int rho = round(j * cosf(theta - 90) + i * sinf(theta - 90)) + currentMaxDistance;

					votes[threadIdx.x * rho * NUMBINS + theta]++;
				}
			}
		}
	}

}


cudaError_t cudaHoughTransform(cv::Mat sourceImage, int nrChunks) {

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t cudaStatus;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	cv::Mat imageWEdges;
	Canny(sourceImage, imageWEdges, 50, 200, 3, false);


	int chunkHeight = imageWEdges.rows / nrChunks;
	int remainingRows = imageWEdges.rows % nrChunks;

	int maxChunkHeight = chunkHeight + (remainingRows > 0 ? 1 : 0);

	//allocate space for votes

	//nrchunks * sqrt(nrRowsInChunk*nrRowsInChunk + cols*cols) * NUMBINS

	int allocatedSpace = nrChunks * 2 * sqrt(maxChunkHeight * maxChunkHeight + imageWEdges.cols * imageWEdges.cols) * NUMBINS;

	//int maxDistance = sqrt(imageWEdges.rows * imageWEdges.rows + imageWEdges.cols * imageWEdges.cols);

	// Allocate and initialize votesHost
	int* votesHost = (int*)malloc(sizeof(int) * allocatedSpace);

	for (int i = 0; i < allocatedSpace; i++) {
		votesHost[i] = 0;
	}


	// Allocate votesDevice as a 2D array
	int* votesDevice;
	cudaStatus = cudaMalloc(&votesDevice, sizeof(int) * allocatedSpace);

	cudaStatus = cudaMemcpy(votesDevice, votesHost, sizeof(int) * allocatedSpace, cudaMemcpyHostToDevice);



	//allocate space for image

	uchar* imageWEgesDevice;
	cudaStatus = cudaMalloc(&imageWEgesDevice, imageWEdges.rows * imageWEdges.cols * sizeof(uchar));


	cudaStatus = cudaMemcpy(imageWEgesDevice, imageWEdges.data, imageWEdges.rows * imageWEdges.cols, cudaMemcpyHostToDevice);

	cudaEventRecord(start);


	//launching kernels
	houghTKernel << <1, nrChunks >> > (imageWEgesDevice, imageWEdges.rows, imageWEdges.cols, votesDevice, chunkHeight, maxChunkHeight, remainingRows);

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);


	cudaStatus = cudaMemcpy(votesHost, votesDevice, allocatedSpace, cudaMemcpyDeviceToHost);

	std::vector<cv::Mat> finalChunks;
	int value = 2 * sqrt(maxChunkHeight * maxChunkHeight + imageWEdges.cols * imageWEdges.cols);
	int lineThreshold = 150;

	for (int i = 0; i < nrChunks; i++) {
		int currentChunkHeight = (i < remainingRows) ? maxChunkHeight : chunkHeight;

		cv::Rect rect(0, i * chunkHeight, imageWEdges.cols, currentChunkHeight);
		cv::Mat chunk(imageWEdges, rect);
		if (currentChunkHeight < maxChunkHeight) {
			// Resize the chunk to have the same number of rows as maxChunkHeight
			cv::Mat resizedChunk;
			resize(chunk, resizedChunk, cv::Size(imageWEdges.cols, maxChunkHeight));
			chunk = resizedChunk;
		}
		cv::Mat finol;
		cv::cvtColor(chunk, finol, cv::ColorConversionCodes::COLOR_GRAY2BGR);


		///actual check for lines
		for (int x = 0; x < value; x++) {
			for (int y = 0; y < NUMBINS; y++) {
				if (votesHost[nrChunks * value + y] >= lineThreshold) {
					int rho = x - value / 2;
					int theta = y - 90;

					cv::Point p1, p2;
					polarToCartesian(rho, theta, p1, p2);

					line(finol, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LineTypes::LINE_AA);
				}
			}
		}

		finalChunks.push_back(finol);
	}


	cv::Mat linesImage = concatenateChunksVertically(finalChunks);

	imshow("source image", sourceImage);
	imshow("detected lines", linesImage);

	// Calculate and print the elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << "cuda Time taken: " << milliseconds << " ms" << std::endl;



	cv::waitKey(0);

Error:

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(imageWEgesDevice);

	free(votesHost);
	cudaFree(votesDevice);

	return cudaStatus;
}



int main()
{
	std::string filename("C:\\Users\\Oana\\source\\repos\\ProjectPDP\\image.png");

	cv::Mat sourceImage;

	sourceImage = cv::imread(filename, cv::ImreadModes::IMREAD_GRAYSCALE);
	
	

	// Add vectors in parallel.
	cudaError_t cudaStatus = cudaHoughTransform(sourceImage, 9);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "houghTransformWithCuda failed!");
		return 1;
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

