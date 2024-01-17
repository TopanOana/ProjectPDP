#include<opencv2/opencv.hpp>
#include<iostream>
#include "DistributedTransform.h"
#include "NormalThreadsTransform.h"

using namespace std;
using namespace cv;


void testNormalThreadsTransform(string filename) {
	int threadsNumber = 4;  // You can adjust the number of threads
	int chunksNumber = 4;   // You can adjust the number of chunks
	NormalThreadsTransform ntt(threadsNumber, chunksNumber);
	ntt.gogoBoys(filename);
}

void testDistributedTransform(string filename) {
	DistributedTransform dt;
	dt.gogoBoys(filename);
}

int main()
{
	std::string filename("C:\\Users\\Naomi\\Downloads\\example.png");
	//testDistributedTransform(filename);

	// Test Normal Threads Transform
	testNormalThreadsTransform(filename);

	//destroyAllWindows();
	return 0;
}