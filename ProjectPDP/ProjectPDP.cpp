#include<opencv2/opencv.hpp>
#include<iostream>
#include "DistributedTransform.h"

using namespace std;
using namespace cv;


void testDistributedTransform(string filename) {
	DistributedTransform dt;
	dt.gogoBoys(filename);
}



int main()
{
	std::string filename("C:\\Users\\Oana\\source\\repos\\ProjectPDP\\image.png");
	testDistributedTransform(filename);

	destroyAllWindows();
	return 0;
}