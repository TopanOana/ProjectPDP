#include "DistributedTransform.h"
#include <mpi.h>
#include "HoughTransformFunctions.h"

using namespace std;
using namespace cv;

void DistributedTransform::sendMat(cv::Mat& mat, int destination)
{
	cout << "in send mat" << endl;
	vector<uchar> byteArray;
	imencode(".png", mat, byteArray);

	int size = byteArray.size();

	MPI_Ssend(&size, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
	cout << "sent size of bytearray" << endl;

	MPI_Ssend(byteArray.data(), size, MPI_UNSIGNED_CHAR, destination, 0, MPI_COMM_WORLD);
	cout << "send bytearray" << endl;

}

Mat DistributedTransform::receiveMat(int source)
{
	cout << "in receive mat" << endl;
	MPI_Status status;
	int size;
	MPI_Recv(&size, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
	cout << "recv size of bytearray" << endl;

	vector<uchar> byteArray(size);
	MPI_Recv(byteArray.data(), size, MPI_UNSIGNED_CHAR, source, 0, MPI_COMM_WORLD, &status);

	cout << "recv bytearray" << endl;
	return imdecode(byteArray, IMREAD_UNCHANGED);
}

void DistributedTransform::masterWorker(std::string filename)
{
	cout << "starting admin" << endl;

	Mat grayscaleImage = imread(filename, IMREAD_GRAYSCALE);
	Mat colorImage = imread(filename, IMREAD_COLOR);

	if (grayscaleImage.empty() || colorImage.empty()) {
		cout << "error opening image" << endl;
		exit(1);
	}

	chrono::high_resolution_clock::time_point const startTime = chrono::high_resolution_clock::now();

	int nrProcesses;
	MPI_Comm_size(MPI_COMM_WORLD, &nrProcesses);

	int nrChunks = nrProcesses - 1;

	vector<Mat> grayscaleChunks = splitImageInChunks(grayscaleImage, nrChunks);
	//vector<Mat> colorChunks = splitImageInChunks(colorImage, nrChunks);

	for (int process = 1; process < nrProcesses; process++) {
		Mat currentGrayscaleChunk = grayscaleChunks[process - 1];
		//Mat currentColorChunk = colorChunks[process - 1];

		sendMat(currentGrayscaleChunk, process);
		//sendMat(currentColorChunk, process);
	}

	vector<Mat> linesResult;

	for (int process = 1; process < nrProcesses; process++) {
		Mat recieveLines = receiveMat(process);
		linesResult.push_back(recieveLines);
	}

	Mat linesImage = concatenateChunksVertically(linesResult);

	imshow("source image", colorImage);
	imshow("detected lines", linesImage);

	chrono::high_resolution_clock::time_point const endTime = chrono::high_resolution_clock::now();

	printf("\distributed time : %ld microseconds\n", chrono::duration_cast<chrono::microseconds>(endTime - startTime).count());

	cout << "admin finished" << endl;

	waitKey(0);
}

void DistributedTransform::worker(int my_rank)
{
	cout << "worker " << my_rank << " started" << endl;

	Mat grayscaleChunk = receiveMat(0);

	Mat chunkLines = houghLinesTransform(grayscaleChunk);

	sendMat(chunkLines, 0);

	cout << "worker " << my_rank << " done" << endl;

}

void DistributedTransform::gogoBoys(string filename)
{
	MPI_Init(NULL, NULL);

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (my_rank == 0)
	{
		this->masterWorker(filename);
	}
	else {
		this->worker(my_rank);
	}
	MPI_Finalize();
}
