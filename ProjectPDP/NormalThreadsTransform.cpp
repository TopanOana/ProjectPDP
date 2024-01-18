#include "NormalThreadsTransform.h"
#include "HoughTransformFunctions.h"
#include <thread>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

NormalThreadsTransform::NormalThreadsTransform(int thNr, int chNr) {
    this->threadsNumber = thNr;
    this->chunksNumber = chNr;
}

vector<Mat> NormalThreadsTransform::processChunks(Mat grayscaleImage, Mat colorImage) {
    // Split the grayscale and color images into chunks
    vector<Mat> grayscaleChunks = splitImageInChunks(grayscaleImage, this->chunksNumber);
    vector<Mat> colorChunks = splitImageInChunks(colorImage, this->chunksNumber);

    // Process each chunk in a separate thread
    vector<thread> threads;
    vector<Mat> processedChunks;

    for (int i = 0; i < this->chunksNumber; ++i) {
        threads.emplace_back([&processedChunks, &grayscaleChunks, i]() {
            Mat processedChunk = houghLinesTransform(grayscaleChunks[i]); // Only using houghLinesTransform
            processedChunks.push_back(processedChunk);
            });
    }

    // Wait for all threads to complete
    for (thread& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    return processedChunks;
}

vector<vector<Mat>> NormalThreadsTransform::imageProcessing(vector<Mat> grayscaleChunks, vector<Mat> colorChunks) {
    vector<vector<Mat>> results(1); // Only one type of result due to one transformation type available

    // Parallel processing for Hough Lines Transform
    vector<thread> threads;

    threads.emplace_back([&]() {
        for (auto& chunk : grayscaleChunks) {
            results[0].push_back(houghLinesTransform(chunk));
        }
        });

    // Wait for the thread to complete
    for (thread& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    return results;
}

void NormalThreadsTransform::gogoBoys(string filename) {
    Mat grayscaleImage = imread(filename, IMREAD_GRAYSCALE);
    Mat colorImage = imread(filename, IMREAD_COLOR);

    if (grayscaleImage.empty() || colorImage.empty()) {
        cout << "Error opening image!" << endl;
        exit(-1);
    }

    auto startTime = chrono::high_resolution_clock::now();

    vector<Mat> grayscaleChunks = splitImageInChunks(grayscaleImage, this->chunksNumber);
    vector<Mat> colorChunks = splitImageInChunks(colorImage, this->chunksNumber);

    vector<vector<Mat>> results = imageProcessing(grayscaleChunks, colorChunks);

    Mat linesImage = concatenateChunksVertically(results[0]);

    imshow("source image", colorImage);
    imshow("detected lines", linesImage);

    auto endTime = chrono::high_resolution_clock::now();
    cout << "Processing time: " << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() << " ms" << endl;

    waitKey(0);
}
