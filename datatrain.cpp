// datatrain.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

int minIndex(double cost[], int n)
{
	double min = cost[0];
	int d = 0;
	for (int i = 1; i < n; i++)
	{
		if (cost[i] < min)
		{
			min = cost[i];
			d = i;
		}
	}
	cout << "min:" << min << endl;
	return d;
}
int _tmain(int argc, _TCHAR* argv[])
{
	ifstream fin("..\\datatruth.txt");
	FileStorage featureMat("..\\featureMat.xml", CV_STORAGE_WRITE);
	FileStorage truthMat("..\\truthMat.xml", CV_STORAGE_WRITE);
	string fileName;
	vector<cv::Mat>desvector;
	int total = 0;
	while (getline(fin,fileName))
	{
		Mat img = imread(fileName);
		Size dsize(0, 0);
		int width = img.cols;
		double fx = 400 / width;
		Mat imgNor;
		double fy = fx;
		resize(img, imgNor, dsize, fx, fy, INTER_CUBIC);

		vector<cv::KeyPoint>keypoints;
		SurfFeatureDetector surf(400);
		surf.detect(imgNor, keypoints);
		cout << keypoints.size() << endl;
		total += keypoints.size();
		SurfDescriptorExtractor extractor;
		Mat descriptors;
		extractor.compute(imgNor, keypoints, descriptors);
		desvector.push_back(descriptors);
	}
	Mat feature(total, 64, CV_32F);
	int n = 0;
	while( n < total)
	{
		for (int i = 0; i < desvector.size(); i++)
		{
			Mat temp = desvector.at(i);
			for (int y = 0; y < temp.rows; y++)
			{
				for (int j = 0; j < 64; j++)
				{
					feature.at<float>(n, j) = temp.at<float>(y, j);
				}
				n++;
			}
			temp.release();
		}
	}
	cout << "total:" << total << endl;
	Mat bestlabels, centers;
	const int k = 200;
	kmeans(feature, k, bestlabels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0e-6), 3, KMEANS_RANDOM_CENTERS, centers);
	cout << centers.size() << endl;
	featureMat << "featureMat" << centers;
	int imgCount = desvector.size();
	Mat truthImgFeature(imgCount, k, CV_8UC1, Scalar::all(0));
	for (int i = 0; i < imgCount; i++)
	{
		Mat imgFeature = desvector.at(i);
		for (int y = 0; y < imgFeature.rows; y++)
		{
			double cost[k] = { 0 };
			for (int ki = 0; ki < k; ki++)
			{
				for (int d = 0; d < 64; d++)
				{
					cost[ki] = cost[ki] + pow(imgFeature.at<float>(y, d) - centers.at<float>(ki, d), 2);
				}
			}
			cout << "image" << i << ":\n";
			truthImgFeature.at<uchar>(i, minIndex(cost, k))++;
		}
	}
	truthMat << "truthMat" << truthImgFeature;
	system("pause");
	return 0;
}

