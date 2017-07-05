// ImageSearch.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "math.h"

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
vector<int> kmin(double cost[], int k, int n)
{
	vector<int> result;
	double max = cost[0];
	for (int i = 1; i < k; i++)
	{
		if (cost[i]>max)
			max = cost[i];
	}
	for (int i = 0; i < n; i++)
	{
		int d = minIndex(cost, k);
		result.push_back(d);
		cost[d] = max + 1;
	}
	return result;
}

vector<float> getColorSpace(Mat image)//RGB色彩空间的（4，4，4）模型。  
{
	const int div = 64;
	const int bin_num = 256 / div;
	const int length = bin_num*bin_num*bin_num;
	int nr = image.rows; // number of rows  
	int nc = image.cols; // number of columns  
	if (image.isContinuous())  {
		// then no padded pixels    
		nc = nc*nr;
		nr = 1;  // it is now a 1D array    
	}
	int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0));// mask used to round the pixel value  
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0  
	int b, g, r;
	vector<float> bin_hist;
	int ord = 0;
	float a[length] = { 0 };
	for (int j = 0; j < nr; j++) {
		const uchar* idata = image.ptr<uchar>(j);
		for (int i = 0; i < nc; i++) {
			b = ((*idata++)&mask) / div;
			g = ((*idata++)&mask) / div;
			r = ((*idata++)&mask) / div;
			ord = b * bin_num * bin_num + g * bin_num + r;
			a[ord] += 1;
		} // end of row        
	}
	float sum = 0;
	for (int i = 0; i < length; i++)
	{
		sum += a[i];
	}
	for (int i = 0; i < length; i++)//归一化  
	{
		a[i] = a[i] / sum;
		bin_hist.push_back(a[i]);
	}
	return bin_hist;
}

float getColorBaDis(Mat img1, Mat img2){
	vector<float> m1 = getColorSpace(img1);
	vector<float> m2 = getColorSpace(img2);
	float temp = 0;
	for (int i = 0; i < m1.size(); i++)
	{
		temp += sqrt(m1.at(i) * m2.at(i));
	}
	float result = -log(temp);
	cout << result << endl;
	return result;
}

int _tmain(int argc, _TCHAR* argv[])
{
	string filePath = "..\\testImage\\";
	int imgNum;
	cout << "imgNum:";
	cin >> imgNum;
	Mat imgInput = imread(filePath+"img_"+to_string(imgNum)+".jpg"); 
  
	Mat img;
  Size dsize(0, 0);
	int width = imgInput.cols;
	double fx = 400 / width;
	double fy = fx;
	resize(imgInput, img, dsize, fx, fy);

	//////////get truth image//////////
	ifstream fin("..\\datatruth\\datatruth.txt");
	vector<string> truthImage;
	string tempImg;
	while (getline(fin, tempImg))
		truthImage.push_back(tempImg);
	/////////////////////////////////

	vector<cv::KeyPoint>keypoints;
	SurfFeatureDetector surf(400);
	surf.detect(img, keypoints);
	Mat featureImage;
	drawKeypoints(img, keypoints, featureImage, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	SurfDescriptorExtractor extractor;
	Mat descriptors;
	extractor.compute(img, keypoints, descriptors);
	imshow("image", imgInput);
	imshow("featureImage", featureImage);
	imwrite("..\\featureImage\\" + to_string(imgNum) + ".jpg", featureImage);
	cout << keypoints.size() << endl;
	cout << descriptors.size() << endl;
	cout << descriptors.type() << endl;

	FileStorage featureMat("..\\featureMat.xml", CV_STORAGE_READ);
	FileStorage truthMat("..\\truthMat.xml", CV_STORAGE_READ);
	Mat feature_truth, descrip_truth;
	featureMat["featureMat"] >> feature_truth;
	truthMat["truthMat"] >> descrip_truth;
	cout << descrip_truth.size()<<endl;
	const int k = 200;
	Mat imgDes(1, descrip_truth.cols, CV_8UC1, Scalar::all(0));
	for (int y = 0; y < descriptors.rows; y++)
	{
		double cost[k] = { 0 };
		for (int ki = 0; ki < k; ki++)
		{
			for (int d = 0; d < 64; d++)
			{
				//cout << descriptors.at<float>(y, d);
				//cout << " " << feature_truth.at<float>(ki, d) << endl;;
				cost[ki] = cost[ki] + pow(descriptors.at<float>(y, d) - feature_truth.at<float>(ki, d), 2);
			}
		}
		imgDes.at<uchar>(0, minIndex(cost, k))++;
	}

	float lamda = 0.6;
	const int truthCount = descrip_truth.rows;
	double cos[100] = { 0 };
	for (int r = 0; r < truthCount; r++)
	{
		float a=0, b=0, c=0;
		for (int col = 0; col < descrip_truth.cols; col++)
		{
			c = c + imgDes.at<uchar>(0, col)*descrip_truth.at<uchar>(r, col);
			a = a + imgDes.at<uchar>(0, col)*imgDes.at<uchar>(0, col);
			b = b + descrip_truth.at<uchar>(r, col)*descrip_truth.at<uchar>(r, col);
		}
    
		Mat img2 = imread(truthImage.at(r));
		cos[r] = lamda* (1 - c / sqrt(a*b)) + (1 - lamda) * getColorBaDis(imgInput, img2);
	} 
  
	int top = 5;
	vector<int> topk = kmin(cos, truthCount, top);
	for (int i = 0; i < top; i++)
	{
		int num = topk.at(i);
		string resultName = "..\\datatruth\\" + to_string(num + 1) + ".jpg";
		Mat result = imread(resultName);
		string winName = "result" + to_string(i);
		imshow(winName, result);
	}
	
	waitKey(0);
	return 0;
}

