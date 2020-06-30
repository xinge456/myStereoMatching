#pragma once
//#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include<string>
#include<iostream>
#include<bitset>
using namespace std;
using namespace cv;
//
// Re compute cost using NLC's implementation
//
// #define RE_COMPUTE_COST

//
// Non-local Cost Aggregatation
//
class NLCCA 
{
public:
	NLCCA(void)
	{
		printf( "\n\t\tNon-local method for cost aggregation" );
	}

	~NLCCA(void) {}
public:
	void aggreCV( const Mat& lImg, const Mat& rImg, const int maxDis, Mat& costVol );
};


