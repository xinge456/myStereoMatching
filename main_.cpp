//#include "cost_calculating.h"
#include "stereoMatching.h"
#include "util.h"

#include <iostream>
//#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define DEBUG
using namespace std;
using namespace cv;

string StereoMatching::costcalculation = "grad";  //grad°¢gradCensus°¢TruncAD, ADCensusZNCC, SSD, Census, ZNCC, S-D, mean-Census, symmetric-Census, Census-CBCA, AD-CBCA(Cross-based Cost Aggregation)°¢ADCensus-CBCA°¢AWS°¢AWS-CBCA°¢BF°¢ADCensusZNCC
string StereoMatching::aggregation = "CBCA"; // CBCA°¢ASW°¢guideFilter 
string StereoMatching::optimization = "";
string StereoMatching::object = ""; 
const string StereoMatching::root = "D:\\experiment\\dataset\\middlebury\\"; 

int main(int argc, char* argv[]) {

	if (StereoMatching::Do_sgm) 
	{
		StereoMatching::optimization += "sgm";
	}
	const string method = StereoMatching::costcalculation + StereoMatching::aggregation+ StereoMatching::optimization;  // 
	printf_s("method: %s\n", method.c_str());

	float disp_reduceCoefficent = 4;
	int MAXDISP = 59;
	string dataset = "MD";
	string mdroot = StereoMatching::root;
	string all_maskN = "all.png";
	string nonocc_maskN = "nonocc.png";
	string disc_maskN = "disc.png";
	// ∂¡»°Õº∆¨
	// middleburry
	vector<string> objectList = { "tsukuba", "venus", "teddy", "cones", "Books", "Art", "Aloe", "Bowling1", "Adirondack" }; // 7, cones, teddy, shopvac, Books°¢venus°¢tsukuba
	vector<string> leftNameList = { "scene1.row3.col3", "im2", "im2", "im2", "view1", "view1", "view1", "view1", "im0"};
	vector<string> rightNameList = {"scene1.row3.col4", "im6", "im6", "im6", "view5", "view5", "view5", "view5", "im1"};
	vector<string> dispNameList = { "truedisp.row3.col3", "disp2", "disp2", "disp2", "disp1", "disp1", "disp1", "disp1", "disp_0"};

	vector<float> disp_reduceCoeffList = {16, 8, 4, 4, 3, 3, 3, 3, 1};
	vector<int> maxdispList = {15, 19, 59, 59, 85, 85, 85, 85, 248};
	int objLenth = objectList.size();
	for (int i = 0; i < 4; i++)
	{
		StereoMatching::object = objectList[i];
		string object = StereoMatching::object;
		printf("object: %s\n", object.c_str());
		string imgroot = mdroot + object + "\\";

		string leftimg = imgroot + leftNameList[i] + ".png";
		string rightimg = imgroot + rightNameList[i] + ".png";
		string img_disp = imgroot + dispNameList[i] + ".png";
		string all_mask = imgroot + all_maskN;
		string nonocc_mask = imgroot + nonocc_maskN;
		string disc_mask = imgroot + disc_maskN;

		cv::Mat I1 = cv::imread(leftimg, 0);  // ∂¡»Îª“∂»Õº
		cv::Mat I2 = cv::imread(rightimg, 0);
		cv::Mat I1_c = cv::imread(leftimg, 1);  // ∂¡»Î≤ …´Õº£®3channel£©
		cv::Mat I2_c = cv::imread(rightimg, 1);
		cv::Mat all_maskM = cv::imread(all_mask, 0);  // ∂¡»Î√…∞Ê
		cv::Mat nonocc_maskM = cv::imread(nonocc_mask, 0);
		cv::Mat disc_maskM = cv::imread(disc_mask, 0);
		cv::Mat DT = cv::imread(img_disp, 0);  // ∂¡»Î ”≤ÓÕº
		if (I1.empty() || I2.empty() || I1_c.empty() || I2_c.empty())
		{
			cout << "can't read original img" << endl;
			return -1;
		}
		if (all_maskM.empty() || nonocc_maskM.empty() || disc_maskM.empty())
		{
			cout << "can't read mask img" << endl;
			//return -1;
		}
		std::cout << "read-in img done" << endl;

		if (StereoMatching::preMedBlur)
		{
			cv::medianBlur(I1_c, I1_c, 3);
			cv::medianBlur(I2_c, I2_c, 3);
		}

		if (dataset == "KT")
			DT.convertTo(DT, CV_32F, 1.0 / 256);
		if (dataset == "MD")
			DT.convertTo(DT, CV_32F, 1.0 / disp_reduceCoeffList[i]);

		StereoMatching::Parameters param(maxdispList[i], I1_c.rows, I1_c.cols);
		StereoMatching sm(I1_c, I2_c, I1, I2, DT, all_maskM, nonocc_maskM, disc_maskM, param);
		sm.showParams();
		sm.pipeline();
		cout << "complete " << object << endl;
		cout << "*********************" << endl;
		cout << "*********************" << endl;
	}
	return 0;
}