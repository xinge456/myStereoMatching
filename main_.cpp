//#include "cost_calculating.h"
#include "stereoMatching.h"
#include "util.h"

#include <iostream>
//#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//#define DEBUG
using namespace std;
using namespace cv;

string StereoMatching::costcalculation = "censusGrad";  //censusGrad,BT, grad°¢TruncAD, ADCensusZNCC, SSD, Census, ZNCC, S-D, mean-Census, symmetric-Census, Census-CBCA, AD-CBCA(Cross-based Cost Aggregation)°¢ADCensus-CBCA°¢AWS°¢AWS-CBCA°¢BF°¢ADCensusZNCC
string StereoMatching::aggregation = "CBCA"; // CBCA°¢ASW°¢GF°¢FIF°¢NL°¢BF°¢GFNL
string StereoMatching::optimization = "sgm"; // sgm, so
string StereoMatching::object = ""; 
const string StereoMatching::root = "D:\\experiment\\dataset\\middlebury\\"; 

int main(int argc, char* argv[]) {

	const string method = StereoMatching::costcalculation + StereoMatching::aggregation+ StereoMatching::optimization;  // 
	printf_s("method: %s\n", method.c_str());

	string dataset = "MD";
	string mdroot = StereoMatching::root;
	string all_maskN = "all.png";
	string nonocc_maskN = "nonocc.png";
	string disc_maskN = "disc.png";
	// ∂¡»°Õº∆¨
	// middleburry
	vector<string> objectList = {"tsukuba", "venus", "teddy", "cones", "Art", "Books", "Dolls", "Laundry", "Moebius", "Reindeer", "Aloe", "Baby1", "Baby2", "Baby3", "Bowling1", "Bowling2", "Cloth1", "Cloth2", "Cloth3", "Cloth4", "Flowerpots", "Lampshade1", "Lampshade2", "Midd1", "Midd2", "Monopoly", "Plastic", "Rocks1", "Rocks2", "Wood1", "Wood2", "Katzaa", "Michmoret"}; // 7, cones, teddy, shopvac, Books°¢venus°¢tsukuba
	vector<string> leftNameList = { "scene1.row3.col3", "im2", "im2", "im2", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "view1", "left_matlab_valid_resize", "left_matlab_valid_resize"};
	vector<string> rightNameList = {"scene1.row3.col4", "im6", "im6", "im6", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "view5", "right_matlab_valid_resize", "right_matlab_valid_resize"};
	vector<string> dispNameList = { "truedisp.row3.col3", "disp2", "disp2", "disp2", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "disp1", "all", "all"};

	vector<float> disp_reduceCoeffList = {16, 8, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5};
	vector<int> maxdispList = {15, 19, 59, 59, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 80, 80};
	// num 33
	int objLenth = objectList.size();
	CV_Assert(objLenth == leftNameList.size());
	CV_Assert(objLenth == rightNameList.size());
	CV_Assert(objLenth == dispNameList.size());
	CV_Assert(objLenth == disp_reduceCoeffList.size());
	CV_Assert(objLenth == maxdispList.size());


	// Baby2_25,Reindeer_13,Wood2_23
	string errCsvName = "20200625Censustest.csv";
	ofstream ofs;
	string paramAddr = StereoMatching::root + errCsvName;

	//ofs << method;
	//ofs << ",";
	//for (int i = 0; i < 4; i++)
	//	ofs << objectList[i] << ",";
	//ofs << endl;

	int lamG = 1;
	int lamCen = 13;
	int M = 2;
	int lamc = 109;
	int ts = 10;
	ofs.open(paramAddr, ios::out | ios::app);
	ofs << endl;
	ofs.close();

	// Teddy£¨Cones£¨Books£¨Reindeer£¨Rocks1£¨Wood2£¨Baby2£¨Bowling1,
	//vector<string> m_n = {"Teddy", "Cones", "Books", "Reindeer", "Rocks1", "Wood2", "Baby2", "Bowling1" };
	vector<string> m_n = {"Aloe"};
	// num 33
	for (ts = 10; ts <= 10; ts++)
	{
		for (int i = 0; i < 31; i++)
		{
			if (std::find(m_n.begin(), m_n.end(), objectList[i]) == m_n.end())
				continue;

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

			cv::Mat I1_c = cv::imread(leftimg, 1);  // ∂¡»Î≤ …´Õº£®3channel£©
			cv::Mat I2_c = cv::imread(rightimg, 1);

			cv::Mat I1 = cv::imread(leftimg, 0);  // ∂¡»Îª“∂»Õº
			cv::Mat I2 = cv::imread(rightimg, 0);
			//Mat I1, I2;
			//Mat tem;
			//cvtColor(I1_c, tem, CV_BGR2RGB);
			//cvtColor(tem, I1, CV_RGB2GRAY);
			//cvtColor(I2_c, tem, CV_BGR2RGB);
			//cvtColor(tem, I2, CV_RGB2GRAY);

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

			int disSc = 1;
			int PY_LEV = 1;
			StereoMatching** smPsy = new StereoMatching * [PY_LEV];
			clock_t start = clock();
			for (int p = 0; p < PY_LEV; p++)
			{
				printf("\n\tPyramid: %d:", p);
				StereoMatching::Parameters param(maxdispList[i], I1_c.rows, I1_c.cols, lamCen, lamG, M, lamc, ts, errCsvName, disSc);
				smPsy[p] = new StereoMatching(I1_c, I2_c, I1, I2, DT, all_maskM, nonocc_maskM, disc_maskM, param);
				smPsy[p]->costCalculate();
				//smPsy[p]->dispOptimize();

				maxdispList[i] = maxdispList[i] / 2 + 1;
				disSc *= 2;
				pyrDown(I1_c, I1_c);
				pyrDown(I2_c, I2_c);
				pyrDown(I1, I1);
				pyrDown(I2, I2);
				pyrDown(DT, DT);
				pyrDown(all_maskM, all_maskM);
				if (nonocc_maskM.data)
					pyrDown(nonocc_maskM, nonocc_maskM);
				if (disc_maskM.data)
					pyrDown(disc_maskM, disc_maskM);
			}
			const auto t1 = std::chrono::system_clock::now();
			float REG_LAMBDA = 0.3; // 0.3 for middlebury
			SolveAll(smPsy, PY_LEV, REG_LAMBDA);
			smPsy[0]->openCSV();

			//smPsy[0]->clearErrTxt();  // «Âø’ŒÛ≤Ótxt
			//smPsy[0]->clearTimeTxt();  // «Âø’ ±º‰txt
			smPsy[0]->dispOptimize();
			
			if (StereoMatching::Do_refine)
				smPsy[0]->refine();

			smPsy[0]->closeCSV();
			clock_t end = clock();
			clock_t time = end - start;
			smPsy[0]->saveTime(time, "all");
			cout << "all Time: " << time << endl;
			for (int p = 0; p < PY_LEV; p++)
			{
				delete smPsy[p];
				smPsy[p] = NULL;
			}
			delete[] smPsy;
			smPsy = NULL;
			//StereoMatching sm(I1_c, I2_c, I1, I2, DT, all_maskM, nonocc_maskM, disc_maskM, param);
			////sm.showParams();
			//sm.pipeline();
			cout << "complete " << object << endl;
			cout << "*********************" << endl;
			cout << "*********************" << endl;

			//ofs.open(paramAddr, ios::out | ios::app);
			//ofs << endl;
			//ofs.close();
			
		}

	}
	return 0;
}