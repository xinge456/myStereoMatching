
#include <algorithm>
//#include <cmath>
#include <omp.h>
//#include <iostream>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "stereoMatching.h"
#include "util.h"

#ifdef _WIN32
#define popcnt32 __popcnt
#define popcnt64 __popcnt64
#else
#define popcnt32 __builtin_popcount
#define popcnt64 __builtin_popcountll
#endif

#define USE_OPENMP
#if defined(_OPENMP) && defined(USE_OPENMP)
#ifdef _WIN32
#define OMP_PARALLEL_FOR __pragma(omp parallel for)
#else
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#endif
#else
#define OMP_PARALLEL_FOR
#endif

#define DEBUG

using namespace std;

void StereoMatching::gradCensus(vector<Mat>& vm)
{
	vector<Mat> gradVm(2);
	vector<Mat> censusVm(2);
	for (int i = 0; i < 2; i++)
	{
		gradVm[i].create(3, size_vm, CV_32F);
		censusVm[i].create(3, size_vm, CV_32F);
	}
	grad(gradVm);
	censusCal(censusVm);
	int img_num = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < img_num; i++)
		gen_vm_from2vm_exp(vm[i], gradVm[i], censusVm[i], 20, 30, i);
	saveFromVm(vm, "gradCensus");
}

void StereoMatching::adGrad(vector<Mat>& vm)
{
	vector<Mat> adVm(2);
	vector<Mat> gradVm(2);
	for (int i = 0; i < 2; i++)
	{
		adVm[i].create(3, size_vm, CV_32F);
		gradVm[i].create(3, size_vm, CV_32F);
	}
	for (int i = 0; i < 2; i++)
		gen_ad_sd_vm(adVm[i], i, 0);
	grad(gradVm);
	for (int i = 0; i < 2; i++)
		gen_vm_from2vm_add(vm[i], adVm[i], gradVm[i], 20, 30, i);
	saveFromVm(vm, "gradCensus");
}

void StereoMatching::asdCal(vector<Mat>& vm_asd, string method, int imgNum)
{
	cout << "\n" << endl;
	cout << "start " << method << " calculation" << endl;
	clock_t start = clock();
	int type = method == "AD" ? 0 : 1;
	for (int i = 0; i < imgNum; i++)
		gen_ad_sd_vm(vm_asd[i], i, type);
	clock_t end = clock();
	clock_t time = end - start;
	cout << time << endl;
#ifdef DEBUG
	saveTime(time, method);
	saveFromVm(vm_asd, method);
#endif
	cout << "AD vm generated" << endl;
}

void StereoMatching::calGrad(Mat& grad, Mat& img)
{
	for (int v = 0; v < h_; v++)
	{
		uchar* iP = img.ptr<uchar>(v);
		float* gP = grad.ptr<float>(v);
		for (int u = 0; u < w_; u++)
		{
			if (u > 0 && u < w_ - 1)
				gP[u] = ((float)iP[u + 1] - iP[u - 1]) / 2;
			else if (u == 0)
				gP[u] = iP[1] - iP[0];
			else
				gP[u] = iP[w_ - 1] - iP[w_ - 2];
			//gP[u] = abs(iP[u + 1] - iP[u]) + abs(img.ptr<uchar>(v + 1)[u] - iP[u]);
			//gP[u] = sqrt(pow(iP[u + 1] - iP[u], 2) + pow(img.ptr<uchar>(v + 1)[u] - iP[u], 2));
		}
	}
}

void StereoMatching::calGrad_y(Mat& grad, Mat& img)
{
	for (int v = 0; v < h_; v++)
	{
		uchar* iP = img.ptr<uchar>(v);
		float* gP = grad.ptr<float>(v);
		for (int u = 0; u < w_; u++)
		{
			if (v > 0 && v < w_ - 1)
				gP[u] = ((float)img.ptr<uchar>(v + 1)[u] - img.ptr<uchar>(v - 1)[u]) / 2;
			else if (v == 0)
				gP[u] = (float)img.ptr<uchar>(1)[u] - img.ptr<uchar>(0)[u];
			else
				gP[u] = (float)img.ptr<uchar>(h_ - 1)[u] - img.ptr<uchar>(h_ - 2)[u];
			//gP[u] = abs(iP[u + 1] - iP[u]) + abs(img.ptr<uchar>(v + 1)[u] - iP[u]);
			//gP[u] = sqrt(pow(iP[u + 1] - iP[u], 2) + pow(img.ptr<uchar>(v + 1)[u] - iP[u], 2));
		}
	}
}

void StereoMatching::calgradvm(Mat& vm, vector<Mat>& grad, vector<Mat>& grad_y, int num)
{
	CV_Assert(grad[0].depth() == CV_32F);
	const int n = param_.numDisparities;
	const float costD = 500;
	int leftCoe = 0, rightCoe = -1;
	if (num == 1)
		leftCoe = 1, rightCoe = 0;
	for (int v = 0; v < h_; v++)
	{
		float* grad0P = grad[0].ptr<float>(v);
		float* grad1P = grad[1].ptr<float>(v);
		float* grad_y0P = grad_y[0].ptr<float>(v);
		float* grad_y1P = grad_y[1].ptr<float>(v);
		for (int u = 0; u < w_; u++)
		{
			float* vP = vm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				int u0 = u + leftCoe * d;
				int u1 = u + rightCoe * d;
				if (u0 >= w_ || u1 < 0)
					vP[d] = costD;
				else
					vP[d] = (abs(grad0P[u0] - grad1P[u1]) + abs(grad_y0P[u0] - grad_y1P[u1])) / 2;
			}
		}
	}
}

void StereoMatching::calgradvm_1d(Mat& vm, vector<Mat>& grad, int num)
{
	CV_Assert(grad[0].depth() == CV_32F);
	const int n = param_.numDisparities;
	const float costD = 500;
	int leftCoe = 0, rightCoe = -1;
	if (num == 1)
		leftCoe = 1, rightCoe = 0;
	for (int v = 0; v < h_; v++)
	{
		float* grad0P = grad[0].ptr<float>(v);
		float* grad1P = grad[1].ptr<float>(v);

		for (int u = 0; u < w_; u++)
		{
			float* vP = vm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				int u0 = u + leftCoe * d;
				int u1 = u + rightCoe * d;
				if (u0 >= w_ || u1 < 0)
					vP[d] = costD;
				else
					vP[d] = abs(grad0P[u0] - grad1P[u1]);
			}
		}
	}
}

void StereoMatching::saveFromVm(vector<Mat> vm, string name)
{
	Mat dispMap(h_, w_, CV_16S);
	gen_dispFromVm(vm[0], dispMap);
	saveFromDisp<short>(dispMap, name);
	if (Do_LRConsis)
	{
		gen_dispFromVm(vm[1], dispMap);
		saveFromDisp<short, 1>(dispMap, name);
	}
}

template <typename T, int LOR>
void StereoMatching::saveFromDisp(Mat disp, string name)
{
	string img_num = to_string(img_counting);
	if (LOR == 0)
	{
		saveDispMap<T>(disp, "d" + img_num + "isp_" + name + "LR");
		saveBiary<short>("b" + img_num + "iary_" + name, disp, DT);
		calErr<short>(disp, DT, name);
		img_counting++;
	}
	else
		saveDispMap<T>(disp, "d" + to_string(img_counting - 1) + "isp_" + name + "RL");

}

void StereoMatching::grad(vector<Mat>& vm_grad)
{
	// 计算梯度
	vector<Mat> grad(2);
	vector<Mat> grad_y(2);
	for (int i = 0; i < 2; i++)
	{
		grad[i].create(h_, w_, CV_32F);
		grad_y[i].create(h_, w_, CV_32F);
		//Sobel(I_g[i], grad[i], CV_32F, 1, 0);
		//Sobel(I_g[i], grad_y[i], CV_32F, 0, 1);
		calGrad(grad[i], I_g[i]);
		//calGrad_y(grad_y[i], I_g[i]);
		saveDispMap<float>(grad[i], to_string(i) + "grad");
		//saveDispMap<float>(grad_y[i], to_string(i) + "grad_y");
	}

	// 计算代价卷
	int imgNum = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < imgNum; i++)
		//calgradvm(vm_grad[i], grad, grad_y, i);
		calgradvm_1d(vm_grad[i], grad, i);

	saveFromVm(vm_grad, "grad");
}

void StereoMatching::truncAD(vector<Mat>& vm)
{
	cout << "\n" << endl;
	cout << "start " << costcalculation << " calculation" << endl;
	clock_t start = clock();
	int imgNum = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < imgNum; i++)
		gen_truncAD_vm(vm[i], i);
	clock_t end = clock();
	clock_t time = end - start;
	cout << time << endl;
	saveTime(time, costcalculation);

#ifdef DEBUG
	saveFromVm(vm, "truncAD");
#endif
	cout << "truncAD vm generated" << endl;
}

void StereoMatching::censusCal(vector<Mat>& vm_census)
{
	cout << "start censusCal" << endl;
	clock_t start = clock();
	vector<Mat> censusCode(2);
	int imgNum = Do_LRConsis ? 2 : 1;
	Mat dispMap(h_, w_, CV_16S);
	int channels = param_.census_channel;  // 1
	int codeLength = (param_.W_U * 2 + 1) * (param_.W_V * 2 + 1) * channels;
	int varNum = ceil(codeLength / 64.);
	int size_census[] = { h_, w_, varNum };
	censusCode[0].create(3, size_census, CV_64F);
	censusCode[1].create(3, size_census, CV_64F);

	//gen_cen_vm_<0>(vm_census[0]);
	//gen_cen_vm_<1>(vm_census[1]);
	if (channels == 3)
		genCensusCode(I_c, censusCode, param_.W_V, param_.W_U);
	else
	{
		genCensus(I_g[0], censusCode[0], param_.W_V, param_.W_U);
		genCensus(I_g[1], censusCode[1], param_.W_V, param_.W_U); //编码0
		//genCensusCode(I_g, censusCode, param_.W_V, param_.W_U);  // 编码1
	}

	for (int i = 0; i < imgNum; i++)
		//gen_cenVM_XOR(censusCode, vm_census[i], i);  // 0
		genCensusVm(censusCode, vm_census[i], i);  // 1

	cout << "finish census cal" << endl;
	clock_t end = clock();
	clock_t time = end - start;
	saveTime(time, "Census");
#ifdef DEBUG 
	saveFromVm(vm_census, "census");
#endif
}

void StereoMatching::ADCensusCal()
{
	cout << "start ADCensus cal" << endl;
	clock_t start = clock();
	int imgNum = Do_LRConsis ? 2 : 1;
	vector<Mat> vm_asd(imgNum), vm_census(imgNum);
	for (int num = 0; num < imgNum; num++)
	{
		vm_asd[num].create(3, size_vm, CV_32F);
		vm_census[num].create(3, size_vm, CV_32F);
	}
	asdCal(vm_asd, "AD", imgNum);
	//BF(vm_asd, DT);
	censusCal(vm_census);
	adCensus(vm_asd, vm_census);
	cout << "finish ADCensus cal" << endl;
	clock_t end = clock();
	clock_t time = end - start;
	cout << "ADCensus time：" << time << endl;
	saveTime(time, "ADCensus");
}

void StereoMatching::costCalculate()
{
	string::size_type idx;
	string::size_type idy;
	int imgNum = Do_LRConsis ? 2 : 1;
	// AD、SD
	if (costcalculation == "AD" || costcalculation == "SD")
		asdCal(vm, costcalculation, imgNum);

	else if (costcalculation == "grad")
		grad(vm);

	else if (costcalculation == "TruncAD")
		truncAD(vm);

	else if (costcalculation == "adGrad")
		adGrad(vm);

	else if (costcalculation == "gradCensus")
		gradCensus(vm);

	// census
	else if (costcalculation == "Census")
		censusCal(vm);

	// ZNCC
	else if (costcalculation == "ZNCC")
		ZNCC(I_g[0], I_g[1], vm);		

	// adCensus
	else if (costcalculation == "ADCensus")
		ADCensusCal();

	// BF 盒式滤波
	idx = aggregation.find("BF");
	if (idx != string::npos)
		BF(vm);

	//AWS
	idx = aggregation.find("AWS");
	if (idx != string::npos)
		AWS();

	// cost aggregation
	//CBCA
	idx = aggregation.find("CBCA");
	if (idx != string::npos)
		CBCA();

	idx = aggregation.find("guideFilter");
	if (idx != string::npos)
		guideFilter();


	cout << "costCalculate finished" << endl;
	cout << endl;
}

void StereoMatching::BF(vector<Mat>& vm)
{
	int imgNum = Do_LRConsis ? 2 : 1;
	const int W_V_B = param_.cbca_box_WV;  // 3, 2
	const int W_U_B = param_.cbca_box_WU; // 4, 3
	const int vm_h = h_ + W_V_B * 2;
	const int vm_w = w_ + W_U_B * 2;
	const int vm_n = param_.numDisparities;
	const int size_vm_[] = { vm_h, vm_w, vm_n };
	Mat vm_borderIp(3, size_vm_, CV_32F);  // cbca and block combine

	for (int i = 0; i < imgNum; i++)
	{
		symmetry_borderCopy_3D(vm[i], vm_borderIp, W_V_B, W_V_B, W_U_B, W_U_B);  // 
		BF_BI(vm_borderIp, vm[i], W_V_B, W_U_B);
	}
#ifdef DEBUG
	saveFromVm(vm, "BF");
#endif // DEBUG
}

void StereoMatching::dispOptimize()
{
	// 视差优化
	if (Do_sgm)
	{
		bool leftFirst = false;
		int loopNum = 1;
		if (Do_LRConsis)
			loopNum = 2;
		for (int i = 0; i < loopNum; i++)
		{
			clock_t start = clock();
			leftFirst = !leftFirst;
			sgm(vm[i], leftFirst);
			clock_t end = clock();
			clock_t time = end - time;
			saveTime(time, "sgm" + to_string(i));
		}
	}
	DP[0].create(h_, w_, CV_16S);
	DP[1].create(h_, w_, CV_16S);

	int img_num = Do_LRConsis ? 2 : 1;
	if (param_.Do_vmTop)
	{
		int sizeVmTop[] = { h_, w_, param_.vmTop_Num + 1, 2 };
		Mat topDisp(4, sizeVmTop, CV_32F);
		for (int i = 0; i < img_num; i++)
		{
			Mat vm_copy = vm[i].clone();
			selectTopCostFromVolumn(vm_copy, topDisp, param_.vmTop_thres, param_.vmTop_Num);
			genDispFromTopCostVm2(topDisp, DP[i]);
		}

	}
	else
		for (int i = 0; i < img_num; i++)
			gen_dispFromVm(vm[i], DP[i]);

	string name = Do_sgm ? "so" : "wta";
#ifdef DEBUG
	saveFromDisp<short, 0>(DP[0], name);
	saveFromDisp<short, 1>(DP[1], name);
#endif // DEBUG
	if (Do_sgm)
		cout << "scanline optimization finished" << endl << endl;
	else
		cout << "wta finished" << endl << endl;
}

void StereoMatching::refine()
{
	// 视差细化
	if (StereoMatching::Do_LRConsis)  // LR consisency check
	{
		clock_t start = clock();
		LRConsistencyCheck(DP[0], DP[1]);
		clock_t end = clock();
		clock_t time = end - start;
#ifdef DEBUG
		saveTime(time, "LRC");
#endif

	}
#ifdef DEBUG
	saveFromDisp<short>(DP[0], "od");
#endif
	std::cout << "LRConsistencyCheck finished" << endl;

	string filename_P;
	Mat dispChange(h_, w_, CV_16S);

	Mat Dp_res;

	if (StereoMatching::Do_regionVote)
	{
		Dp_res = DP[0].clone();
		string::size_type idx;
		idx = aggregation.find("CBCA");
		if (idx == string::npos)  // 如果没有执行CBCA，则此处需单独生成HVL[0]
		{
			const uchar L0 = param_.cbca_crossL[0];  // 17
			const uchar L_out0 = param_.cbca_crossL_out[0];
			const uchar C_D = param_.Cross_C;  // 20
			const uchar C_D_out = 6;
			const uchar minL = param_.cbca_minArmL;  // 1
			HVL.resize(2);
			int HVL_size[] = { h_, w_, 5 };
			HVL[0].create(3, HVL_size, CV_16U);
			int channels = cbca_genArm_isColor ? 3 : 1;
			calHorVerDis(0, channels, L0, L_out0, C_D, C_D_out, minL);
		}

		float rv_ratio[] = {0.4, 0.4, 0.4, 0.4};
		//float rv_ratio[] = {0.4, 0.5, 0.6, 0.7};
		//float rv_ratio[] = {0.7, 0.6, 0.5, 0.4};
		int rv_s[] = {20, 20, 20, 20};
		for (int i = 0; i < param_.region_vote_nums; i++)  // region vote
		{
			clock_t start = clock();
			RV_combine_BG(DP[0], rv_ratio[i], rv_s[i]);
			//signDispChange_forRV(Dp_res, DP[0], DT, I_mask[0], dispChange);
			//filename_P = "dispRVChange" + to_string(i + 1);
			//saveDispMap<short>(dispChange, filename_P);
			clock_t end = clock();
			clock_t time = end - start;
			cout << "RV" + to_string(i) << " time" << time << endl;
#ifdef DEBUG
			string name = "RV" + to_string(i);
			saveTime(time, name);
			saveFromDisp<short>(DP[0], name);
#endif
			cout << "region_vote iteration:" + to_string(i) + " finished";
		}
		//coutInterpolaterEffect(Dp_res, DP[0]);
		//Dp_res = DP[0].clone();
	}

	if (StereoMatching::Do_cbbi) //cut-based border interpolate  
	{
		//Mat Dp_res = DP[0].clone();
		cbbi(DP[0]);
		signDispChange_forRV(Dp_res, DP[0], DT, I_mask[0], dispChange);
		saveDispMap<short>(dispChange, "dispCBBIChange.png");
		saveFromDisp<short>(DP[0], "CBBI");
	}

	if (StereoMatching::Do_bgIpol)
	{
		Dp_res = DP[0].clone();
		for (int i = 0; i < param_.region_vote_nums; i++)  // region vote
			BGIpol(DP[0]);
		coutInterpolaterEffect(Dp_res, DP[0]);
	}

	if (StereoMatching::Do_properIpol)
	{
		Dp_res = DP[0].clone();
		clock_t start = clock();
		for (int i = 0; i < param_.region_vote_nums; i++)  // region vote
			properIpol(DP[0], I_c[0]);  // proper interpolation
		clock_t end = clock();
		clock_t time = end - start;
		cout << "PI time: " << time << endl;
#ifdef DEBUG
		saveTime(time, "PI");
		saveFromDisp<short>(DP[0], "pi");
#endif // DEBUG
		cout << "properInterpolation finished" << endl;
		//coutInterpolaterEffect(Dp_res, DP[0]);
	}

	if (StereoMatching::Do_discontinuityAdjust)
	{
		discontinuityAdjust(DP[0]);  // discontinutity adjustment
#ifdef DEBUG
		saveFromDisp<short>(DP[0], "da");
#endif // DEBUG
		cout << "discontinutyAdjustment finished" << endl;
	}

	if (StereoMatching::Do_subpixelEnhancement)
	{
		Mat SE(h_, w_, CV_32F);
		subpixelEnhancement(DP[0], SE);  // subpixel enhancement
#ifdef DEBUG
		saveFromDisp<float>(DP[0], "se");
#endif // DEBUG
		cout << "subpixelEnhancement finished" << endl;
		medianBlur(SE, SE, 3);
#ifdef DEBUG
		saveFromDisp<float>(DP[0], "mb");
#endif
		cout << "medianBlur finished" << endl;
	}

	if (StereoMatching::Do_lastMedianBlur)
	{
		medianBlur(DP[0], DP[0], 3);
#ifdef DEBUG
		saveFromDisp<short>(DP[0], "mb");
#endif
		cout << "medianBlur finished" << endl;
	}

	cout << "disparity refine finished" << endl;
	cout << endl;
}

// 算法描述见论文
void StereoMatching::genDispFromTopCostVm2(Mat& topDisp, Mat& disp)
{
	clock_t start = clock();
	clock_t time_sum = 0, time_sum2 = 0, time_sum3 = 0, time_sum4 = 0;
	const int disp_DifThres = 10;
	const int disp_DifThres2 = 14;
	const int disp_DifThres3 = 5;
	const int num_candi = topDisp.size[2];  // 设定的候选视差数加一个实际候选视差指示位
	map<float, int> dispCost_container;
	map<float, int>::iterator iter_dispCost;
	map<int, int> disp__num;
	map<int, int>::iterator iter_disp_num;
	map<int, float> disp__cost;
	map<int, float>::iterator iter_disp_cost;
	unsigned __int64 step = 0;
	unsigned __int64 step2 = 0;
	unsigned __int64 mapsize = 0;

	int neigh_v[] = {0, 0, 1, -1};
	int neigh_u[] = {-1, 1, 0, 0};
	for (int v = 0; v < h_; v++)
	{
		short* disP = disp.ptr<short>(v);
		for (int u = 0; u < w_; u++)
		{
			if (u == 0 || v == 0)
				disP[u] = topDisp.ptr<float>(v, u, 0)[0];
			else
			{
				int n = topDisp.ptr<float>(v, u, num_candi - 1)[0];
				if (n == 1)
					disP[u] = topDisp.ptr<float>(v, u, 0)[0];
				else if (n > 1)
				{
					int disp_pre1 = disP[u - 1];
					int disp_pre2 = disp.ptr<short>(v - 1)[u];
					clock_t start1 = clock();
					for (int i = 0; i < n; i++)
					{
						int d0 = topDisp.ptr<float>(v, u, i)[0];
						float c0 = topDisp.ptr<float>(v, u, i)[1];
						for (int j = i + 1; j < n; j++)
						{
							int d1 = topDisp.ptr<float>(v, u, j)[0];
							float c1 = topDisp.ptr<float>(v, u, j)[1];
							if (abs(d0 - d1) < disp_DifThres)
							{
								dispCost_container.insert(pair<float, int>(c0, d0));
								dispCost_container.insert(pair<float, int>(c1, d1));
								step2++;
							}
						}
					}
					clock_t end1 = clock();
					time_sum += (end1 - start1);
					if (dispCost_container.empty())
					{
						clock_t start3 = clock();
						int difMostSmall1 = numeric_limits<int>::max();
						int disp1 = -1;
						int difMostSmall2 = numeric_limits<int>::max();
						int disp2 = -1;
						for (int i = 0; i < n; ++i)
						{
							int disp__ = topDisp.ptr<float>(v, u, i)[0];
							int dif1 = abs(disp__ - disp_pre1);
							if (dif1 < difMostSmall1)
							{
								difMostSmall1 = dif1;
								disp1 = disp__;
							}
							int dif2 = abs(disp__ - disp_pre2);
							if (dif2 < difMostSmall2)
							{
								difMostSmall2 = dif2;
								disp2 = disp__;
							}
						}
						int difMostSmall = min(difMostSmall1, difMostSmall2);
						int disp = difMostSmall == difMostSmall1 ? disp1 : disp2;
						if (difMostSmall < disp_DifThres3)
							disP[u] = disp;
						else
							disP[u] = topDisp.ptr<float>(v, u, 0)[0];
						clock_t end3 = clock();
						time_sum3 += (end3 - start3);
					}
					else
					{
						mapsize += dispCost_container.size();
						clock_t start4 = clock();
						int difMostSmall = numeric_limits<int>::max();
						int disp = -1;
						bool has_find1 = false;
						bool has_find2 = false;
						for (iter_dispCost = dispCost_container.begin(); iter_dispCost != dispCost_container.end(); iter_dispCost++)
						{
							step++;
							int disp = iter_dispCost->second;
							float cost = iter_dispCost->first;
							disp__num[disp]++;
							disp__cost[disp] += cost;
							if (abs(disp - disp_pre1) < disp_DifThres2)
								has_find1 = true;
							if (abs(disp - disp_pre2) < disp_DifThres2)
								has_find2 = true;
						}
						clock_t end4 = clock();
						time_sum4 += (end4 - start4);
						int start = 2, end = 2;
						if (has_find1)
							start = 0;
						if (has_find2)
							end = 4;
						if (has_find1 || has_find2)
						{
							clock_t start2 = clock();
							for (int i = start; i < end; i++)
							{
								int v_ = v + neigh_v[i];
								int u_ = u + neigh_u[i];
								if (v_ >= 0 && v_ < h_ && u_ >= 0 && u_ < w_)
								{
									int num = topDisp.ptr<float>(v_, u_, num_candi - 1)[0];
									for (int x = 0; x < num; x++)
									{
										int disp = topDisp.ptr<float>(v_, u_, x)[0];
										float cost = topDisp.ptr<float>(v_, u_, x)[1];
										if (disp__num.count(disp) == 1)
										{
											disp__num[disp]++;
											disp__cost[disp] += cost;
										}
									}
								}
							}

							int disp_A = -1;
							int num_most = -1;
							float cost_A = numeric_limits<float>::max();
							for (iter_disp_num = disp__num.begin(); iter_disp_num != disp__num.end(); iter_disp_num++)
							{
								int num = iter_disp_num->second;
								int disp = iter_disp_num->first;
								iter_disp_cost = disp__cost.find(disp);
								if (iter_disp_cost == disp__cost.end())
								{
									cout << "map 错误" << endl;
									abort();
								}
								float cost = iter_disp_cost->second;
								if (num > num_most || num == num_most && cost < cost_A)
								{
									num_most = num;
									cost_A = cost;
									disp_A = disp;
								}
							}
							disP[u] = disp_A;
							disp__cost.clear();
							disp__num.clear();
							clock_t end2 = clock();
							time_sum2 += (end2 - start2);
						}
						else
						{
							iter_dispCost = dispCost_container.begin();
							disP[u] = iter_dispCost->second;
						}
						dispCost_container.clear();
					}
				}
			}
		}
	}
	clock_t end = clock();
	cout << "time_sum is: " << time_sum << endl;
	cout << "time_sum2 is: " << time_sum2 << endl;
	cout << "time_sum3 is: " << time_sum3 << endl;
	cout << "time_sum4 is: " << time_sum4 << endl;
	printf("step is: %I64u", step);
	printf("step2 is: %I64u", step2);
	printf("mapSize is : %I64u", mapsize);
	cout << "genDispFromTopVm2 time is: " << end - start << endl;
}


void StereoMatching::clearErrTxt()
{
	string addr = param_.savePath + "err.txt";
	try
	{
		FILE* fp;
		errno_t err;
		if ((err = fopen_s(&fp, addr.c_str(), "w")) != 0)
		{
			throw "ERROR: Couldn't generate/store output statistics!";
		}

		std::fprintf(fp, "");
		if (err == 0)
		{
			std::fclose(fp);
		}
	}
	catch (const char* err)
	{
		cerr << *err << endl;
	}
}

void StereoMatching::clearTimeTxt()
{
	string addr = param_.savePath + "time.txt";
	try
	{
		FILE* fp;
		errno_t err;
		if ((err = fopen_s(&fp, addr.c_str(), "w")) != 0)
		{
			throw "ERROR: Couldn't generate/store output statistics!";
		}

		std::fprintf(fp, "");
		if (err == 0)
		{
			std::fclose(fp);
		}
	}
	catch (const char* err)
	{
		cerr << *err << endl;
	}
}

void StereoMatching::pipeline()
{
	const auto t1 = std::chrono::system_clock::now();

	clearErrTxt();  // 清空误差txt
	clearTimeTxt();  // 清空时间txt
	
	costCalculate(); // 代价计算（含代价聚合）

	if (Do_dispOptimize)
		dispOptimize(); // 视差优化

	if (Do_refine)
		refine();// 视差细化

	const auto t2 = std::chrono::system_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "disparity computation time: " << duration << "[msec]" << std::endl;
	saveTime(duration, "ALL");
}

void StereoMatching::costScan(cv::Mat& Lr, cv::Mat& vm, int rv, int ru, bool leftFirst)
{
	CV_Assert(Lr.size == vm.size);

	int W_V = 0;
	int W_U = 0;

	if (!LRM)
	{
		W_U = param_.W_U;
		W_V = param_.W_V;
	}

	int h = vm.size[0];
	int w = vm.size[1];
	int n = param_.numDisparities;

	int v0 = 0, v1 = h, u0 = W_U, u1 = w - W_U, dv = +1, du = +1;
	if ((rv > 0) || (rv == 0 && ru > 0))
	{
		v0 = h - 1; v1 = -1; u0 = w - W_U - 1; u1 = -1; dv = -1; du = -1;
	}

	bool preIsInner = true;
	for (int v = v0; v != v1; v += dv)
	{
		for (int u = u0; u != u1; u += du)
		{
			preIsInner = true;
			if (v + rv > h - 1 || v + rv < 0 || u + ru > w - W_U - 1 || u + ru < W_U)
				preIsInner = false;
			try
			{
				switch (vm.type())
				{
				case CV_8U:  // census
					updateCost<uchar>(Lr, vm, v, u, n, rv, ru, preIsInner, leftFirst);
					break;
				case CV_16U:  // SD
					updateCost<ushort>(Lr, vm, v, u, n, rv, ru, preIsInner, leftFirst);
					break;
				case CV_32F:  // SSD, CBCA, AWS
					updateCost<float>(Lr, vm, v, u, n, rv, ru, preIsInner, leftFirst);
					break;
				default:
					throw "cost volumn's type is not reasonable";
					exit(2);
				}
			}
			catch (const char* msg)
			{
				cerr << msg << endl;
			}
		}
	}
}

void StereoMatching::gen_sgm_vm(Mat& vm, vector<cv::Mat1f>& Lr, int numOfDirec)
{
	const int h = Lr[0].size[0];
	const int w = Lr[0].size[1];
	const int n = Lr[0].size[2];

	const int Lsize = Lr.size();

	for (int v = 0; v < h; v++)
	{
		for (int u = 0; u < w; u++)
		{
			float* vmP = vm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				float sum = 0;
				for (int num = 0; num < numOfDirec; num++)
				{
					sum += Lr[num].ptr<float>(v, u)[d];
				}
				vmP[d] = sum / numOfDirec;
			}
		}
	}
}

StereoMatching::StereoMatching(cv::Mat& I1_c, cv::Mat& I2_c, cv::Mat& I1_g, cv::Mat& I2_g, cv::Mat& DT, cv::Mat& all_mask, cv::Mat& nonocc_mask, cv::Mat& disc_mask, const Parameters& param) : param_(param)
{
	this->h_ = I1_c.rows;
	this->w_ = I1_c.cols;
	this->I_c.resize(2);
	this->I_g.resize(2);
	this->I_mask.resize(3);
	this->I_c[0] = I1_c;
	cv::cvtColor(I1_c, this->Lab, COLOR_BGR2Lab);
	cv::cvtColor(I1_c, this->HSV, COLOR_BGR2HSV);
	this->I_c[1] = I2_c;
	this->I_g[0] = I1_g;
	this->I_g[1] = I2_g;
	this->DT = DT;
	this->I_mask[0] = nonocc_mask;
	this->I_mask[1] = all_mask;
	this->I_mask[2] = disc_mask;
	size_vm[0] = h_;
	size_vm[1] = w_;
	size_vm[2] = param_.numDisparities;
	vm.resize(2);
	vm[0].create(h_, w_, CV_32FC(param_.numDisparities));
	vm[1].create(h_, w_, CV_32FC(param_.numDisparities));
	//vm[0].create(3, size_vm, CV_32F);
	//vm[1].create(3, size_vm, CV_32F);
	this->img_counting = 0;

	string::size_type idx;
	idx = aggregation.find("CBCA");
	if (idx != string::npos)
	{
		HVL_num = 2; // 分别表示左图和右图
		HVL.resize(HVL_num);
		//if (param_.cbca_intersect)  // 这个后面再做处理（因为涉及到后面一些代码）
		HVL_INTERSECTION.resize(HVL_num);

		int HVL_size[] = { h_, w_, 5 };
		int HVL_IS_size[] = { h_, w_, param_.numDisparities, 5 };
		for (int i = 0; i < HVL_num; i++)
		{
			HVL[i].create(3, HVL_size, CV_16U);
			//if (param_.cbca_intersect)
			HVL_INTERSECTION[i].create(4, HVL_IS_size, CV_16U);
		}
		param_.sgm_P1 = 1.0;
		param_.sgm_P2 = 3.0;
	}

	idx = aggregation.find("AWS");
	if (idx != string::npos)
		param_.sgm_P1 = 0.5, param_.sgm_P2 = 1.0;

	if (aggregation == "guideFilter")
	{
		guideVm.resize(2);
		for (int i = 0; i < 2; i++)
		{
			guideVm[i].resize(param_.numDisparities);
			for (int d = 0; d < param_.numDisparities; d++)
				guideVm[i][d].create(h_, w_, CV_32F);
		}
	}
}

static inline int wta(const uchar* vPos, int n)
{
	int minS = std::numeric_limits<int>::max();
	int disp = 0;
	for (int d = 0; d < n; d++)
	{
		if (vPos[d] < minS)
		{
			minS = vPos[d];
			disp = d;
		}
	}
	return disp;
}

int StereoMatching::HammingDistance(uint64_t c1, uint64_t c2) { return static_cast<int>(popcnt64(c1 ^ c2)); }
static inline int HammingDistance(uint64_t c1, uint64_t c2) { return static_cast<int>(popcnt64(c1 ^ c2)); }

// 将视差图的外框区域设为不合理值
void StereoMatching::fillSurronding(cv::Mat& D1, cv::Mat& D2)
{
	int h = D1.rows;
	int w = D1.cols;

	int W_V = param_.W_V;
	int W_U = param_.W_U;
	short DISP_INV = param_.DISP_INV;

	OMP_PARALLEL_FOR
		for (int v = 0; v < W_V; v++)
		{
			for (int u = 0; u < w; u++)
			{
				D1.ptr<short>(v)[u] = DISP_INV;
				D2.ptr<short>(v)[u] = DISP_INV;
			}
		}
	OMP_PARALLEL_FOR
		for (int v = h - 1; v >= h - W_V; v--)
		{
			for (int u = 0; u < w; u++)
			{
				D1.ptr<short>(v)[u] = DISP_INV;
				D2.ptr<short>(v)[u] = DISP_INV;

			}
		}
	OMP_PARALLEL_FOR
		for (int u = 0; u < W_U; u++)
		{
			for (int v = W_V; v < h - W_V; v++)
			{
				D1.ptr<short>(v)[u] = DISP_INV;
				D2.ptr<short>(v)[u] = DISP_INV;
			}
		}
	OMP_PARALLEL_FOR
		for (int u = w - 1; u >= w - W_U; u--)
		{
			for (int v = W_V; v < h - W_V; v++)
			{
				D1.ptr<short>(v)[u] = DISP_INV;
				D2.ptr<short>(v)[u] = DISP_INV;
			}
		}
}

void StereoMatching::LRConsistencyCheck(cv::Mat& D1, cv::Mat& D2)
{
	const int n = param_.numDisparities;
	int dis_occ = 0;
	int dis_mis = 0;
	int dis_err = 0;

	//OMP_PARALLEL_FOR
		for (int v = 0; v < h_; v++)
		{
			short* _D1 = D1.ptr<short>(v);
			short* _D2 = D2.ptr<short>(v);
			float* DTP = DT.ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				const short d = _D1[u];
				if (d < 0 || u - d < 0 || abs(d - _D2[u - d]) > param_.LRmaxDiff)
				{
					if (DTP[u] != 0)
						dis_err++;
					int disp = param_.DISP_OCC;
					for (int d = 0; d < n && u - d >= 0; d++)
					{
						if (_D2[u - d] == d)
						{
							disp = param_.DISP_MIS;
							if (DTP[u] != 0)
								dis_mis++;
							break;
						}
					}
					_D1[u] = disp;
				}
			}
		}

	cout << "first num" << endl;
	cout << "dis_err: " << dis_err << endl;
	cout << "dis_occ: " << dis_err - dis_mis << endl;
	cout << "dis_mis: " << dis_mis << endl;
}

void StereoMatching::cal_ave_std_ncc(cv::Mat& I, cv::Mat& E_std)
{
	const int W_U = param_.W_U;
	const int W_V = param_.W_V;

	const int W_area = (W_U * 2 + 1) * (W_V * 2 + 1);

	// cal mean and std
	//#pragma omp parallel for
	//OMP_PARALLEL_FOR
	for (int v = W_V; v < h_ - W_V; v++)
	{
		for (int u = W_U; u < w_ - W_U; u++)
		{
			float sum = 0;
			for (int dv = -W_V; dv <= W_V; dv++)
			{
				for (int du = -W_U; du <= W_U; du++)
					sum += I.ptr<uchar>(v + dv)[u + du];
			}

			float ave = sum / W_area;
			E_std.ptr<float>(v, u)[0] = ave;

			double var = 0;
			for (int dv = -W_V; dv <= W_V; dv++)
			{
				for (int du = -W_U; du <= W_U; du++)
					var += pow((float)I.ptr<uchar>(v + dv)[u + du] - ave, 2);
			}
			E_std.ptr<float>(v, u)[1] = (float)sqrt(var);
		}
	}
}

template <int LOR>
void StereoMatching::gen_NCC_vm(cv::Mat& I0, cv::Mat& I1, cv::Mat& E_std0, cv::Mat& E_std1, cv::Mat& vm)
{
	const int n = param_.numDisparities;

	const int W_U = param_.W_U;
	const int W_V = param_.W_V;

	CV_Assert(E_std0.elemSize() == 4 && E_std1.elemSize() == 4);

	//memset(E_std1.data, 0, E.rows * dst.cols * sizeof(uint64_t));

	const float DEFAULT_MC = param_.ZNCC_DEFAULT_MC;

	// 计算左图每个点（除了W_U、W_V范围外）在所有可取到的视差上的NCC值，不能取到的视差其NCC值置为0
//#pragma omp parallel for
	int leftRatio = 0, rightRatio = 1;
	if (LOR == 1)
	{
		leftRatio = 1;
		rightRatio = 0;
	}
	int v_;
	//OMP_PARALLEL_FOR
	for (v_ = W_V; v_ < h_ - W_V; v_++)
	{
		for (int u = W_U; u != w_ - W_U; u++)
		{
			float* vmP = vm.ptr<float>(v_, u);
			for (int d = 0; d != n; d++)  // 若左图的点的指定视差超出了右图的左边界(以W_U为边界, []暂且认为在W_U、W_V边界外的未计算均值和标准差的像素点存着0默认值)，则此NCC设为0（互相关中1是代表最相似）
			{
				if (u - d * rightRatio < W_U || u + d * leftRatio >= w_ - W_U)
				{
					vmP[d] = DEFAULT_MC;
					continue;
				}
				float numer = 0;
				for (int dv = -W_V; dv != W_V; dv++)
					for (int du = -W_U; du != W_U; du++)
						numer += ((float)I0.ptr<uchar>(v_ + dv)[u + du + d * leftRatio] - E_std0.ptr<float>(v_, u + d * leftRatio)[0]) *
						((float)I1.ptr<uchar>(v_ + dv)[u - d * rightRatio + du] - E_std1.ptr<float>(v_, u - d * rightRatio)[0]);
				vmP[d] = numer / (E_std0.ptr<float>(v_, u + d * leftRatio)[1] * E_std1.ptr<float>(v_, u - d * rightRatio)[1]);
			}
		}
	}
}

// ad/ssd
// AOS：0指示AD，1指示SD
void StereoMatching::gen_ad_sd_vm(Mat& vm_asd, int LOR, int AOS)
{
	const int channels = param_.SD_AD_channel;
	Mat I0 = channels == 3 ? I_c[0] : I_g[0];
	Mat I1 = channels == 3 ? I_c[1] : I_g[1];
	const int n = param_.numDisparities;
	const float DEFAULT = AOS == 0 ? 255 : 65535;
	const int pow_index = AOS == 0 ? 1 : 2;
	int leftCoefficient = 0, rightCoefficient = -1;
	//float thre = AOS == 0 ? 20 : 400;
	if (LOR == 1)
	{
		leftCoefficient = 1;
		rightCoefficient = 0;
	}
	//OMP_PARALLEL_FOR
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vPtr = vm_asd.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				int uL = u + d * leftCoefficient;
				int uR = u + d * rightCoefficient;
				if (uL >= w_ || uR < 0)
					vPtr[d] = DEFAULT;
				else
				{
					uchar* lPtr = I0.ptr<uchar>(v, uL);
					uchar* rPtr = I1.ptr<uchar>(v, uR);
					float sum = 0;
					for (int cha = 0; cha < channels; cha++)
						sum += pow(abs((float)lPtr[cha] - (float)rPtr[cha]), pow_index);
					vPtr[d] = sum / channels;
				}
			}
		}
	}
}

void StereoMatching::gen_truncAD_vm(Mat& vm_asd, int LOR)
{
	const int channels = param_.SD_AD_channel;
	Mat I0 = channels == 3 ? I_c[0] : I_g[0];
	Mat I1 = channels == 3 ? I_c[1] : I_g[1];
	const int n = param_.numDisparities;
	const int DEFAULT = 60;
	int leftCoefficient = 0, rightCoefficient = -1;
	if (LOR == 1)
	{
		leftCoefficient = 1;
		rightCoefficient = 0;
	}
	//OMP_PARALLEL_FOR
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vPtr = vm_asd.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				int uL = u + d * leftCoefficient;
				int uR = u + d * rightCoefficient;
				if (uL >= w_ || uR < 0)
					vPtr[d] = DEFAULT;
				else
				{
					uchar* lPtr = I0.ptr<uchar>(v, uL);
					uchar* rPtr = I1.ptr<uchar>(v, uR);
					int sum = 0;
					for (int cha = 0; cha < channels; cha++)
						sum += abs((int)lPtr[cha] - (int)rPtr[cha]);
					vPtr[d] = min(sum, DEFAULT);
				}
			}
		}
	}
}

void StereoMatching::gen_ssd_vm(cv::Mat& sd_vm, cv::Mat& ssd_vm)
{
	const int h = sd_vm.size[0];
	const int w = sd_vm.size[1];
	const int n = sd_vm.size[2];

	int W_U = param_.W_U;
	int W_V = param_.W_V;
	int W_S = 1;
	if (Ssdave)
		W_S = (W_U * 2 + 1) * (W_V * 2 + 1);

	CV_Assert(ssd_vm.elemSize() == 4);
	memset(ssd_vm.data, 0, (uint64_t)h * w * n * sizeof(float));

	OMP_PARALLEL_FOR
		for (int v = W_V; v < h - W_V; v++)
		{
			for (int u = W_U; u < w - W_U; u++)
			{
				for (int d = 0; d < n; d++)
				{
					int ssd = 0;
					for (int dv = -W_V; dv <= W_V; dv++)
					{
						for (int du = -W_U; du < W_U; du++)
						{
							ssd += sd_vm.ptr<ushort>(v + dv, u + du)[d];
						}
					}
					ssd_vm.ptr<float>(v, u)[d] = (float)ssd / W_S;
				}
			}
		}
}

void StereoMatching::SSD(std::string::size_type id_ssd)
{
	const int n = param_.numDisparities;
	// 对灰度图
	vector<cv::Mat> sd_vm(2);
	sd_vm[0].create(3, size_vm, CV_16U);
	sd_vm[1].create(3, size_vm, CV_16U);
	//gen_sd_vm<0>(sd_vm[0]);  // 生成ad vm
	//gen_sd_vm<1>(sd_vm[1]);  // 生成ad vm
	gen_ssd_vm(sd_vm[0], vm[0]); // 生成ssd cost volumn
	gen_ssd_vm(sd_vm[1], vm[1]); // 生成ssd cost volumn

	cout << "SSD finished" << endl;
}

void StereoMatching::census_XOR(Mat& censusCode0, Mat& censusCode1)
{
	cout << "start censusCode cal" << endl;
	string::size_type id_census;
	id_census = costcalculation.find("symmetric-census");
	Mat I0 = param_.census_channel == 3 ? I_c[0] : I_g[1];
	Mat I1 = param_.census_channel == 3 ? I_c[1] : I_g[1];
	if (id_census != string::npos)
	{
		genSymCensus<0>(I0, censusCode0);
		genSymCensus<1>(I1, censusCode1);
	}
	else // census、mean-censue
	{
		genCensus(I0, censusCode0, param_.W_V, param_.W_U);  // 生成左右图每个像素（[W_U、u-W_U)、[W_V、v-W_V))区域内，外部区域为0值）的census编码
		genCensus(I1, censusCode0, param_.W_V, param_.W_U);
	}
	cout << "censusCode generated" << endl;

}

void StereoMatching::ZNCC(cv::Mat& I0, cv::Mat& I1, vector<cv::Mat>& vm)
{
	const int n = param_.numDisparities;

	vector<Mat> E_std;
	E_std.resize(2);
	const int size_E_std[3] = { h_, w_, 2 };

	E_std[0].create(3, size_E_std, CV_32F);  // 存储左图中每个点所在窗口的均值和标准差
	E_std[1].create(3, size_E_std, CV_32F); // 右图

	cal_ave_std_ncc(I0, E_std[0]);
	cal_ave_std_ncc(I1, E_std[1]);
	gen_NCC_vm<0>(I0, I1, E_std[0], E_std[1], vm[0]);  // 计算左右图每个点（[W_U、u-W_U)、[W_V、v-W_V))区域内）的支持框均值和方差 + 生成ncc的匹配代价卷
	gen_NCC_vm<1>(I0, I1, E_std[0], E_std[1], vm[1]);
	//param_.ChooseSmall = false;
	Mat disp(h_, w_, CV_16S);
	transform_NCCVm2(vm[0]);
	transform_NCCVm2(vm[1]);
	//param_.ChooseSmall = true;
	cout << "ZNCC finished" << endl;
#ifdef DEBUG
	saveFromVm(vm, "ZNCC");
#endif // DEBUG
}

void StereoMatching::transform_NCCVm(Mat& vm)
{
	vm = 1 - abs(vm);
}

void StereoMatching::transform_NCCVm2(Mat& vm)
{
	const int n = param_.numDisparities;
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vP = vm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				float val = vP[d];
				vP[d] = val < 0 ? 1 : 1 - val;
			}
		}
	}
}

void StereoMatching::wta_Co(cv::Mat& vm, cv::Mat& D1, cv::Mat& D2)
{
	CV_Assert(vm.depth() == CV_32F);
	//CV_Assert(D1.type() == D2.type() == CV_16S);
	CV_Assert(vm.size[0] == D1.rows);
	CV_Assert(vm.size[1] == D1.cols);

	const int h = vm.size[0];
	const int w = vm.size[1];
	const int n = param_.numDisparities;

	const int DISP_SCALE = param_.DISP_SCALE;

	for (int v = 0; v < h; v++)
	{
		short* _D2 = D2.ptr<short>(v);
		short* _D1 = D1.ptr<short>(v);
		for (int u = 0; u < w; u++)
		{


			float* vmPtr = vm.ptr<float>(v, u);
			float minC_L = std::numeric_limits<float>::max();
			float minC_R = std::numeric_limits<float>::max();
			int disp_L = 0;
			int disp_R = 0;

			// cal right disp
			for (int d = 0; d < n; d++)
			{
				if (u + d >= w)
					break;
				float cost_r = vm.ptr<float>(v, u + d)[d];
				if (cost_r < minC_R)
				{
					minC_R = cost_r;
					disp_R = d;
				}
			}
			_D2[u] = static_cast<short>(disp_R * DISP_SCALE);

			// cal left disp
			for (int d = 0; d < n; d++)
			{
				if (u - d < 0)
					break;

				if (vmPtr[d] < minC_L)
				{
					disp_L = d;
					minC_L = vmPtr[d];
				}
			}

			if (UniqCk)
			{
				int d = 0;
				for (; d < n; d++)
				{
					if (vmPtr[d] * param_.uniquenessRatio_2small < minC_L && abs(d - disp_L) > 1)
					{
						_D1[u] = param_.DISP_INV;
						break;
					}
				}
				if (d < n)
					continue;
			}

			if (SubIpl)
			{
				if (disp_L > 0 && disp_L < w - 1)
				{
					float numer = vmPtr[disp_L - 1] - vmPtr[disp_L + 1];
					float denom = vmPtr[disp_L - 1] + vmPtr[disp_L + 1] - 2 * vmPtr[disp_L];
					_D1[u] = denom != 0 ? disp_L * DISP_SCALE + (DISP_SCALE * numer + denom) / (2 * denom) : disp_L * DISP_SCALE;
					break;
				}
			}
			_D1[u] = disp_L * DISP_SCALE;
		}
	}

}

void StereoMatching::genTrueHorVerArms()
{
	const int n = param_.numDisparities;
	if (param_.cbca_armHV)
	{
		memset(HVL_INTERSECTION[0].data, 0, (uint64_t)h_ * w_ * n * 5 * sizeof(ushort));
		memset(HVL_INTERSECTION[1].data, 0, (uint64_t)h_ * w_ * n * 5 * sizeof(ushort));
	}
	cout << "start combine arms" << endl;
	int leftCoefficient = 0, rightCoefficient = -1;

	int HVL_pos = -1;

	for (int num = 0; num < HVL_num; num++)
	{
		if (num % 2 == 1)
			leftCoefficient = 1, rightCoefficient = 0;

		//OMP_PARALLEL_FOR
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				for (int d = 0; d < n; d++)
				{
					int u_l = u + d * leftCoefficient;
					int u_r = u + d * rightCoefficient;
					if (u_r < 0 || u_l >= w_)
						break;
					ushort* HL1Ptr = HVL[0].ptr<ushort>(v, u_l);
					ushort* HL2Ptr = HVL[1].ptr<ushort>(v, u_r);
					ushort* pPtr = HVL_INTERSECTION[num].ptr<ushort>(v, u, d);
					ushort L1 = HL1Ptr[0];
					ushort L2 = HL1Ptr[1];
					ushort L3 = HL1Ptr[2];
					ushort L4 = HL1Ptr[3];

					ushort R1 = HL2Ptr[0];
					ushort R2 = HL2Ptr[1];
					ushort R3 = HL2Ptr[2];
					ushort R4 = HL2Ptr[3];
					pPtr[0] = min(L1, R1);  // L1, R1类型相同，min的结果类型和两者相同
					pPtr[1] = min(L2, R2);
					pPtr[2] = min(L3, R3);
					pPtr[3] = min(L4, R4);
					pPtr[4] = pPtr[0] + pPtr[1] + pPtr[2] + pPtr[3];
				}
			}
		}
	}
	cout << "finish combine arms" << endl;
}

static bool judgeColorDif(uchar* target, uchar* refer, int thres, int channel)
{
	for (int c = 0; c < channel; c++)
	{
		if (abs(target[c] - refer[c]) > thres)
			return false;
	}
	return true;
}
//  L, L_out, C_D, C_D_out, minL
void StereoMatching::calHorVerDis(int imgNum, int channel, uchar L, uchar L_out, uchar C_D, uchar C_D_out, uchar minL)
{
	clock_t start = clock();
	cout << "start cal CrossArm" << endl;
	Mat I = channel == 3 ? I_c[imgNum].clone() : I_g[imgNum].clone();
	//medianBlur(I, I, 3);
	const int h = I.rows;
	const int w = I.cols;

	int du_[] = { -1, 0 }, dv_[] = {0, -1}, du, dv;
	for (int direc = 0; direc < 4; direc++)
	{
		switch (direc)
		{
		case 0:
			du = du_[0], dv = dv_[0];
			break;
		case 1:
			du = -du;
			dv = -dv;
			break;
		case 2:
			du = du_[1];
			dv = dv_[1];
			break;
		case 3:
			du = -du;
			dv = -dv;
			break;
		}

		Mat cross = HVL[imgNum];
		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				uchar* IPtr = I.ptr<uchar>(v, u);
				ushort* HVLPtr = cross.ptr<ushort>(v, u);
				ushort arm = 1;
				for (; arm <= L_out; arm++)
				{
					int v_arm = v + arm * dv, u_arm = u + arm * du;
					if (v_arm < 0 || v_arm >= h || u_arm < 0 || u_arm >= w)
						break;
					uchar* armPtr = I.ptr<uchar>(v_arm, u_arm);
					uchar* armPrePtr = I.ptr<uchar>(v + (arm - 1) * dv, u + (arm - 1) * du);
					bool neighborDifInThres = judgeColorDif(armPtr, armPrePtr, C_D, channel);
					bool initPresDifInThres = arm <= L ? judgeColorDif(IPtr, armPtr, C_D, channel) : judgeColorDif(IPtr, armPtr, C_D_out, channel);
					if (!neighborDifInThres || !initPresDifInThres)
						break;
				}
				if (--arm >= minL)
					HVLPtr[direc] = arm;  // l已经被减过1了
				else
				{
					for (int len = minL; len >= 0; len--)  // 动态求取边框区像素点的外部臂长，使其能取到min(minL，dis2border)
					{
						if (u + len * du >= 0 && u + len * du <= w - 1 && v + len * dv >= 0 && v + len * dv <= h - 1)
						{
							HVLPtr[direc] = len;
							break;
						}
					}
				}
			}
		}

		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				ushort* crossP = cross.ptr<ushort>(v, u);
				int armSum = 0;
				for (int num = 0; num < 4; num++)
				{
					armSum += crossP[num];
				}
				crossP[4] = (ushort)armSum;
			}
		}

	}
	clock_t end = clock();
	clock_t time = end - start;
	if (imgNum == 0)
	{
		saveTime(time, "genArmForImg_L");
	}
	cout << "finish cal CrossArm" << endl;
}

static int calColorDif(uchar* p0, uchar* p1, int channels)
{
	int maxDif = 0;
	for (int cha = 0; cha < channels; cha++)
		maxDif = max(maxDif, abs(p0[cha] - p1[cha]));
	return maxDif;
}

void StereoMatching::calTileNeigh(int imgNum, int channels, cv::Mat& tile_neigh, uchar DIFThres)
{
	Mat I = channels == 3 ? I_c[imgNum] : I_g[imgNum];
	int du[] = { -1, 1, 0, 0 }, dv[] = { 0, 0, -1, 1 };  // 左右上下的顺序
	memset(tile_neigh.data, 0, (uint64_t)h_ * w_ * tile_neigh.size[2] * sizeof(char));
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			int cdif_min = numeric_limits<int>::max();
			uchar* IP = I.ptr<uchar>(v, u);
			int dir_num = -1;
			for (int dir = 0; dir < 4; dir++)
			{
				int u_neigh = u + du[dir];
				int v_neigh = v + dv[dir];
				if (u_neigh < 0 || u_neigh >= w_ || v_neigh < 0 || v_neigh >= h_)
					continue;
				uchar* I_neighP = I.ptr<uchar>(v_neigh, u_neigh);
				int colorDif = calColorDif(IP, I_neighP, channels);
				if (cdif_min > colorDif && colorDif < DIFThres)
				{
					cdif_min = colorDif;
					dir_num = dir;
				}
			}
			if (dir_num != -1)
			{
				char* tile_neighP = tile_neigh.ptr<char>(v, u);
				tile_neighP[0] = dv[dir_num];
				tile_neighP[1] = du[dir_num];
			}
		}
	}
}

static void genADcensueHCumu(cv::Mat& census1, cv::Mat& census2, cv::Mat& I1_c, cv::Mat& I2_c, cv::Mat& HCumu)
{
	CV_Assert(census1.type() == CV_64F);
	CV_Assert(I1_c.type() == CV_8UC3);
	//CV_Assert(HCumu.type() == CV_8UC3);

	const int h = HCumu.size[0];
	const int w = HCumu.size[1];
	const int n = HCumu.size[2];
	const float DEFAULT_MC = 2;

	const int ARG_CEN = 30;
	const int ARG_AD = 10;

	float cost = 0;
	for (int v = 0; v < h; v++)
	{
		for (int u = 0; u < w; u++)
		{
			uchar* I1Ptr = I1_c.ptr<uchar>(v, u);
			float* HCPtr = HCumu.ptr<float>(v, u);
			uint64_t c1 = census1.ptr<uint64_t>(v)[u];
			uint64_t* c2Ptr = census2.ptr<uint64_t>(v, w - 1 - u);
			for (int d = 0; d < n; d++)
			{
				if (u - d < 0)
					cost = DEFAULT_MC;
				else
				{
					float ADcost = 0;
					for (int c = 0; c < 3; c++)
					{
						ADcost += abs(I1Ptr[c] - I2_c.ptr<uchar>(v, u - d)[c]);
					}
					ADcost = 1 - exp(-(ADcost / 3) / ARG_AD);

					float censusCost = static_cast<float>(HammingDistance(c1, c2Ptr[d]));
					censusCost = 1 - exp(-censusCost / ARG_CEN);
					cost = ADcost + censusCost;
				}
				HCPtr[d] = u > 0 ? cost + HCumu.ptr<float>(v, u - 1)[d] : cost;
			}
		}
	}
}

void StereoMatching::gen_vm_from2vm_exp(cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, const float ARU0, const float ARU1, int LOR)
{
	CV_Assert(vm0.depth() == CV_32F);
	CV_Assert(vm1.depth() == CV_32F);
	CV_Assert(combinedVm.depth() == CV_32F);
	Mat dispMap(h_, w_, CV_16S);

	const int n = param_.numDisparities;
	
	const float DEFAULT_MC = 2.f * 0.999;
	const int W_V = param_.W_V;
	const int W_U = param_.W_U;

	int leftCoefficient = 0, rightCoefficient = 1;
	if (LOR == 1)
	{
		leftCoefficient = 1;
		rightCoefficient = 0;
	}
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vm0Ptr = vm0.ptr<float>(v, u);
			float* vm1Ptr = vm1.ptr<float>(v, u);
			float* combinedVmPtr = combinedVm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				if (u - W_U < 0 || u + W_U >= w_ || u - d * rightCoefficient - W_U < 0 || u + d * leftCoefficient + W_U >= w_ ||
					v - W_V < 0 || v + W_V >= h_)
					combinedVmPtr[d] = DEFAULT_MC;
				else
					combinedVmPtr[d] = 2 - exp(-vm0Ptr[d] / ARU0) - exp(-vm1Ptr[d] / ARU1);
			}
		}
	}
}

void StereoMatching::gen_vm_from2vm_add(cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, const float weight0, const float weight1, int LOR)
{
	CV_Assert(vm0.depth() == CV_32F);
	CV_Assert(vm1.depth() == CV_32F);
	CV_Assert(combinedVm.depth() == CV_32F);
	Mat dispMap(h_, w_, CV_16S);

	const int n = param_.numDisparities;

	const int W_V = param_.W_V;
	const int W_U = param_.W_U;

	int leftCoefficient = 0, rightCoefficient = 1;
	if (LOR == 1)
	{
		leftCoefficient = 1;
		rightCoefficient = 0;
	}
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vm0Ptr = vm0.ptr<float>(v, u);
			float* vm1Ptr = vm1.ptr<float>(v, u);
			float* combinedVmPtr = combinedVm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				//if (u - W_U < 0 || u + W_U >= w_ || u - d * rightCoefficient - W_U < 0 || u + d * leftCoefficient + W_U >= w_ ||
				//	v - W_V < 0 || v + W_V >= h_)
				//	combinedVmPtr[d] = DEFAULT_MC;
				//else
				combinedVmPtr[d] = vm0Ptr[d] * weight0 + vm1Ptr[d] * weight1;
			}
		}
	}
}

void StereoMatching::gen_adCenZNCC_vm(cv::Mat& adVm, cv::Mat& censVm, cv::Mat& znccVm, cv::Mat& adCenZnccVm, int LOR)
{
	CV_Assert(adVm.type() == CV_32F);
	CV_Assert(censVm.type() == CV_32F);
	CV_Assert(znccVm.type() == CV_32F);
	CV_Assert(adCenZnccVm.type() == CV_32F);
	Mat dispMap(h_, w_, CV_16S);

	const int n = param_.numDisparities;
	const float ARU_AD = 10;
	const float ARU_CEN = 30;
	//const float ARU_ZNCC1 = 0.3;
	//const float ARU_ZNCC2 = 1;
	const float DEFAULT_MC = 2.f * 0.999;
	const int W_V = param_.W_V;
	const int W_U = param_.W_U;
	float ARU_ZNCC = 1;

	int leftCoefficient = 0, rightCoefficient = 1;
	if (LOR == 1)
	{
		leftCoefficient = 1;
		rightCoefficient = 0;
	}
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* adPtr = adVm.ptr<float>(v, u);
			float* cenPtr = censVm.ptr<float>(v, u);
			float* znccPtr = znccVm.ptr<float>(v, u);
			float* adCenPtr = adCenZnccVm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				if (u - W_U < 0 || u + W_U >= w_ || u - d * rightCoefficient - W_U < 0 || u + d * leftCoefficient + W_U >= w_ ||
					v - W_V < 0 || v + W_V >= h_)
					adCenPtr[d] = DEFAULT_MC;
				else
				{
					//if (znccPtr[d] < 0.5)
					//	ARU_ZNCC = ARU_ZNCC1;
					//else
					//	ARU_ZNCC = ARU_ZNCC2;
					adCenPtr[d] = 2 - exp(-adPtr[d] / ARU_AD) - exp(-cenPtr[d] / ARU_CEN) - (1 - znccPtr[d]);
					//adCenPtr[d] = 2 - exp(-cenPtr[d] / ARU_CEN) - (1 - znccPtr[d]);
					//adCenPtr[d] = 2 - exp(-adPtr[d] / ARU_AD) - exp(-cenPtr[d] / ARU_CEN);
				}
			}
		}
	}
}

void StereoMatching::gen1DCumu(cv::Mat& vm, cv::Mat& area, Mat& areaIS, int dv, int du)
{
	const int n = param_.numDisparities;
	int* areaISP = NULL;

	//OMP_PARALLEL_FOR
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			int vPre = v + dv, uPre = u + du;
			bool inner = (vPre >= 0 && vPre < h_ && uPre >= 0 && uPre < w_);
			if (!param_.cbca_intersect && inner)
				area.ptr<int>(v)[u] += area.ptr<int>(vPre)[uPre];
			float* vmPtr = vm.ptr<float>(v, u);
			if (param_.cbca_intersect)
			{
				areaISP = areaIS.ptr<int>(v, u);
			}
			for (int d = 0; d < n; d++)
			{
				if (inner)
				{
					vmPtr[d] += vm.ptr<float>(vPre, uPre)[d];
					if (param_.cbca_intersect)
						areaISP[d] += areaIS.ptr<int>(vPre, uPre)[d];
				}
			}
		}
	}
}

void StereoMatching::gen_dispFromVm(Mat& vm, Mat& dispMap)
{

	CV_Assert(vm.depth() == CV_32F);
	CV_Assert(dispMap.depth() == CV_16S);
	int maxDisp = param_.numDisparities;

	for (int v = 0; v < h_; v++)
	{
		short* dispP = dispMap.ptr<short>(v);
		for (int u = 0; u < w_; u++)
		{
			float minC = numeric_limits<float>::max();
			float maxC = numeric_limits<float>::min();
			int disp = -1;
			float* vmP = vm.ptr<float>(v, u);
			for (int d = 0; d < maxDisp; d++)
			{
				if (param_.ChooseSmall)   // 值越小越相似还是值越大越相似 true代表小，代表大（用于ZNCC）
				{
					if (minC > vmP[d])
					{
						minC = vmP[d];
						disp = d;
					}

				}
				else
				{
					if (maxC < vmP[d])
					{
						maxC = vmP[d];
						disp = d;
					}
				}
			}
			dispP[u] = disp;
		}
	}
}

void StereoMatching::genfinalVm_cbca(Mat& vm, Mat& area, Mat& areaIS, int imgNum)
{
	CV_Assert(vm.type() == CV_32F);
	const int n = param_.numDisparities;

	int s = 0;
	//OMP_PARALLEL_FOR
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vmP = vm.ptr<float>(v, u);
			int* areaISP = areaIS.ptr<int>(v, u);
			if (!param_.cbca_intersect)
				s = area.ptr<int>(v)[u];
			for (int d = 0; d < n; d++)
			{
				if (param_.cbca_intersect)
					s = areaISP[d];
				vmP[d] /= s;
			}
		}
	}
}

void StereoMatching::combine_HV_Tilt(Mat& vm_HV, Mat& vm_Tile, Mat& area_HV, Mat& area_tile, Mat& areaIS_HV, Mat& areaIS_tile, int imgNum)
{
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vmP = vm[imgNum].ptr<float>(v, u);

			int HV_len = 0;
			int tile_len = 0;
			if (param_.cbca_armCombineType == 0)  // 通过比较非交的臂长
			{
				ushort* HVP = HVL[imgNum].ptr<ushort>(v, u);
				ushort* tiltP = tileCrossL[imgNum].ptr<ushort>(v, u);
				for (int dir = 0; dir < 4; dir++)
				{
					HV_len += HVP[dir];
					tile_len += tiltP[dir];
				}
				float* vm_srcP = NULL;
				if (HV_len >= tile_len)
					vm_srcP = vm_HV.ptr<float>(v, u);
				else
					vm_srcP = vm_Tile.ptr<float>(v, u);

				for (int d = 0; d < param_.numDisparities; d++)
				{
					vmP[d] = vm_srcP[d];
				}
			}

			if (param_.cbca_armCombineType == 1)  // 通过比较交的臂长（即再每个深度下都比较）
			{
				float* vm_HVP = vm_HV.ptr<float>(v, u);
				float* vm_TileP = vm_Tile.ptr<float>(v, u);
				ushort* HV_INP = HVL_INTERSECTION[imgNum].ptr<ushort>(v, u);
				ushort* tilt_INP = tile_INTERSECTION[imgNum].ptr<ushort>(v, u);
				for (int d = 0; d < param_.numDisparities; d++)
				{
					for (int dir = 0; dir < 4; dir++)
					{
						HV_len += HV_INP[dir];
						tile_len += tilt_INP[dir];
					}
					if (HV_len > tile_len)
						vmP[d] = vm_HVP[d];
					else
						vmP[d] = vm_TileP[d];
				}
			}

			if (param_.cbca_armCombineType == 2)  // 通过比较非交的面积
			{
				int* area_HVP = area_HV.ptr<int>(v, u);
				int* area_tileP = area_tile.ptr<int>(v, u);
				float* vm_HVP = vm_HV.ptr<float>(v, u);
				float* vm_TileP = vm_Tile.ptr<float>(v, u);
				for (int d = 0; d < param_.numDisparities; d++)
				{
					//if (param_.cbca_intersect)
					//{
					//	if (area_HVP[d] >= area_tileP[d])
					//		vmP[d] = vm_HVP[d];
					//	else
					//		vmP[d] = vm_TileP[d];
					//}
					//else
						if (area_HVP[0] >= area_tileP[0])
							vmP[d] = vm_HVP[d];
						else
							vmP[d] = vm_TileP[d];
				}
			}

			if (param_.cbca_armCombineType == 3)  // 通过比较交的面积
			{
				int* area_HVP = areaIS_HV.ptr<int>(v, u);
				int* area_tileP = areaIS_tile.ptr<int>(v, u);
				float* vm_HVP = vm_HV.ptr<float>(v, u);
				float* vm_TileP = vm_Tile.ptr<float>(v, u);
				for (int d = 0; d < param_.numDisparities; d++)
				{
					if (area_HVP[d] >= area_tileP[d])
						vmP[d] = vm_HVP[d];
					else
						vmP[d] = vm_TileP[d];
				}
			}

		}
	}
}

void StereoMatching::CBCA()
{
	cbca_aggregate();
}

void StereoMatching::guideFilter()
{
	//vmTrans(vm, guideVm);  // 从3维mat值赋给2维的mat数组
	const int n = param_.numDisparities;
	vector<Mat> I(2);

	for (int i = 0; i < 2; i++)
	{
		I_c[i].convertTo(I[i], CV_32F);
		split(vm[i], guideVm[i]);
		for (int d = 0; d < n; d++)
			//guideFilterCore(guideVm[i][d], I_g[i], guideVm[i][d], 9, 0.001); // radius: 8, 4 // epsilon: 500, 0.0001
			guideVm[i][d] = guideFilterCore_matlab(I[i], guideVm[i][d], 9, 0.0001);
		merge(guideVm[i], vm[i]);
	}
	//vmTrans(guideVm, vm); // 从2维mat数组赋为3维的mat
	saveFromVm(vm, "guide");
}

void StereoMatching::guideFilterCore(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon)
{
	CV_Assert(radius >= 2 && epsilon > 0);
	CV_Assert(source.data != NULL && source.channels() == 1);
	CV_Assert(guided_image.channels() == 1);
	CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);
	//CV_Assert(source.depth() == CV_32F && guided_image.depth() == CV_32F);

	Mat guided;
	if (guided_image.data == source.data)
	{
		//make a copy
		guided_image.copyTo(guided);
	}
	else
	{
		guided = guided_image;
	}

	if (guided.depth() != CV_32F)
		guided.convertTo(guided, CV_32F);
	if (source.depth() != CV_32F)
		source.convertTo(source, CV_32F);

	//计算I*p和I*I
	Mat mat_Ip, mat_I2;
	multiply(guided, source, mat_Ip);
	multiply(guided, guided, mat_I2);

	//计算各种均值
	Mat mean_p, mean_I, mean_Ip, mean_I2;
	Size win_size(2 * radius + 1, 2 * radius + 1);
	boxFilter(source, mean_p, CV_32F, win_size);
	boxFilter(guided, mean_I, CV_32F, win_size);
	boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
	boxFilter(mat_I2, mean_I2, CV_32F, win_size);

	//计算Ip的协方差和I的方差
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	Mat var_I = mean_I2 - mean_I.mul(mean_I);
	var_I += epsilon;

	//求a和b
	Mat a, b;
	divide(cov_Ip, var_I, a);
	b = mean_p - a.mul(mean_I);

	//对包含像素i的所有a、b做平均
	Mat mean_a, mean_b;
	boxFilter(a, mean_a, CV_32F, win_size);
	boxFilter(b, mean_b, CV_32F, win_size);

	//计算输出 (depth == CV_32F)
	output = mean_a.mul(guided) + mean_b;
}

Mat StereoMatching::guideFilterCore_matlab(Mat& I, Mat p, int r, float eps)
{
	vector<Mat> I_ch;
	split(I, I_ch);

	Size kerS(2 * r + 1, 2 * r + 1);
	Mat mean_I_b, mean_I_g, mean_I_r;
	boxFilter(I_ch[0], mean_I_b, CV_32F, kerS);
	boxFilter(I_ch[1], mean_I_g, CV_32F, kerS);
	boxFilter(I_ch[2], mean_I_r, CV_32F, kerS);

	Mat mean_p;
	boxFilter(p, mean_p, CV_32F, kerS);

	Mat mean_Ip_b, mean_Ip_g, mean_Ip_r;
	boxFilter(I_ch[0].mul(p), mean_Ip_b, CV_32F, kerS);
	boxFilter(I_ch[1].mul(p), mean_Ip_g, CV_32F, kerS);
	boxFilter(I_ch[2].mul(p), mean_Ip_r, CV_32F, kerS);

	Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);
	Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
	Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);

	Mat var_I_bb, var_I_bg, var_I_br, var_I_gg, var_I_gr, var_I_rr;

	boxFilter(I_ch[0].mul(I_ch[0]), var_I_bb, CV_32F, kerS);
	boxFilter(I_ch[0].mul(I_ch[1]), var_I_bg, CV_32F, kerS);
	boxFilter(I_ch[0].mul(I_ch[2]), var_I_br, CV_32F, kerS);
	boxFilter(I_ch[1].mul(I_ch[1]), var_I_gg, CV_32F, kerS);
	boxFilter(I_ch[1].mul(I_ch[2]), var_I_gr, CV_32F, kerS);
	boxFilter(I_ch[2].mul(I_ch[2]), var_I_rr, CV_32F, kerS);
	var_I_bb -= mean_I_b.mul(mean_I_b);
	var_I_bg -= mean_I_b.mul(mean_I_g);
	var_I_br -= mean_I_b.mul(mean_I_r);
	var_I_gg -= mean_I_g.mul(mean_I_g);
	var_I_gr -= mean_I_g.mul(mean_I_r);
	var_I_rr -= mean_I_r.mul(mean_I_r);


	Mat a = Mat::zeros(h_, w_, CV_32FC3);
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			Mat sigma = (Mat_<float>(3, 3) << var_I_bb.ptr<float>(v)[u], var_I_bg.ptr<float>(v)[u], var_I_br.ptr<float>(v)[u], var_I_bg.ptr<float>(v)[u], var_I_gg.ptr<float>(v)[u],
				var_I_gr.ptr <float>(v)[u], var_I_br.ptr<float>(v)[u], var_I_gr.ptr<float>(v)[u],
				var_I_rr.ptr<float>(v)[u]);

			Mat cov_Ip = (Mat_<float>(1, 3) << cov_Ip_b.ptr<float>(v)[u], cov_Ip_g.ptr<float>(v)[u], cov_Ip_r.ptr<float>(v)[u]);

			Mat cache = cov_Ip * (sigma + eps * Mat::eye(3, 3, CV_32F)).inv();
			for (int c = 0; c < 3; c++)
				a.ptr<float>(v, u)[c] = cache.ptr<float>(0)[c];
		}
	}
	vector<Mat> a_cha;
	split(a, a_cha);
	Mat b = mean_p - a_cha[0].mul(mean_I_b) - a_cha[1].mul(mean_I_g) - a_cha[2].mul(mean_I_r);

	Mat mean_a0, mean_a1, mean_a2, mean_b;
	boxFilter(a_cha[0], mean_a0, CV_32F, kerS);
	boxFilter(a_cha[1], mean_a1, CV_32F, kerS);
	boxFilter(a_cha[2], mean_a2, CV_32F, kerS);
	boxFilter(b, mean_b, CV_32F, kerS);

	Mat q = mean_a0.mul(I_ch[0]) + mean_a1.mul(I_ch[1]) + mean_a2.mul(I_ch[2]) + b;
	return q;
}

void StereoMatching::vmTrans(vector<Mat>& vm, vector<vector<Mat>>& guideVm)
{
	CV_Assert(vm[0].dims == 3 && guideVm[0][0].dims == 2);
	int n = param_.numDisparities;

	for (int i = 0; i < 2; i++)
	{
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* vmP = vm[i].ptr<float>(v, u);
				for (int d = 0; d < n; d++)
					guideVm[i][d].ptr<float>(v)[u] = vmP[d];
			}
		}
	}

}

void StereoMatching::vmTrans(vector<vector<Mat>>& guideVm, vector<Mat>& vm)
{
	CV_Assert(vm[0].dims == 3 && guideVm[0][0].dims == 2);
	int n = param_.numDisparities;

	for (int i = 0; i < 2; i++)
	{
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* vmP = vm[i].ptr<float>(v, u);
				for (int d = 0; d < n; d++)
					vmP[d] = guideVm[i][d].ptr<float>(v)[u];
			}
		}
	}

}

void StereoMatching::adCensus(vector<Mat>& adVm, vector<Mat>& cenVm)
{
	const int n = param_.numDisparities;
	Mat dispMap(h_, w_, CV_16S);
	int imgNum = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < imgNum; i++)
		gen_vm_from2vm_exp(vm[i], adVm[i], cenVm[i], 10, 30, i);
#ifdef DEBUG
	saveFromVm(vm, "adCensus");
#endif // DEBUG
	cout << "adCensus vm generated" << endl;
}

void StereoMatching::adCensusZncc(vector<Mat>& adVm, vector<Mat>& cenVm, vector<Mat>& znccVm)
{
	const int n = param_.numDisparities;
	Mat dispMap(h_, w_, CV_16S);
	int imgNum = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < imgNum; i++)
		gen_adCenZNCC_vm(adVm[i], cenVm[i], znccVm[i], vm[i], i);
#ifdef DEBUG
	saveFromVm(vm, "adCensusZncc");
#endif // DEBUG
	cout << "adCensus vm generated" << endl;
}

void StereoMatching::genCensus(cv::Mat& img, cv::Mat& censusCode, int R_V, int R_U)
{
	Mat imgB;
	copyMakeBorder(img, imgB, R_V, R_V, R_U, R_U, BORDER_REFLECT_101);
	CV_Assert(censusCode.elemSize() == 8);

	const int W_area = (R_V * 2 + 1) * (R_U * 2 + 1);

	float mean = 0;
	for (int v = 0; v < h_; v++)
	{
		uint64_t* censusPtr = censusCode.ptr<uint64_t>(v);
		uchar* imgP = imgB.ptr<uchar>(v + R_V);
		for (int u = 0; u < w_; u++)
		{
			uchar imgV = imgP[u + R_U];
			if (costcalculation == "mean-census")
			{
				int sum = 0;
				for (int dv = -R_V; dv <= R_V; dv++)
				{
					for (int du = -R_U; du <= R_U; du++)
					{
						sum += img.ptr<uchar>(v + R_V + dv)[u + R_U + du];
					}
				}
				mean = sum / W_area;
				imgV = mean;
			}

			uint64_t c = 0;
			for (int dv = -R_V; dv <= R_V; dv++)
			{
				for (int du = -R_U; du <= R_U; du++)
				{
					c <<= 1;
					c += imgV < imgB.ptr<uchar>(v + R_V + dv)[u + R_U + du] ? 1 : 0;
				}
			}
			censusPtr[u] = c;
		}
	}
}

// 计算左右图的每个点的左、右、上、下四条臂的长度（最短可以是0）**需要找一个参数来控制臂的最短长度
void StereoMatching::calArms()
{
	int channels = cbca_genArm_isColor ? 3 : 1;
	const uchar C_D = param_.Cross_C;  // 20
	const uchar C_D_out = 6;
	const uchar minL = param_.cbca_minArmL;  // 1

	for (int num = 0; num < HVL_num; num++)
	{
		cout << "start cal horVerArm for img " << num << endl;
		const uchar L = param_.cbca_crossL[0];  // 17
		const uchar L_out = param_.cbca_crossL_out[0];  // 34  
		calHorVerDis(num, channels, L, L_out, C_D, C_D_out, minL);

		// 生成左右图每个点在每个视差下的水平、竖直轴（通过交运算），存成一个h*w*n*4的mat
		if (param_.cbca_intersect)
			genTrueHorVerArms();

		cout << "finish cal horVerArm for img " << num << endl;
	}

	cout << "HVL generated" << endl;
}

void StereoMatching::cbca_aggregate()
{
	cout << "\n" << endl;
	cout << "start CBCA aggregation" << endl;
	clock_t start = clock();

	// 计算臂长
	calArms();

	int imgNum = Do_LRConsis ? 2 : 1;

	Mat dispMap(h_, w_, CV_16S);
	Mat dispMap2(h_, w_, CV_16S);
	Mat result(h_, w_, CV_16S);

	int du[2] = { -1, 0 }, dv[2] = { 0, -1 };

	Mat disp_res(h_, w_, CV_16S);
	gen_dispFromVm(vm[0], disp_res);
	Mat disp_lastIte_save = disp_res.clone();
	Mat cross, crossIntersec, area, areaIS, vm_copy;
	for (int LOR = 0; LOR < imgNum; LOR++)
	{
		for (int agItNum = 0; agItNum < param_.cbca_iterationNum; agItNum++)
		{
			cout << "start img " << LOR << " crossIteration " << agItNum << endl;

			if (param_.cbca_intersect)
				areaIS = Mat(3, size_vm, CV_32S, Scalar::all(1));
			else
				area = Mat::ones(h_, w_, CV_32S);
			clock_t start_inner = clock();
			if (agItNum % 2 == 0)
			{
				gen1DCumu(vm[LOR], area, areaIS, dv[0], du[0]);
				cal1DCost(vm[LOR], HVL[LOR], area, areaIS, HVL_INTERSECTION[LOR], dv[0], du[0], 0);
				gen1DCumu(vm[LOR], area, areaIS, dv[1], du[1]);
				cal1DCost(vm[LOR], HVL[LOR], area, areaIS, HVL_INTERSECTION[LOR], dv[1], du[1], 1);
			}
			else
			{
				gen1DCumu(vm[LOR], area, areaIS, dv[1], du[1]);
				cal1DCost(vm[LOR], HVL[LOR], area, areaIS, HVL_INTERSECTION[LOR], dv[1], du[1], 1);
				gen1DCumu(vm[LOR], area, areaIS, dv[0], du[0]);
				cal1DCost(vm[LOR], HVL[LOR], area, areaIS, HVL_INTERSECTION[LOR], dv[0], du[0], 0);
			}
			genfinalVm_cbca(vm[LOR], area, areaIS, LOR);
			if (LOR == 0 && agItNum == 0)
			{
				gen_dispFromVm(vm[LOR], dispMap);
				disp_lastIte_save = dispMap.clone();
			}

			clock_t end_inner = clock();
			clock_t time_inner = end_inner - start_inner;
			cout << "time: " << time_inner << endl;
			if (LOR == 0)
			{
				saveTime(time_inner, "cbciInner0");
			}
			cout << "img " << LOR << " aggregation " << agItNum << " finished " << endl;
#ifdef DEBUG
			if (LOR == 0)
			{
				gen_dispFromVm(vm[LOR], dispMap);
				if (!param_.Do_vmTop)
					saveFromDisp<short>(dispMap, "cbca" + to_string(agItNum));
				else
				{
					int sizeVmTop[] = { h_, w_, param_.vmTop_Num + 1, 2 };
					Mat topDisp(4, sizeVmTop, CV_32F);
					vm_copy = vm[0].clone();
					selectTopCostFromVolumn(vm_copy, topDisp, param_.vmTop_thres, param_.vmTop_Num);
					//signCorrectFromTopVm("correctFromTopVmCBCA" + to_string(agItNum) + ".png", topDisp, DT);
					genExcelFromTopDisp(topDisp, DT);
					//genDispFromTopCostVm(topDisp, dispMap2);
					genDispFromTopCostVm2(topDisp, dispMap2);
					signDispChange_for2Disp(dispMap, dispMap2, DT, I_mask[0], result);
					saveDispMap<short>(result, "candidate_Change" + to_string(agItNum));
					saveFromDisp<short>(dispMap2, "cbcatopVm" + to_string(agItNum));
				}
			}
			cout << "cbca aggregation iteration:" + to_string(agItNum) + " finished" << endl;
#endif // DEBUG
		}
	}
	clock_t end = clock();
	clock_t time = end - start;
	cout << time << endl;
	saveTime(time, "CBCA aggre");
	cout << "finish CBCA aggregation" << endl;
}

void StereoMatching::AWS()
{
	const auto t1 = std::chrono::system_clock::now();
	std::cout << "AWS start" << endl;
	// 计算内部区域代价
	const int W_U_AWS = 17, W_V_AWS = 17;
	const int n = param_.numDisparities;
	const int W_S = (W_U_AWS * 2 + 1) * (W_V_AWS * 2 + 1);
	const int HI_L = W_U_AWS;  // 左右边缘往外拓展的长度
	int size_wt[] = { h_, w_, W_S };
	Mat wt[2], vmTem[2], Lab[2], I_IpolBorder[2], vm_IB[2];
	int size_vmIB[] = { h_ + W_V_AWS * 2, w_ + HI_L * 2, param_.numDisparities };
	int loopNum = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < loopNum; i++)
	{
		vmTem[i].create(3, size_vm, CV_32F);
		initializeMat<float>(vmTem[i], W_S * 3);
		vm_IB[i].create(3, size_vmIB, CV_32F);
	}
	for (int i = 0; i < 2; i++)
	{
		wt[i].create(3, size_wt, CV_32F);
		//copyMakeBorder(I_c[i], I_IpolBorder[i], W_V_AWS, W_V_AWS, 0, 0, BORDER_REFLECT_101);
		copyMakeBorder(I_c[i], I_IpolBorder[i], W_V_AWS, W_V_AWS, HI_L, HI_L, BORDER_REFLECT_101);
		cv::cvtColor(I_IpolBorder[i], Lab[i], COLOR_BGR2Lab);
	}

	genWeight_AWS(0, h_, W_U_AWS - HI_L, w_ - (W_U_AWS - HI_L), W_V_AWS, W_U_AWS, W_V_AWS, HI_L, wt, Lab);
	genTadVm<0>(I_IpolBorder, vm_IB[0]);
	if (Do_LRConsis)
		genTadVm<1>(I_IpolBorder, vm_IB[1]);
	calvm_AWS<0>(0, h_, W_U_AWS - HI_L, w_ - (W_U_AWS - HI_L), n, W_V_AWS, W_U_AWS, W_V_AWS, HI_L, vmTem[0], wt, vm_IB[0]);
	if (Do_LRConsis)
		calvm_AWS<1>(0, h_, W_U_AWS - HI_L, w_ - (W_U_AWS - HI_L), n, W_V_AWS, W_U_AWS, W_V_AWS, HI_L, vmTem[1], wt, vm_IB[1]);

	//// 计算左右边界代价
	//const int W_V_BORDER = 3, W_U_BORDER = 3;
	//const int W_S_Border = (W_U_BORDER * 2 + 1) * (W_V_BORDER * 2 + 1);
	//const int size_wt_border[] = { h_, w_, W_S_Border };
	//int borderL = param_.numDisparities + W_U_AWS;
	//Mat vmTem_Border[2];
	//Mat wt_border[2];

	//for (int i = 0; i < loopNum; i++)
	//{
	//	vmTem_Border[i].create(3, size_vm, CV_32F);
	//	initializeMat<float>(vmTem_Border[i], W_S_Border * 3);
	//}
	//wt_border[0].create(3, size_wt_border, CV_32F);
	//wt_border[1].create(3, size_wt_border, CV_32F);

	//genWeight_AWS(0, h_, W_U_BORDER, borderL - 1, W_V_BORDER, W_U_BORDER, W_V_AWS, wt_border, Lab);
	//genWeight_AWS(0, h_, w_ - borderL, w_ - W_U_BORDER, W_V_BORDER, W_U_BORDER, W_V_AWS, wt_border, Lab);
	//calvm_AWS<0>(0, h_, W_U_BORDER, borderL, n, W_V_BORDER, W_U_BORDER, W_V_AWS, vmTem_Border[0], wt_border, vm_IB);
	//calvm_AWS<0>(0, h_, w_ - borderL, w_ - W_U_BORDER, n, W_V_BORDER, W_U_BORDER, W_V_AWS, vmTem_Border[0], wt_border, vm_IB);
	//if (Do_LRConsis)
	//{
	//	calvm_AWS<1>(0, h_, W_U_BORDER, borderL, n, W_V_BORDER, W_U_BORDER, W_V_AWS, vmTem_Border[1], wt_border, vm_IB);
	//	calvm_AWS<1>(0, h_, w_ - borderL, w_ - W_U_BORDER, n, W_V_BORDER, W_U_BORDER, W_V_AWS, vmTem_Border[1], wt_border, vm_IB);
	//}
	//updateBorder_vm<0>(vmTem[0], vmTem_Border[0], borderL, W_U_AWS, W_U_BORDER, W_U_AWS);
	//if (Do_LRConsis)
	//	updateBorder_vm<1>(vmTem[1], vmTem_Border[1], borderL, W_U_AWS, W_U_BORDER, W_U_AWS);

	// 以下本来想通过census补充左右边界，但效果不好
	//const int W_V_cen = 3;
	//const int W_U_cen = 4;
	//Mat census_AWS_BORDER[2];
	//int varN = ceil((W_V_cen * 2 + 1) * (W_U_cen * 2 + 1) * 3 / 64.);
	//int size_AWS_BORDER[] = { h_, w_ , varN };
	//census_AWS_BORDER[0].create(3, size_AWS_BORDER, CV_64F);
	//census_AWS_BORDER[1].create(3, size_AWS_BORDER, CV_64F);

	//genCensus_AWS_BORDER(I_c, census_AWS_BORDER, borderL, W_V_cen, W_U_cen);
	//update_AWS_vm(census_AWS_BORDER, borderL, W_V_cen, W_U_cen, W_U_AWS, vmTem);

	const auto t2 = std::chrono::system_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	cout << "AWS time: " << duration << endl;

	vmTem[0].copyTo(vm[0]);
	if (Do_LRConsis)
		vmTem[1].copyTo(vm[1]);
#ifdef DEBUG
	saveFromVm(vm, "asw");
#endif // DEBUG
}

void StereoMatching::combine_Cross_FW(Mat& vm_dst, Mat& vm_BF, Mat& area, Mat& areaIS, int imgNum)
{
	const int n = param_.numDisparities;
	const int armS = param_.cbca_armSLimit;
	ushort armLenLimit = param_.cbca_armSmallLimit;
	ushort armP = 0;
	ushort armSingleMostLen = 0;

	Mat dispMat(h_, w_, CV_16S);
	gen_dispFromVm(vm_dst, dispMat);
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vm_dstP = vm_dst.ptr<float>(v, u);
			float* vm_BFP = vm_BF.ptr<float>(v, u);
			armP = HVL[imgNum].ptr<ushort>(v, u)[4];
			int* areaISP = areaIS.ptr<int>(v, u);
			for (int d = 0; d < n; d++)
			{
				if (param_.cobineCrossFWType == 1)
				{
					if (param_.cbca_intersect)
						armP = HVL_INTERSECTION[imgNum].ptr<ushort>(v, u, d)[4]; // ***
					if (armP < armLenLimit)
						vm_dstP[d] = vm_BFP[d];
				}
				else if (param_.cobineCrossFWType == 3)
				{
					int S = areaISP[d];
					if (S < 5)
						vm_dstP[d] = vm_BFP[d];
				}
				else if (param_.cobineCrossFWType == 4)
				{
					ushort* HVL_INTP = HVL_INTERSECTION[imgNum].ptr<ushort>(v, u, d);
					armP = HVL_INTP[4];
					int arm_HL = HVL_INTP[0] + HVL_INTP[1];
					int arm_VL = HVL_INTP[2] + HVL_INTP[3];
					if (armP < param_.armLSum && arm_HL < param_.armLSingle && arm_VL < param_.armLSingle)
					{
						if (param_.cbca_do_costCompare)
						{
							if (vm_dstP[d] > vm_BFP[d])
								vm_dstP[d] = vm_BFP[d];
						}
						else
							vm_dstP[d] = vm_BFP[d];
					}
				}
			}
		}
	}
	Mat result(h_, w_, CV_16S);
	Mat dispMat2(h_, w_, CV_16S);
	gen_dispFromVm(vm_dst, dispMat2);
	//signDispChange_forRV(dispMat, dispMat2, );
	signDispChange_for2Disp(dispMat, dispMat2, DT, I_mask[0], result);
	saveDispMap<short>(result, "BF-CBBI_Change" + to_string(imgNum));
}

void StereoMatching::iterpolateBackground(cv::Mat& ipol)
{
	const int height_ = ipol.rows;
	const int width_ = ipol.cols;

	const int ipolThres = param_.ipolThrehold * param_.DISP_SCALE;

	// for each row do
	for (int32_t v = 0; v < height_; v++) {

		// init counter 
		int32_t count = 0; //  用于统计不合理的像素的数量

		// for each pixel do
		for (int32_t u = 0; u < width_; u++) {

			// if disparity valid
			if (ipol.ptr<short>(v)[u] > ipolThres) {

				// at least one pixel requires interpolation
				if (count >= 1) {

					// first and last value for interpolation
					int32_t u1 = u - count;
					int32_t u2 = u - 1;

					// set pixel to min disparity
					if (u1 > 0 && u2 < width_ - 1) {  // 0值区域两侧必须含非0区域才插值（选两侧较小的值来填充）
						float d_ipol = std::min(ipol.ptr<short>(v)[u1 - 1], ipol.ptr<short>(v)[u2 + 1]);
						for (int32_t u_curr = u1; u_curr <= u2; u_curr++)
							ipol.ptr<short>(v)[u_curr] = d_ipol;
					}
				}

				// reset counter
				count = 0;

				// otherwise increment counter
			}
			else {
				count++;
			}
		}

		// extrapolate to the left
		for (int32_t u = 0; u < width_; u++) {
			if (ipol.ptr<short>(v)[u] > ipolThres) {
				for (int32_t u2 = 0; u2 < u; u2++)
					ipol.ptr<short>(v)[u2] = ipol.ptr<short>(v)[u];
				break;
			}
		}

		// extrapolate to the right
		for (int32_t u = width_ - 1; u >= 0; u--) {
			if (ipol.ptr<short>(v)[u] > ipolThres) {
				for (int32_t u2 = u + 1; u2 <= width_ - 1; u2++)
					ipol.ptr<short>(v)[u2] = ipol.ptr<short>(v)[u];
				break;
			}
		}
	}

	// for each column do
	for (int32_t u = 0; u < width_; u++) {

		// extrapolate to the top
		for (int32_t v = 0; v < height_; v++) {
			if (ipol.ptr<short>(v)[u] > ipolThres) {
				for (int32_t v2 = 0; v2 < v; v2++)
					ipol.ptr<short>(v2)[u] = ipol.ptr<short>(v)[u];
				break;
			}
		}

		// extrapolate to the bottom
		for (int32_t v = height_ - 1; v >= 0; v--) {
			if (ipol.ptr<short>(v)[u] > ipolThres) {
				for (int32_t v2 = v + 1; v2 <= height_ - 1; v2++)
					ipol.ptr<short>(v2)[u] = ipol.ptr<short>(v)[u];
				break;
			}
		}
	}
}

void StereoMatching::biaryImg(cv::Mat& DP, cv::Mat& DT, cv::Mat& biaryImg)  // 限制DP和DT是float类型的, biarImg为uchar类型
{
	CV_Assert(DP.type() == CV_32F && DT.type() == CV_32F && biaryImg.type() == CV_8UC3);
	CV_Assert(DP.size() == DT.size());
	const int h = DP.rows;
	const int w = DP.cols;

	for (int v = 0; v < h; v++)
	{
		float* dpPtr = DP.ptr<float>(v);
		float* dtPtr = DT.ptr<float>(v);
		for (int u = 0; u < w; u++)
		{
			uchar* biPtr = biaryImg.ptr<uchar>(v, u);
			if (dtPtr[u] == 0)
			{
				biPtr[0] = 0;
				biPtr[1] = 0;
				biPtr[2] = 125;

			}
			else
			{
				if (abs(dpPtr[u] - dtPtr[u]) > 1)
				{
					biPtr[0] = 255;
					biPtr[1] = 255;
					biPtr[2] = 255;
				}
				else
				{
					biPtr[0] = 0;
					biPtr[1] = 0;
					biPtr[2] = 0;
				}
			}

		}
	}
}

void StereoMatching::errorMap(cv::Mat& DP, cv::Mat& DT, cv::Mat& errorMap)
{
	CV_Assert(DP.size() == DT.size());
	int h = DP.rows;
	int w = DP.cols;

	for (int v = 0; v < h; v++)
	{
		float* _DP = DP.ptr<float>(v);
		float* _DT = DT.ptr<float>(v);

		for (int u = 0; u < w; u++)
		{
			float dt = _DT[u];
			uint8_t* errMPos = errorMap.ptr<uchar>(v, u);
			if (dt > 0)
			{
				float dif = std::min(fabs(dt - _DP[u]), 1.0f) / 1.0f;
				errMPos[0] = (uint8_t)(dif * 255.0);
				errMPos[1] = (uint8_t)(dif * 255.0);
				errMPos[2] = (uint8_t)(dif * 255.0);
			}
			else
			{
				errMPos[0] = 0;
				errMPos[1] = 0;
				errMPos[2] = 125;
			}
		}
	}
}

void StereoMatching::cal_err(cv::Mat& DT, cv::Mat& DP, FILE* save_txt)
{
	CV_Assert(DT.type() == CV_32F && DP.type() == CV_32F);
	const int h = DP.rows;
	const int w = DP.cols;

	int errorNumer = 0, validNum = 0;
	float errorValueSum = 0;
	const int THRES = param_.errorThreshold;
	for (int v = 0; v < h; v++)
	{
		float* ptrDP = DP.ptr<float>(v);
		float* ptrDT = DT.ptr<float>(v);
		for (int u = 0; u < w; u++)
		{
			if (ptrDT[u] > 0)
			{
				validNum++;
				float dif = abs(ptrDT[u] - ptrDP[u]);
				errorValueSum += dif;
				if (dif > THRES)
					errorNumer++;
			}
		}
	}
	float errorRatio = (float)errorNumer / validNum;
	float epe = errorValueSum / validNum;

	std::cout << "errorRatio: " << errorRatio << " epe: " << epe << endl;
	std::fprintf(save_txt, "%f ", errorRatio);
	std::fprintf(save_txt, "%f ", epe);
}

void StereoMatching::discontinuityAdjust(cv::Mat& disp)
{
	cv::Mat disp_E;
	disp.convertTo(disp_E, CV_8U);
	equalizeHist(disp_E, disp_E);  // 直方图均衡化用于增大图像的对比度
	//blur(disp_E, disp_E, cv::Size(3, 3));  //均值滤波
	GaussianBlur(disp_E, disp_E, cv::Size(3, 3), 4, 4);
	Canny(disp_E, disp_E, 20, 60, 3);  // 执行Canny边缘检测，生成黑白图（其中白色表示边缘，黑色非边缘）

	string add = "D:\\workspace\\visualStudio\\cost_calculating\\e.png";
	cv::imwrite(add, disp_E);

	int directionsH[] = { -1, 1, -1, 1, -1, 1, 0, 0 };
	int directionsW[] = { -1, 1, 0, 0, 1, -1, -1, 1 };

	for (int h = 1; h < disp.rows - 1; h++)
	{
		for (int w = 1; w < disp.cols - 1; w++)
		{
			// if pixel is on an edge
			if (disp_E.at<uchar>(h, w) != 0)
			{
				int direction = -1;
				if (disp_E.at<uchar>(h - 1, w - 1) != 0 && disp_E.at<uchar>(h + 1, w + 1) != 0)
				{
					direction = 4;
				}
				else if (disp_E.at<uchar>(h - 1, w + 1) != 0 && disp_E.at<uchar>(h + 1, w - 1) != 0)
				{
					direction = 0;
				}
				else if (disp_E.at<uchar>(h - 1, w) != 0 || disp_E.at<uchar>(h - 1, w - 1) != 0 || disp_E.at<uchar>(h - 1, w + 1) != 0)
				{
						if (disp_E.at<uchar>(h + 1, w) != 0 || disp_E.at<uchar>(h + 1, w - 1) != 0 || disp_E.at<uchar>(h + 1, w + 1) != 0)
							direction = 6;
				}
				else
				{
					if (disp_E.at<uchar>(h - 1, w - 1) != 0 || disp_E.at<uchar>(h, w - 1) != 0 || disp_E.at<uchar>(h + 1, w - 1) != 0)
						if (disp_E.at<uchar>(h - 1, w + 1) != 0 || disp_E.at<uchar>(h, w + 1) != 0 || disp_E.at<uchar>(h + 1, w + 1) != 0)
							direction = 2;
				}

				if (direction != -1)
				{
					short dp = disp.at<short>(h, w);

					// select pixels from both sides of the edge
					if (dp >= 0)
					{
						float cost = vm[0].ptr<float>(h, w)[dp];
						//float cost = costs[0][disp - dMin].at<costType>(h, w);
						short d1 = disp.at<short>(h + directionsH[direction], w + directionsW[direction]);
						short d2 = disp.at<short>(h + directionsH[direction + 1], w + directionsW[direction + 1]);

						float cost1 = (d1 >= 0)
							? vm[0].at<float>(h + directionsH[direction], w + directionsW[direction], d1)
							: -1;

						float cost2 = (d2 >= 0)
							? vm[0].at<float>(h + directionsH[direction + 1], w + directionsW[direction + 1], d2)
							: -1;

						if (cost1 >= 0 && cost1 < cost)
						{
							dp = d1;
							cost = cost1;
						}

						if (cost2 != -1 && cost2 < cost)
						{
							dp = d2;
						}
					}
					disp.at<short>(h, w) = dp;
				}
			}
		}
	}
}

void StereoMatching::subpixelEnhancement(cv::Mat& disparity, Mat& floatDisp)
{
	CV_Assert(disparity.type() == CV_16S);
	CV_Assert(floatDisp.type() == CV_32F);

	//OMP_PARALLEL_FOR
	for (int h = 0; h < disparity.rows; h++)
	{
		for (int w = 0; w < disparity.cols; w++)
		{
			short disp = disparity.at<short>(h, w);

			if (disp > 0 && disp < param_.numDisparities - 1)
			{
				float cost = vm[0].at<float>(h, w, disp);
				float costPlus = vm[0].at<float>(h, w, disp + 1);
				float costMinus = vm[0].at<float>(h, w, disp - 1);

				float denom = 2 * (costPlus + costMinus - 2 * cost);
				if (denom != 0)
				{
					float diff = (costPlus - costMinus) / denom;
					if (diff > -1 && diff < 1)
						disp -= diff;
				}
			}
			floatDisp.at<float>(h, w) = (float)disp;
		}
	}
}

void StereoMatching::saveErrorMap(string addr, Mat& DP, Mat& DT)
{
	cv::Mat errMap(h_, w_, CV_8UC3);

	CV_Assert(DP.size() == DT.size());

	OMP_PARALLEL_FOR
		for (int v = 0; v < h_; v++)
		{
			float* _DP = DP.ptr<float>(v);
			float* _DT = DT.ptr<float>(v);

			for (int u = 0; u < w_; u++)
			{
				float dt = _DT[u];
				uint8_t* errMPos = errMap.ptr<uchar>(v, u);
				if (dt > 0)
				{
					float dif = std::min(fabs(dt - _DP[u]), 1.0f) / 1.0f;
					errMPos[0] = (uint8_t)(dif * 255.0);
					errMPos[1] = (uint8_t)(dif * 255.0);
					errMPos[2] = (uint8_t)(dif * 255.0);
				}
				else
				{
					errMPos[0] = 0;
					errMPos[1] = 0;
					errMPos[2] = 125;
				}
			}
		}
	imwrite(addr, errMap);

}

void StereoMatching::sgm(cv::Mat& vm, bool leftFirst)  // vm的数据类型由代价计算方法而定
{
	const int MAX_DIRECTIONS = 8;
	const int rv[MAX_DIRECTIONS] = { +1, -1, 0, 0, +1, +1, -1, -1 };
	const int ru[MAX_DIRECTIONS] = { 0, 0, +1, -1, -1, +1, +1, -1 };

	int numOfDirec = param_.sgm_scanNum;
	L.resize(numOfDirec);

	for (int i = 0; i < numOfDirec; i++)
	{
		L[i].create(3, size_vm);
		costScan(L[i], vm, rv[i], ru[i], leftFirst);
		cout << "scanLline " << i << "finished" << endl;
	}
	gen_sgm_vm(vm, L, numOfDirec);
}

int StereoMatching::cal_histogram_for_HV(Mat& dispImg, int v_ancher, int u_ancher, int numThres, float ratioThre)
{
	int n = param_.numDisparities + 1;
	vector<int> hist(n, 0);
	int validNum = 0;
	int v_begin = v_ancher - HVL[0].ptr<ushort>(v_ancher, u_ancher)[2];
	int v_end = v_ancher + HVL[0].ptr<ushort>(v_ancher, u_ancher)[3];
	for (int v = v_begin; v <= v_end; v++)
	{
		int u_begin = u_ancher - HVL[0].ptr<ushort>(v, u_ancher)[0];
		int u_end = u_ancher + HVL[0].ptr<ushort>(v, u_ancher)[1];
		short* dispP = dispImg.ptr<short>(v);
		for (int u = u_begin; u <= u_end; u++)
		{
			if (dispP[u] >= 0)
			{
				validNum++;
				hist[dispP[u]]++;
			}
		}
	}
	if (validNum <= numThres)
		return -1;

	int dispMost = 0;
	for (int d = 1; d < n; d++)
	{
		if (hist[d] > hist[dispMost])
			dispMost = d;
	}
	float ratioMost = (float)hist[dispMost] / validNum;
	return ratioMost > ratioThre ? dispMost : -1;
}

int StereoMatching::cal_histogram_for_Tile(Mat& dispImg, int v_ancher, int u_ancher, int numThres, float ratioThre)
{
	int n = param_.numDisparities;
	int num_valid = 0;
	vector<int> hist(n, 0);
	int du[] = { -1, 1 }, dv[] = { 1, 1 };
	int v_neigh = v_ancher + tile_neighbor[0].ptr<char>(v_ancher, u_ancher)[0];
	int u_neigh = u_ancher + tile_neighbor[0].ptr<char>(v_ancher, u_ancher)[1];
	int pointNum = (v_neigh != v_ancher || u_neigh != u_ancher) ? 2 : 1;
	int v_aim = v_ancher, u_aim = u_ancher;
	for (int point = 0; point < 1; point++)
	{
		if (point == 1)
			v_aim = v_neigh, u_aim = u_neigh;

		int armRT = tileCrossL[0].ptr<ushort>(v_aim, u_aim)[2], armLD = tileCrossL[0].ptr<ushort>(v_aim, u_aim)[3];
		for (int mainArm = -armRT; mainArm <= armLD; mainArm++)
		{
			int v_main = v_aim + mainArm * dv[0];
			int u_main = u_aim + mainArm * du[0];
			int armLT = tileCrossL[0].ptr<ushort>(v_main, u_main)[0];
			int armRD = tileCrossL[0].ptr<ushort>(v_main, u_main)[1];
			for (int branchArm = -armLT; branchArm <= armRD; branchArm++)
			{
				int v = v_main + branchArm * dv[1];
				int u = u_main + branchArm * du[1];
				short disp = dispImg.ptr<short>(v)[u];
				if (disp >= 0)
				{
					hist[disp]++;
					num_valid++;
				}

			}
		}
	}
	if (num_valid <= numThres)
		return -1;

	short dispMost = 0;
	for (int d = 1; d < n; d++)
	{
		if (hist[dispMost] < hist[d])
			dispMost = d;	
	}
	return (float)hist[dispMost] / num_valid > ratioThre ? dispMost : -1;
}

int StereoMatching::compareArmL(int v, int u)
{
	ushort* HVLP = HVL[0].ptr<ushort>(v, u);
	ushort* tileLP = tileCrossL[0].ptr<ushort>(v, u);
	int len_HV = 0, len_tile = 0;
	for (int dir = 0; dir < 4; dir++)
	{
		len_HV += HVLP[dir];
		len_tile += tileLP[dir];
	}
	return len_HV > len_tile ? 0 : 1;
}

int StereoMatching::regionVoteCore(Mat& Dp, int v, int u, int SThres, float hratioThres)
{
	int dp_;
	if (param_.cbca_armHV && param_.cbca_armTile)
	{
		if (param_.regVote_type == 2)
		{
			int result = compareArmL(v, u);
			dp_ = result == 0 ? cal_histogram_for_HV(Dp, v, u, SThres, hratioThres) : cal_histogram_for_Tile(Dp, v, u, SThres, hratioThres);
		}
		else if (param_.regVote_type == 1)
			dp_ = cal_histogram_for_Tile(Dp, v, u, SThres, hratioThres);
		else
			dp_ = cal_histogram_for_HV(Dp, v, u, SThres, hratioThres);

	}
	else if (param_.cbca_armHV)
		dp_ = cal_histogram_for_HV(Dp, v, u, SThres, hratioThres);
	else
		dp_ = cal_histogram_for_Tile(Dp, v, u, SThres, hratioThres);
	return dp_;
}

/*
 *@searchDepth 表示往左搜寻的深度
 *@numThres 表示找到几个相同的值就不再寻找
 */
int StereoMatching::backgroundInterpolateCore(Mat& Dp, int v, int u)
{
	const float color_thre = 40;
	int neigh_num = param_.bgIpDir;
	const int searchDepth = param_.bgIplDepth;
	int dv[] = {0, 0, -1, 1};
	int du[] = {-1, 1, 0, 0};

	short* aimP = Dp.ptr<short>(v);
	short neighVal = -1;
	vector<float> initial(2, -1);
	vector<vector<float>> candiDispCost(neigh_num, initial);
	//uchar* IcP = HSV.ptr<uchar>(v, u);
	//uchar* IcP = Lab.ptr<uchar>(v, u);
	//Mat I_tem;
	//medianBlur(I_c[0], I_tem, 3);
	//uchar* IcP = I_tem.ptr<uchar>(v, u);
	Mat I_aim = I_c[0];
	uchar* IcP = I_aim.ptr<uchar>(v, u);
	int securityNum = 1;
	vector<int> dispContainer;
	int neigh_disp_limit = 2;
	

	for (int dire = 0; dire < neigh_num; dire++)
	{
		for (int dep = 1; dep <= searchDepth; dep++)
		{
			for (int num = 0; num < securityNum; num++)
			{
				int v_nei = v + (dep + num) * dv[dire];
				int u_nei = u + (dep + num) * du[dire];
				if (v_nei >= 0 && v_nei < h_ && u_nei >= 0 && u_nei < w_)
				{
					neighVal = Dp.ptr<short>(v_nei)[u_nei];
					if (neighVal >= 0)
					{

						if (dispContainer.empty())
						{
							dispContainer.push_back(neighVal);
						}
						else
						{
							int vec_n = dispContainer.size();
							if (abs(dispContainer[vec_n - 1] - neighVal) < neigh_disp_limit)
								dispContainer.push_back(neighVal);
							else
							{
								dispContainer.clear();
								break;
							}
						}
					}
				}
				else
					break;
			}
			if (dispContainer.size() == securityNum)
			{
				uchar* I_neigh = I_aim.ptr<uchar>(v + dep * dv[dire], u + dep * du[dire]);
				int dif = 0;
				for (int c = 0; c < 3; c++)
					dif = max(dif, abs(I_neigh[c] - IcP[c]));
				int disp = Dp.ptr<short>(v + dep * dv[dire])[u + dep * du[dire]];
				candiDispCost[dire][0] = disp;
				candiDispCost[dire][1] = dif;
				dispContainer.clear();
				break;
			}
		}
	}

	int disp_nei = 10000;
	int i = 0;
	int j = -1;
	int num_result = -1;
	int color_dif;
	for (; i < candiDispCost.size(); i++)
	{
		if (candiDispCost[i][0] >= 0 && candiDispCost[i][0] < disp_nei)
		{
			disp_nei = candiDispCost[i][0];
			j = i;
		}
	}

	if (j >= 2)
	{
		int z = j;
		int disp_ = 10000;
		for (int i = 0; i < 2; i++)
		{
			if (candiDispCost[i][0] >= 0 && candiDispCost[i][0] < disp_)
			{
				z = i;
				disp_ = candiDispCost[i][0];
			}

		}
		if (candiDispCost[z][1] < candiDispCost[j][1])
			j = z;
	}
	return j == -1 ? -1 :  candiDispCost[j][0];
}

void StereoMatching::RV_combine_BG(cv::Mat& Dp, float rv_ratio, int rv_s)
{
	cout << "start RV_combine_BG_IP" << endl;
	CV_Assert(Dp.type() == CV_16S);
	const int n = param_.numDisparities;
	const int SThres = rv_s;  // 15
	const float hratioThres = rv_ratio;  // 0.4

	const int bgiplDepth = param_.bgIplDepth; 

	//OMP_PARALLEL_FOR  加上后误差增大，原因未知
	Mat dp_res = Dp.clone();
	int dp_ = -1;
	int dp_bg = -1, dp_rv = -1;
	for (int v = 0; v < h_; v++)
	{
		short* dpP = Dp.ptr<short>(v);
		short* dp_resP = dp_res.ptr<short>(v);
		for (int u = 0; u < w_; u++)
		{
			short dpV = dpP[u];
			if (dpV < 0)
				{
					if (param_.interpolateType == 0)  // 区域投票
						dp_ = regionVoteCore(Dp, v, u, SThres, hratioThres);
					else if (param_.interpolateType == 1)  // 背景插值
						dp_ = backgroundInterpolateCore(Dp, v, u);
					else if (param_.interpolateType == 2) // 在遮挡区用背景插值，误匹配区用区域投票
					{
						if (dpV == param_.DISP_OCC)
							dp_ = backgroundInterpolateCore(Dp, v, u);
						else if (dpV == param_.DISP_MIS)
							dp_ = regionVoteCore(Dp, v, u, SThres, hratioThres);
					}
					else if (param_.interpolateType == 3)
					{
						if (dpV == param_.DISP_OCC)
						{
							dp_bg = backgroundInterpolateCore(Dp, v, u);
							//dp_bg = properIpolCore(Dp, v, u);
							dp_rv = regionVoteCore(Dp, v, u, SThres, hratioThres);
							if (dp_bg >= 0 && dp_rv < 0)
								dp_ = dp_bg;
							else if (dp_bg < 0 && dp_rv >= 0)
								dp_ = dp_rv;
							else if (dp_bg >= 0 && dp_rv >= 0)
								dp_ = dp_rv <= dp_bg ? dp_rv : dp_bg;  // dp_rv为bg_rv1, dp_bg为bg_rv2
							else
								dp_ = -1;
							//else
							//	dp_ = dp_rv;
						}
						else if (dpV == param_.DISP_MIS)
							dp_ = regionVoteCore(Dp, v, u, SThres, hratioThres);
					}
					if (dp_ >= 0)
						dp_resP[u] = dp_;
						//dpP[u] = dp_;
				}
		}
	}
	dp_res.copyTo(Dp);
	cout << "finish RV_combine_BG" << endl;
}

void StereoMatching::BGIpol(cv::Mat& Dp)
{
	for (int v = 0; v < h_; v++)
	{
		short* vDptr = Dp.ptr<short>(v);
		for (int u = 0; u < w_; u++)
		{
			if (vDptr[u] < 0)
			{
				int dp_ = backgroundInterpolateCore(Dp, v, u);
				if (dp_ >= 0)
					vDptr[u] = dp_;
			}
		}
	}
}

void StereoMatching::properIpol(cv::Mat& Dp, cv::Mat& I1_c)
{
	CV_Assert(Dp.type() == CV_16S);

	const int h = Dp.rows;
	const int w = Dp.cols;
	const int searchDepth = 20;

	Mat DpCopy(h, w, CV_16S);

	int directionsW[] = { 0, 2, 2, 2, 0, -2, -2, -2, 1, 2, 2, 1, -1, -2, -2, -1 };
	int directionsH[] = { 2, 2, 0, -2, -2, -2, 0, 2, 2, 1, -1, -2, -2, -1, 1, 2 };

	for (int v = 0; v < h; v++)
	{
		short* vDptr = Dp.ptr<short>(v);
		short* vDcoptr = DpCopy.ptr<short>(v);
		for (int u = 0; u < w; u++)
		{
			if (vDptr[u] >= 0)
				vDcoptr[u] = vDptr[u];
			if (vDptr[u] < 0)
			{
				vector<int> directionDisp(16, vDptr[u]);
				vector<int> directionDiff(16, -1);
				for (int direction = 0; direction < 16; direction++)
				{
					int ph = directionsH[direction];
					int pw = directionsW[direction];
					int posw = u;
					int posh = v;
					bool inner = true;
					for (int dep = 0; dep < searchDepth; dep++)
					{
						if (dep % 2 == 0)
						{
							posw += pw / 2;
							posh += ph / 2;
						}
						else
						{
							posw += pw - pw / 2;
							posh += ph - ph / 2;
						}
						inner = posw >= 0 && posw < w && posh >= 0 && posh < h;
						if (!inner)
							break;
						else
						{
							if (Dp.ptr<short>(posh)[posw] >= 0)
							{
								directionDisp[direction] = Dp.ptr<short>(posh)[posw];
								int Cdif = 0;
								for (int c = 0; c < 3; c++)
								{
									int channelDif = abs(I1_c.ptr<uchar>(v, u)[c] - I1_c.ptr<uchar>(posh, posw)[c]);
									if (channelDif > Cdif)
										Cdif = channelDif;
								}
								directionDiff[direction] = Cdif;
								break;
							}
						}
					}
				}
				if (vDptr[u] == param_.DISP_OCC)  // param.DISP_OCC = -2 * 16
				{
					int minDisp = numeric_limits<int>::max();
					int init = minDisp;
					for (int direction = 0; direction < 16; direction++)
					{
						if (directionDisp[direction] >= 0 && minDisp > directionDisp[direction])
							minDisp = directionDisp[direction];
					}
					vDcoptr[u] = init != minDisp ? minDisp : vDptr[u];  // 防止没有找到的情况
				}
				else
				{
					int minDifColor = 255;
					int disp = -1;
					for (int direction = 0; direction < 16; direction++)
					{
						if (directionDiff[direction] >= 0 && minDifColor > directionDiff[direction])
						{
							minDifColor = directionDiff[direction];
							disp = directionDisp[direction];
						}
					}
					vDcoptr[u] = disp >= 0 ? disp : vDptr[u];
				}

			}
		}
	}
	DpCopy.copyTo(Dp);
}

int StereoMatching::properIpolCore(cv::Mat& Dp, int v, int u)
{
	CV_Assert(Dp.type() == CV_16S);

	const int searchDepth = 20;

	int directionsW[] = { 0, 2, 2, 2, 0, -2, -2, -2, 1, 2, 2, 1, -1, -2, -2, -1 };
	int directionsH[] = { 2, 2, 0, -2, -2, -2, 0, 2, 2, 1, -1, -2, -2, -1, 1, 2 };
	short disp = Dp.ptr<short>(v)[u];
	vector<int> directionDisp(16, disp);
	vector<int> directionDiff(16, -1);
	for (int direction = 0; direction < 16; direction++)
	{
		int ph = directionsH[direction];
		int pw = directionsW[direction];
		int posw = u;
		int posh = v;
		bool inner = true;
		for (int dep = 0; dep < searchDepth; dep++)
		{
			if (dep % 2 == 0)
			{
				posw += pw / 2;
				posh += ph / 2;
			}
			else
			{
				posw += pw - pw / 2;
				posh += ph - ph / 2;
			}
			inner = posw >= 0 && posw < w_ && posh >= 0 && posh < h_;
			if (!inner)
				break;
			else
			{
				if (Dp.ptr<short>(posh)[posw] >= 0)
				{
					directionDisp[direction] = Dp.ptr<short>(posh)[posw];
					int Cdif = 0;
					for (int c = 0; c < 3; c++)
					{
						int channelDif = abs(I_c[0].ptr<uchar>(v, u)[c] - I_c[0].ptr<uchar>(posh, posw)[c]);
						if (channelDif > Cdif)
							Cdif = channelDif;
					}
					directionDiff[direction] = Cdif;
					break;
				}
			}
		}
	}
	if (disp == param_.DISP_OCC)  // param.DISP_OCC = -2 * 16
	{
		int minDisp = numeric_limits<int>::max();
		int init = minDisp;
		for (int direction = 0; direction < 16; direction++)
		{
			if (directionDisp[direction] >= 0 && minDisp > directionDisp[direction])
				minDisp = directionDisp[direction];
		}
		disp = init != minDisp ? minDisp : disp;  // 防止没有找到的情况
	}
	else
	{
		int minDifColor = 255;
		int disp_ = -1;
		for (int direction = 0; direction < 16; direction++)
		{
			if (directionDiff[direction] >= 0 && minDifColor > directionDiff[direction])
			{
				minDifColor = directionDiff[direction];
				disp_ = directionDisp[direction];
			}
		}
		disp = disp_ >= 0 ? disp_ : disp;
	}
	return disp;
}

void StereoMatching::cbbi(Mat& Dp)
{
	clock_t start = clock();
	//const int borderL = 60;
	const int zvalue = 4;
	const int fvalue = 4;
	Scalar lowdifference = Scalar(zvalue, zvalue, zvalue);
	Scalar updifference = Scalar(fvalue, fvalue, fvalue);
	int flags = 8 + (1 << 8);
	Mat cutImg = I_c[0].clone();
	GaussianBlur(cutImg, cutImg, Size(7, 7), 4, 4);
	//GaussianBlur(cutImg, cutImg, Size(3, 3), 4, 4);
	//blur(cutImg, cutImg, Size(3, 3));
	//Mat imgEdg(h_ + 2, w_ + 2, CV_8U, Scalar::all(0));
	Mat imgEdg(h_, w_, CV_8U);
	execCanny(imgEdg);

	Rect rect;
	string path = param_.savePath;
	system(("IF NOT EXIST " + path + " (mkdir " + path + ")").c_str());
	path += "canny.png";
	imwrite(path, imgEdg);

	for (int v = 0; v < h_; v += 3)
	{
		for (int u = 0; u < w_; u += 10)
		{
			int b = (unsigned)theRNG() & 255;
			int g = (unsigned)theRNG() & 255;
			int r = (unsigned)theRNG() & 255;

			Scalar color = Scalar(b, g, r);
			floodFill(cutImg, imgEdg, Point(u, v), color, &rect, lowdifference, updifference, flags);
		}
	}
	clock_t end = clock();
	clock_t time = end - start;
	cout << "cbci cut" << endl;
	saveTime(time, "cbci cut");

	imwrite(param_.savePath + "cutImg.png", cutImg);

	int securityNum = 1;
	int searchDirections = 16;
	int directionX[] = { 1, 1, 1, 0, 0, -1, -1, -1, 2, 2, 1, -1, -2, -2, -1, 1 };
	int directionY[] = { 0, -1, 1, -1, 1, -1, 1, 0, 1, -1, -2, -2. - 1, 1, 2, 2 };

	//int directionX[] = {1, 0, 0, 1, 1};
	//int directionY[] = {0, 1, -1, -1, 1};

	int u0 = 0, u1 = w_ - 1, uCoefficient = 1;
	//for (int iteration = 0; iteration < 2; iteration++)
	//{
	//	clock_t start = clock();
	//	if (iteration == 1)
	//		u0 = u0 ^ u1, u1 = u0 ^ u1, u0 = u0 ^ u1, uCoefficient = -uCoefficient;
	//	for (int v = 0; v < h_; v++)
	//	{
	//		for (int u = u0; u != u1 + uCoefficient; u += uCoefficient)
	//		{
	//			if (Dp.ptr<short>(v)[u] < 0)
	//			{
	//				int satisfied_direc = 0;
	//				short disp_cache[3];
	//				uchar* tarP = cutImg.ptr<uchar>(v, u);
	//				int v_neigh = 0, u_neigh = 0;
	//				for (int direc = 0; direc < searchDirections; direc++)
	//				{
	//					bool has_find = true;
	//					short neighValue = -1;
	//					for (int dep = 1; dep <= securityNum; dep++)
	//					{
	//						int v_neigh = v + dep * directionY[direc];
	//						int u_neigh = u + dep * directionX[direc];
	//						bool is_inImg = v_neigh < h_ && v_neigh >= 0 && u_neigh >= 0 && u_neigh < w_;
	//						if (!is_inImg)
	//						{
	//							has_find = false;
	//							break;
	//						}
	//						uchar* neighP = cutImg.ptr<uchar>(v_neigh, u_neigh);
	//						bool inOneArea = true;
	//						for (int c = 0; c < 3; c++)
	//						{
	//							if (tarP[c] != neighP[c])
	//							{
	//								inOneArea = false;
	//								break;
	//							}
	//						}
	//							
	//						neighValue = Dp.ptr<short>(v_neigh)[u_neigh];
	//						if (inOneArea == false || neighValue < 0)
	//						{
	//							has_find = false;
	//							break;
	//						}
	//					}

	//					if (has_find)
	//					{
	//						disp_cache[satisfied_direc] = neighValue;
	//						satisfied_direc++;
	//						if (satisfied_direc == 1)
	//						{
	//							Dp.ptr<short>(v)[u] = disp_cache[0];
	//							break;
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
	//	if (iteration == 0)
	//		saveDispMap<short>(Dp, "06disp_cbbi0");

	//	clock_t end = clock();
	//	clock_t time = end - start;
	//	cout << "cbci interpolate " + to_string(iteration) << "time: " << time << endl;
	//	saveTime(time, "cbci interpolate");
	//}
	for (int iteration = 0; iteration < 2; iteration++)
	{
		clock_t start = clock();
		if (iteration == 1)
			u0 = u0 ^ u1, u1 = u0 ^ u1, u0 = u0 ^ u1, uCoefficient = -uCoefficient;
		for (int v = h_ - 1; v >= 0; v--)
		{
			for (int u = u0; u != u1 + uCoefficient; u += uCoefficient)
			{
				if (Dp.ptr<short>(v)[u] < 0)
				{
					int satisfied_direc = 0;
					short disp_cache[3];
					uchar* tarP = cutImg.ptr<uchar>(v, u);
					int v_neigh = 0, u_neigh = 0;
					for (int direc = 0; direc < searchDirections; direc++)
					{
						bool has_find = true;
						short neighValue = -1;
						for (int dep = 1; dep <= securityNum; dep++)
						{
							int v_neigh = v + dep * directionY[direc];
							int u_neigh = u + dep * directionX[direc];
							bool is_inImg = v_neigh < h_ && v_neigh >= 0 && u_neigh >= 0 && u_neigh < w_;
							if (!is_inImg)
							{
								has_find = false;
								break;
							}
							uchar* neighP = cutImg.ptr<uchar>(v_neigh, u_neigh);
							bool inOneArea = true;
							for (int c = 0; c < 3; c++)
							{
								if (tarP[c] != neighP[c])
								{
									inOneArea = false;
									break;
								}
							}

							neighValue = Dp.ptr<short>(v_neigh)[u_neigh];
							if (inOneArea == false || neighValue <= 0)
							{
								has_find = false;
								break;
							}
						}

						if (has_find)
						{
							disp_cache[satisfied_direc] = neighValue;
							satisfied_direc++;
							if (satisfied_direc == 1)
							{
								Dp.ptr<short>(v)[u] = disp_cache[0];
								break;
							}
						}
					}
				}
			}
		}
		if (iteration == 0)
			saveDispMap<short>(Dp, "06disp_cbbi0");

		clock_t end = clock();
		clock_t time = end - start;
		cout << "cbci interpolate " + to_string(iteration) << "time: " << time << endl;
		saveTime(time, "cbci interpolate");
	}
}

void StereoMatching::cutImg(Mat& imgCut)
{
	clock_t start = clock();
	//const int borderL = 60;
	const int zvalue = 4;
	const int fvalue = 4;
	Scalar lowdifference = Scalar(zvalue, zvalue, zvalue);
	Scalar updifference = Scalar(fvalue, fvalue, fvalue);
	int flags = 8 + (1 << 8);

	GaussianBlur(imgCut, imgCut, Size(7, 7), 4, 4);
	//Mat imgEdg(h_ + 2, w_ + 2, CV_8U, Scalar::all(0));
	Mat imgEdg(h_, w_, CV_8U);
	execCanny(imgEdg);

	string path = param_.savePath;
	system(("IF NOT EXIST " + path + " (mkdir " + path + ")").c_str());
	path += "canny.png";
	imwrite(path, imgEdg);

	for (int v = 0; v < h_; v += 3)
	{
		for (int u = 0; u < w_; u += 10)
		{
			int b = (unsigned)theRNG() & 255;
			int g = (unsigned)theRNG() & 255;
			int r = (unsigned)theRNG() & 255;

			Scalar color = Scalar(b, g, r);
			floodFill(imgCut, imgEdg, Point(u, v), color, 0, lowdifference, updifference, flags);
		}
	}
	clock_t end = clock();
	clock_t time = end - start;
	cout << "cbci cut" << endl;
	saveTime(time, "cbci cut");


	imwrite(param_.savePath + "cutImg.png", imgCut);
}

int StereoMatching::cbbi_core(Mat& Dp, Mat& cutImg, int v, int u)
{
	CV_Assert(Dp.type() == CV_16S);
	CV_Assert(cutImg.type() == CV_8UC3);

	int securityNum = 1;
	int searchDirections = 16;
	int directionX[] = { 1, 1, 1, 0, 0, -1, -1, -1, 2, 2, 1, -1, -2, -2, -1, 1 };
	int directionY[] = { 0, -1, 1, -1, 1, -1, 1, 0, 1, -1, -2, -2. - 1, 1, 2, 2 };

	int satisfied_direc = 0;
	short disp_cache[3];
	uchar* tarP = cutImg.ptr<uchar>(v, u);
	int v_neigh = 0, u_neigh = 0;
	for (int direc = 0; direc < searchDirections; direc++)
	{
		bool has_find = true;
		short neighValue = -1;
		for (int dep = 1; dep <= securityNum; dep++)
		{
			int v_neigh = v + dep * directionY[direc];
			int u_neigh = u + dep * directionX[direc];
			bool is_inImg = v_neigh < h_ && v_neigh >= 0 && u_neigh >= 0 && u_neigh < w_;
			if (!is_inImg)
			{
				has_find = false;
				break;
			}
			uchar* neighP = cutImg.ptr<uchar>(v_neigh, u_neigh);
			bool inOneArea = true;
			for (int c = 0; c < 3; c++)
			{
				if (tarP[c] != neighP[c])
				{
					inOneArea = false;
					break;
				}
			}

			neighValue = Dp.ptr<short>(v_neigh)[u_neigh];
			if (inOneArea == false || neighValue <= 0)
			{
				has_find = false;
				break;
			}
		}

		if (has_find)
		{
			disp_cache[satisfied_direc] = neighValue;
			satisfied_direc++;
			if (satisfied_direc == 1)
			{
			/*	Dp.ptr<short>(v)[u] = disp_cache[0];
				break;*/
				return neighValue;
			}
		}
	}

	return -1;
}

void StereoMatching::showParams()
{
	string::size_type idx;
	std::printf("LRCheck: %d\n", Do_LRConsis);
	std::printf("UniqCk: %d\n", UniqCk);
	std::printf("SubIpl: %d\n", SubIpl);
	printf("LRM: %d\n", LRM);
	idx = costcalculation.find("SSD");
	if (idx != string::npos)
		printf("ssdave: %d\n", Ssdave);
	printf("errorThreshold: %d\n", param_.errorThreshold);
	printf("LRmaxDiff: %f\n", param_.LRmaxDiff);
	printf("ipolThreshold: %d\n", param_.ipolThrehold);
	printf("medianKernelSize: %d\n", param_.medianKernelSize);
	printf("Last_MedBlur: %d\n", Last_MedBlur);
	printf("Last_MedBlur_Size: %d\n", param_.medianKernelSize_Last);
	printf("sgm_scanNum: %d\n", param_.sgm_scanNum);
	idx = aggregation.find("CBCA");
	if (idx != string::npos)
	{
		printf("cbca_minArmL: %d\n", param_.cbca_minArmL);
		printf("cbca_genArm_isColor: %d\n", cbca_genArm_isColor);
		printf("cbca_AD_isColor: %d\n", cbca_AD_isColor);
		printf("cbca_Cross_C: %d\n", param_.Cross_C);
		printf("cbca_Cross_L[0]: %d\n", param_.cbca_crossL[0]);
		printf("cbca_Cross_L[1]: %d\n", param_.cbca_crossL[1]);
		printf("cbca_arm_out: %d\n", cbca_arm_out);
		printf("cbca_iterationNum: %d\n", param_.cbca_iterationNum);

		printf("region_vote_nums: %d\n", param_.region_vote_nums);
		printf("regVote_SThres: %d\n", param_.regVote_SThres);
		printf("regVote_hratioThres: %f\n", param_.regVote_hratioThres);
	}
	cout << endl;
}