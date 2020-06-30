#pragma once
#include "util.h"
#include <stdio.h>
#include <cstdio>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <omp.h>
#include <streambuf>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <time.h>
#include <map>
#include <set>
#include "NL/NLCCA.h"

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

//#define JBF_STANDARD
#define DEBUG
//#define FAST_INV
//#define MY_GUIDE
//#define USE_RECONCV

using namespace std;
using namespace cv;

// 左右输入图、生成左右视差图，并且包含许多优化步骤的控制开关

class StereoMatching
{
public:

	static const std::string root;
	static std::string costcalculation; 
	static std::string aggregation; // CBCA, guideFilter, AWS
	static std::string optimization;
	static std::string object; 

	// 步骤开关控制
	static const bool UniqCk = 0;
	static const bool SubIpl = 0;
	static const bool Ssdave = 0;
	static const bool Do_sgm = 0;
	static const bool LRM = 1;  // 用于控制sgm在代价聚合的时候，是否包括左右边界部分
	static const bool Last_MedBlur = 1;
	static const bool cbca_AD_isColor = 1;  // AD代价是用彩图计算还是灰度图计算
	static const bool cbca_genArm_isColor = 1; // 是用彩色图产生的臂还是灰度图
	static const bool cbca_arm_out = 1; // 生成臂的时候是否加外层臂
	static const bool preMedBlur = 0; // 对彩色图是否进行中值滤波的预处理

	static const bool Do_dispOptimize = 1;
	// wxy
	static const bool Do_refine = 0;
	static const bool hv_nolimt = 0;
	static const bool Do_LRConsis = 1;
	static const bool Do_calPKR = 0;
	static const bool Do_WM = 0;
	static const bool Do_regionVote = 1;
	static const bool Do_properIpol = 1;
	static const bool Do_bgIpol = 0;
	static const bool Do_discontinuityAdjust = 0;  //经实验，DA在teddy、cones、venus、tsukuba中都会增大误差
	static const bool Do_subpixelEnhancement = 0;  //经实验，SE在上述四图也增大误差
	static const bool Do_lastMedianBlur = 1;
	static const bool Do_cbbi = 0;
	static const bool is_HVL_Change = 0; // CBCA代价聚合的区域（臂长）是每次迭代都一样还是不同的（默认是一样的）
	static const bool Do_cross2Vm = 0;

	struct Parameters
	{
		// 参数
		int W_U;
		int	W_V;
		int numDisparities;
		float uniquenessRatio_2small;
		float uniquenessRatio_2big;
		float LRmaxDiff;
		int medianKernelSize;
		int medianKernelSize_Last;
		int DISP_INV;
		int DISP_OCC;
		int DISP_MIS;
		int DISP_PKR;
		int DISP_SCALE;
		int DISP_SHIFT;
		float ZNCC_DEFAULT_MC;
		ushort AD_DEFAULT_DIF;
		bool ChooseSmall; // vm中最小代价为最相似还是最大代价最相似，true代表最小（默认），false代表最大（用于原始的ZNCC）

		int errorThreshold;
		int ipolThrehold;

		int sgm_scanNum;
		float sgm_P1;
		float sgm_P2;
		int sgm_corDifThres;
		int sgm_reduCoeffi1;
		int sgm_reduCoeffi2;

		int SD_AD_channel;
		int census_channel;
		int censusFunc;
		int census_W[2];
		bool is_censusNorm;
		bool is_adNorm;
		bool is_gradNorm;

		bool adCensus_useAdpWgt;
		bool adGrad_useAdpWgt;

		bool gradFuse_adpWgt;
		bool grad_use2direc;

		bool has_initArm;
		bool has_calArms;
		bool has_initTileArm;
		bool has_calTileArms;
		int cbcaTrunc_MC_gray;
		int cbcaTrunc_MC_color;
		int cbca_minArmL;
		int cbca_iterationNum;
		int cbca_ad_channels;
		bool cbca_intersect;
		uchar cbca_crossL[3];
		uchar cbca_crossL_out[3];
		uchar cbca_cTresh[3];
		uchar cbca_cTresh_out[3];
		bool cbca_double_win;
		bool GF_double_win;
		bool cbca_use_adpArm;
		int cbca_armDirec; // 0 代表水平和垂直，1代表倾斜45度的两个方向
		int cbca_armHV;
		int cbca_armTile;
		int cbca_armCombineType;
		bool cbca_armTile_addNeigh;
		bool cbca_do_limitArmSmallLen;
		bool cbca_limitArmSamllLenFirst;
		int cbca_box_WV;
		int cbca_box_WU;
		int cbca_armSmallLimit;
		int cobineCrossFWType;
		int armLSum;
		int armLSingle;
		bool cbca_do_costCompare;
		int	cbca_armSLimit;
		bool doGF_bef_calArm;

		int gf_r[2];
		float gf_eps[2];
		bool gf_channel_isColor;

		bool FIF_neighLimit;

		int region_vote_nums;
		int regVote_SThres;
		float regVote_hratioThres;
		int regVote_type;
		string RVdir;
		int bgIplDepth;
		int bgIpDir;
		bool cbbi_inner;
		int interpolateType;
		int err_ip_dispV;
		int cor_ip_dispV;

		bool Do_vmTop;
		int vmTop_method;
		int vmTop_Num;
		float vmTop_thres;
		int vmTop_thres_dirNum;
		bool vmTop_hasCir2;
		bool vmTop_cir3_doColorLimit;
		int sign_v_range[2];
		int sign_u_range[2];

		std::string savePath;
		std::string err_name;
		bool saveErr2CsvOnlyAll;
		bool saveErr2CsvOnlyNonocc;
		string errCsvName;

		int lamG;
		int lamCen;

		int ts;
		int disSc;

		Parameters(int maxDisp, int h, int w, int lamCen_, int lamG_, int M_, int lamc_, int ts_, string errCsvName_, int disSc_)
		{
			W_U = 4;
			W_V = 3;
			ChooseSmall = true;
			numDisparities = maxDisp + 1;
			uniquenessRatio_2small = 0.95;
			uniquenessRatio_2big = 1.05;
			LRmaxDiff = 0; // 5、10、15、20、30
			medianKernelSize = 3;  // 发现中值滤波核设为5时，错误率比3和7都低
			medianKernelSize_Last = 5;
			DISP_INV = -16;
			DISP_OCC = -2 * 16;
			DISP_MIS = -3 * 16;
			DISP_PKR = -4 * 16;

			DISP_SCALE = 16;
			DISP_SHIFT = 4;
			ZNCC_DEFAULT_MC = -1.0;
			AD_DEFAULT_DIF = 65535;

			errorThreshold = 1;
			ipolThrehold = 0;

			SD_AD_channel = 3;
			census_channel = 1;
			census_W[0] = 2;
			census_W[1] = 5;

			adCensus_useAdpWgt = 0;
			adGrad_useAdpWgt = 0;

			sgm_scanNum = 4;  //4
			sgm_P1 = 10000;  // ssd: 10、 2500、110000、200,  census:10
			sgm_P2 = 10000;  // ssd: 120、 30600、16386300、10000,  census:120
			sgm_corDifThres = 15;
			sgm_reduCoeffi1 = 4; // 4
			sgm_reduCoeffi2 = 10;  // 10
		
			// wxy
			censusFunc = 3; // 0代表中心像素点比，1代表前一个像素点和后一个比，2代表两者的结合，3代表传统Census后接上最内圈前后比较编码
			gradFuse_adpWgt = 1;
			grad_use2direc = 1;
			is_censusNorm = 0;
			is_adNorm = 0;
			is_gradNorm = 0;

			has_initArm = 0;
			has_calArms = 0;

			has_initTileArm = 0;
			has_calTileArms = 0;
			// 彩色时Cross_L、Cross_C、cbcaTrunc_MC_color的默认值分别为17，20、60
			cbcaTrunc_MC_gray = 60;
			cbcaTrunc_MC_color = 60;
			cbca_minArmL = 1;
			cbca_iterationNum = 2;
			cbca_ad_channels = 3;
			cbca_intersect = true;
			cbca_crossL[0] = 17;  // 17
			cbca_crossL[1] = 23;  // 12, 6, 17
			cbca_crossL[2] = 34;
			cbca_crossL_out[0] = 34;  // 34
			cbca_crossL_out[1] = 23;  // 22, 11, 34
			cbca_crossL_out[2] = 34;
			cbca_cTresh[0] = 20;
			cbca_cTresh[1] = 30;
			cbca_cTresh[2] = 30;
			cbca_cTresh_out[0] = 6;
			cbca_cTresh_out[1] = 0;
			cbca_cTresh_out[2] = 0;
			cbca_double_win = false;
			GF_double_win = false;
			cbca_use_adpArm = false;
			cbca_armDirec = 1;
			cbca_armHV = 1;
			cbca_armTile = 0;
			cbca_armCombineType = 0; // 0 代表根据非交臂长来选， 1代表根据交的臂长来选，2代表根据非交面积来选，3代表根据交面积来选
			cbca_armTile_addNeigh = false;
			cbca_do_limitArmSmallLen = false;  // 代价聚合改进控制开关
			cbca_limitArmSamllLenFirst = true;
			cbca_box_WV = 1;
			cbca_box_WU = 1;
			cbca_do_costCompare = true;
			
			cbca_armSmallLimit = 12;
			cbca_armSLimit = 10;
			cobineCrossFWType = 4; // 1代表根据交的臂长总和来选择，3代表根据交的面积来选, 4代表通过交的总臂长和单个臂长来选
			armLSum = 8;
			armLSingle = 6;
			doGF_bef_calArm = false;


			gf_r[0] = 9;  // 13
			gf_eps[0] = 0.0001;
			gf_r[1] = 27;
			gf_eps[1] = 0.0001;
			gf_channel_isColor = true;

			FIF_neighLimit = 0;

			
			region_vote_nums = 2;
			regVote_SThres = 20;
			regVote_hratioThres = 0.4;
			regVote_type = 0;  // 0代表用HV的，1代表tile的，2代表HV+tlie的
			interpolateType = 0;  // 0代表只用RV，1代表只用BG，2代表RV+BG, 3代表改进版RV+BG
			bgIplDepth = 1000;
			bgIpDir = 2;
			RVdir = "1Dir";
			cbbi_inner = false;

			err_ip_dispV = -50;
			cor_ip_dispV = -100;

			// wxy
			Do_vmTop = false;
			vmTop_method = 0; // 10
			vmTop_Num = M_; // 6
			vmTop_thres = lamc_ * 0.01; //1.08
			vmTop_thres_dirNum = 8;
			vmTop_hasCir2 = true;
			vmTop_cir3_doColorLimit = false;
			// 括号中为左上角和右下角的坐标(x,y)，teddy_cbca_4_1.05:(347,6)（407，27）;(257,43)(296,103);(0,370)(450,372);(0,360)(450,365);(0,361)(449,370)
			// Reindeer_cbca_4_1.05:(206, 222)（270，250）;
			// Baby2_cbca_4_1.05:(80,223)(135,339)
			// Wood2_cbca_4_1.05:(310,39)(407,86);(150,286)(246,328)
			// Cones_cbca_4_1.05:(46,300)(76,359)
			// Books_cbca_4_1.05:(60,325)(149,350);(204,216)(242,240);(201,44)(278,86)
			// Cloth2:(98,265)(157,322)
			// Bowling1:(115,54)(204,85)
			sign_v_range[0] = 361; // teddy: h-14 venus:91	teddy_GF_1.08_0:91, teddy_GF_1.08_1:358 teddy_CBCA_1.08_0:61,teddy_CBCA_1.08_1:358, teddy_cbca_4_1.05:
			sign_v_range[1] = 370; // teddy: h-11 venus:92	teddy_GF_1.08_0:94, teddy_GF_1.08_1:361 	teddy_CBCA_1.08_0:64,teddy_CBCA_1.08_0:361
			sign_u_range[0] = 0; // teddy:0 venus:0			teddy_GF_1.08_0:0
			sign_u_range[1] = 449; // teddy:w	venus:w			teddy_GF_1.08_0:w

			lamCen = lamCen_;
			lamG = lamG_;
			ts = ts_;
			// wxy
			errCsvName = errCsvName_;
			disSc = disSc_;
			saveErr2CsvOnlyAll = true;
			saveErr2CsvOnlyNonocc = true;
			savePath = root + object + "\\" + costcalculation + "-" +aggregation + "-" + optimization + "\\" + "20200627_test_so" + "\\";
			err_name = "1.txt";
		}  
	};
  // construct
	StereoMatching(cv::Mat& I1_c, cv::Mat& I2_c, cv::Mat& I1_g, cv::Mat& I2_g, cv::Mat& DT, cv::Mat& all_mask, cv::Mat& nonocc_mask, cv::Mat& disc_mask, const Parameters& param);

public:

	void censusGrad(vector<Mat>& vm);

	void adGrad(vector<Mat>& vm);

	void asdCal(vector<Mat>& vm_asd, string method, int imgNum, float Trunc);

	void bt(vector<Mat>& vm);

	void bt_color(vector<Mat>& vm);

	void calCostForBT(Mat& vm, vector<Mat>& grayNei, vector<Mat>& I_g, int LOR);

	void calNeiMaxMin(vector<Mat>& grayNei);

	void calNeiMaxMin_color(vector<Mat>& colorNei);

	void calgradvm(Mat& vm, vector<Mat>& grad, vector<Mat>& grad_y, int num, float Trunc);

	void calgradvm_mag_and_phase(Mat& vm, vector<Mat>& grad_xy, vector<Mat>& grad_atan, int num, float Trunc);

	void calgradvm_1d(Mat& vm, vector<Mat>& grad, int num, float trunc = 2);

	void calGrad(Mat& grad, Mat& img);

	void calGrad_y(Mat& grad, Mat& img);

	void grad(vector<Mat>& vm_grad, float Trunc);

	void grad_color(vector<Mat>& vm_grad, float Trunc);

	void combineXYGrad(Mat& grad_x, Mat& grad_y, Mat& grad_xy);

	void getAtanGrad(Mat& grad_x, Mat& grad_y, Mat& grad_atan);

	void truncAD(vector<Mat>& vm);

	void censusCal(vector<Mat>& vm_census, float truncRatio);

	void ADCensusCal();
	
	void ADCensusGrad();

	int HammingDistance(uint64_t c1, uint64_t c2);

	void pipeline();

	void openCSV();
	void closeCSV();

	void costCalculate();

	void dispOptimize();

	void refine();

	void fillSurronding(cv::Mat& D1, cv::Mat& D2);

	// AD、SD
	void gen_ad_sd_vm(Mat& asd_vm, int LOR, int AOS, float trunc = 1000000);

	void gen_ad_vm(Mat& vm, Mat& I_l, Mat& I_r, float Trunc, int LOR);

	void gen_truncAD_vm(Mat& truncAD_vm, int LOR);

	template <int LorR = 0>
	void gen_ad_vm(cv::Mat & adVm)
	{
		CV_Assert(adVm.type() == CV_32F);
		const int n = param_.numDisparities;
		const float DEFAULT_MC = 255.;

		int leftMove = 0;
		int rightMove = -1;
		if (LorR == 1)
		{
			leftMove = 1;
			rightMove = 0;
		}

		OMP_PARALLEL_FOR
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* vmPtr = adVm.ptr<float>(v, u);
				
				for (int d = 0; d < n; d++)
				{
					int u1 = u + d * leftMove;
					int u2 = u + d * rightMove;
					if (u1 >= w_ || u2 < 0)
					{
						vmPtr[d] = DEFAULT_MC;
						continue;
					}		
					float sum = 0;
					if (param_.cbca_ad_channels == 3)
					{
						uchar* I1Ptr = I_c[0].ptr<uchar>(v, u1);
						uchar* I2Ptr = I_c[1].ptr<uchar>(v, u2);
						for (int c = 0; c < 3; c++)
							sum += abs(I1Ptr[c] - I2Ptr[c]);
						sum /= 3;
					}
					else
					{
						sum += abs(I_g[0].ptr<uchar>(v)[u1] - I_g[1].ptr<uchar>(v)[u2]);
					}
					vmPtr[d] = sum;
				}
			}
		}
	};

	void SSD(std::string::size_type id_ssd);

	void gen_ssd_vm(cv::Mat& sd_vm, cv::Mat& ssd_vm);

	template <typename T>
	void caldisp_SSD(cv::Mat& ssd_vm, cv::Mat& D1, cv::Mat& D2)
	{
		const int h = ssd_vm.size[0];
		const int w = ssd_vm.size[1];
		const int n = ssd_vm.size[2];

		int W_U = param_.W_U;
		int W_V = param_.W_V;

		const float uniquenessRatio = param_.uniquenessRatio_2small;
		const int DISP_INV = param_.DISP_INV;
		const int DISP_SCALE = param_.DISP_SCALE;

		// 将视差图的外框区域设为不合理值
		fillSurronding(D1, D2);

		OMP_PARALLEL_FOR
			for (int v = W_V; v < h - W_V; v++)
			{
				short* _D1 = D1.ptr<short>(v);
				short* _D2 = D2.ptr<short>(v);

				for (int u = W_U; u < w - W_U; u++)
				{
					// cal left disp
					T minSSDL = std::numeric_limits<T>::max();
					T minSSDR = std::numeric_limits<T>::max();
					T* ssdPos = ssd_vm.ptr<T>(v, u);
					int dispL = 0, dispR = 0;
					for (int d = 0; d < n; d++)  // 这里本来导致了右视差图左侧的黑色区域，因为之前的循环停止条件是 d<n&& u - d >= W_U,限制了d取值范围，导致在图的左侧只能取到较小的视差 
					{
						if (ssdPos[d] < minSSDL)
						{
							minSSDL = ssdPos[d];
							dispL = d;
						}
						if (u + d < w - W_U)
						{
							if (ssd_vm.ptr<T>(v, u + d)[d] < minSSDR)
							{
								minSSDR = ssd_vm.ptr<T>(v, u + d)[d];
								dispR = d;
							}
						}
					}
					_D2[u] = static_cast<short>(dispR * DISP_SCALE);

					// uniqueness check
					if (UniqCk)
					{
						int d;
						for (d = 0; d < n; d++)
						{
							if (ssdPos[d] * uniquenessRatio < minSSDL && abs(dispL - d) > 1)
							{
								_D1[u] = DISP_INV;
								break;
							}
						}
						if (d < n)
							continue;
					}

					// subpixel interpolation
					if (SubIpl)
					{
						if (u - dispL > W_U && u - dispL < w - W_U - 1 && dispL > 0 && dispL < n)
							// 前两个条件保证窗口在右视差图的内部，如果窗口超出这个范围，因为超出范围的代价取了比较大的数值，和图片内正常计算的代价是不连续的，所以不用来做亚像素增强
						{
							const T numer = ssdPos[dispL - 1] - ssdPos[dispL + 1];
							const T denom = ssdPos[dispL - 1] - 2 * ssdPos[dispL] + ssdPos[dispL + 1];
							if (denom != 0)
							{
								dispL = dispL * DISP_SCALE + (DISP_SCALE * numer + denom) / (2 * denom);
								_D1[u] = static_cast<short>(dispL);
								break;
							}
						}

					}

					_D1[u] = static_cast<short>(dispL * DISP_SCALE);

				}
			}
	}

	// census
	void census_XOR(cv::Mat& censusCode0, Mat& censusCode1);

	template <int LorR = 0>
	void gen_cen_vm_(cv::Mat& vm)
	{
		CV_Assert(vm.depth() == CV_32F);

		const int W_U = param_.W_U;
		const int W_V = param_.W_V;
		const int DEFAULT_MC = (W_U * 2 + 1) * (W_V * 2 + 1) * param_.census_channel;

		const int n = param_.numDisparities;
		const int channels = param_.census_channel;

		int leftMove = 1;
		int rightMove = 0;
		if (LorR == 0)
		{
			leftMove = 0;
			rightMove = -1;
		}

		Mat I0, I1, IB0, IB1;
		I0 = channels == 3 ? I_c[0] : I_g[0];
		I1 = channels == 3 ? I_c[1] : I_g[1];

		copyMakeBorder(I0, IB0, W_V, W_V, W_U, W_U, BORDER_REFLECT_101);
		copyMakeBorder(I1, IB1, W_V, W_V, W_U, W_U, BORDER_REFLECT_101);

		//int v, u, d, dv, du;
		//#pragma omp parallel default (shared) private(v, u, d, dv, du) num_threads(omp_get_max_threads())
		//#pragma omp for schedule(static)
		//OMP_PARALLEL_FOR
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* cost = vm.ptr<float>(v, u);
				for (int d = 0; d < n; d++)
				{
					int u1 = u + leftMove * d;
					int u2 = u + rightMove * d;

					if (u1 < 0 || u2 >= w_)
						cost[d] = DEFAULT_MC;
					else
					{
						uchar* IB0P = IB0.ptr<uchar>(v + W_V, u1 + W_U);
						uchar* IB1P = IB1.ptr<uchar>(v + W_V, u2 + W_U);
						int dif = 0;
						for (int dv = -W_V; dv <= W_V; dv++)
						{
							for (int du = -W_U; du <= W_U; du++)
							{

								for (int c = 0; c < channels; c++)
								{
									if ((IB0P[c] - IB0.ptr<uchar>(v + W_V + dv, u1 + W_U + du)[c]) *
										(IB1P[c] - IB1.ptr<uchar>(v + W_V + dv, u2 + W_U + du)[c]) < 0)
										dif++;
								}
							}
						}
						cost[d] = dif;
					}
				}
			}
		}
	}

	template <typename T>
	void genCensusCode(vector<Mat>& I, vector<Mat>& census, int R_V, int R_U)
	{
		vector<Mat> I_B(2);
		vector<Mat> I_cache(2);
		//Mat I_cache;
		for (int i = 0; i < 2; i++)
		{
			copyMakeBorder(I[i], I_B[i], R_V, R_V, R_U, R_U, BORDER_REFLECT_101);
			//ximgproc::guidedFilter(I_B[i], I_B[i], I_B[i], R_V, 10);
			//bilateralFilter(I_B[i], I_cache[i], 7, 10, 10);
			//I_B[i] = I_cache[i];
		}
		const int channels = I[0].channels();
		//OMP_PARALLEL_FOR
		for (int num = 0; num < 2; num++)
		{
			for (int v = 0; v < h_; v++)
			{
				for (int u = 0; u < w_; u++)
				{
					T* IP = I_B[num].ptr<T>(v + R_V, u + R_U);
					uint64_t* censP = census[num].ptr<uint64_t>(v, u);
					uint64_t cs = 0;
					int step = 0, dep = 0;
					for (int dv = -R_V; dv <= R_V; dv++)
					{
						for (int du = -R_U; du <= R_U; du++)
						{
							//if (dv != 0 || du != 0)
							//{
								for (int c = 0; c < channels; c++)
								{
									if (step > 63)
									{
										censP[dep] = cs;
										cs = 0;
										step = 0;
										dep++;
									}
									cs <<= 1;
									if (IP[c] - I_B[num].ptr<T>(v + R_V + dv, u + R_U + du)[c] < 0)
										cs++;
									step++;
								}
							//}
						}
					}
					if (step > 0)
						censP[dep] = cs;
				}
			}

		}
	}

	// 前一个和后一个像素比较大小来获取编码(沿着每一圈从外往内进行)
	void genCensusCode_neighC1(vector<Mat>& I, vector<Mat>& census, int R_V, int R_U)
	{
		vector<Mat> I_B(2);
		for (int i = 0; i < 2; i++)
			copyMakeBorder(I[i], I_B[i], R_V, R_V, R_U, R_U, BORDER_REFLECT_101);
		const int channels = I[0].channels();
		//OMP_PARALLEL_FOR
		int loop_num = min(param_.W_V, param_.W_U);
		int cof = 0;
		for (int num = 0; num < 2; num++)
		{
			for (int v = 0; v < h_; v++)
			{
				for (int u = 0; u < w_; u++)
				{
					uint64_t* censP = census[num].ptr<uint64_t>(v, u);
					uint64_t cs = 0;
					int step = 0, dep = 0;
					int v_ct = v + R_V;
					int u_ct = u + R_U;
					for (int i = 0; i < loop_num; i++)
					{
						int V_pos = -(param_.W_V - i);
						int U_pos = -(param_.W_U - i);
						for (int loop_ = 0; loop_ < 4; loop_++)
						{
							if (loop_ % 2 == 0)
							{
								if (U_pos > 0) cof = -1; else cof = 1;
								for (int du = U_pos; du != -U_pos; du += cof)
								{
									int v_ = v_ct + V_pos;
									int u_ = u_ct + du;
									for (int c = 0; c < channels; c++)
									{
										if (step > 63)
										{
											censP[dep] = cs;
											cs = 0;
											step = 0;
											dep++;
										}
										cs <<= 1;
										if (I_B[num].ptr<uchar>(v_, u_)[c] - I_B[num].ptr<uchar>(v_, u_ + cof)[c] < 0)
											cs++;
										step++;
									}
								}
								U_pos = -U_pos;
							}
							else
							{
								if (V_pos > 0) cof = -1; else cof = 1;
								for (int dv = V_pos; dv != -V_pos; dv += cof)
								{
									int v_ = v_ct + dv;
									int u_ = u_ct + U_pos;
									for (int c = 0; c < channels; c++)
									{
										if (step > 63)
										{
											censP[dep] = cs;
											cs = 0;
											step = 0;
											dep++;
										}
										cs <<= 1;
										if (I_B[num].ptr<uchar>(v_, u_)[c] - I_B[num].ptr<uchar>(v_ + cof, u_)[c] < 0)
											cs++;
										step++;
									}

								}
								V_pos = -V_pos;
							}
						}
					}
					int u_remainer = param_.W_U - loop_num;
					for (int du = -u_remainer; du < u_remainer; du++)
					{
						for (int c = 0; c < channels; c++)
						{
							if (step > 63)
							{
								censP[dep] = cs;
								cs = 0;
								step = 0;
								dep++;
							}
							cs <<= 1;
							if (I_B[num].ptr<uchar>(v, u + du)[c] - I_B[num].ptr<uchar>(v, u + du + 1)[c] < 0)
								cs++;
							step++;
						}
					}
					if (step > 0)
						censP[dep] = cs;
				}
			}

		}
	}

	// 前一个和后一个像素比较大小来获取编码(蛇行)
	void genCensusCode_neighC2(vector<Mat>& I, vector<Mat>& census, int R_V, int R_U)
	{
		vector<Mat> I_B(2);
		for (int i = 0; i < 2; i++)
			copyMakeBorder(I[i], I_B[i], R_V, R_V, R_U, R_U, BORDER_REFLECT_101);
		const int channels = I[0].channels();
		//OMP_PARALLEL_FOR
		for (int num = 0; num < 2; num++)
		{
			for (int v = 0; v < h_; v++)
			{
				for (int u = 0; u < w_; u++)
				{
					uint64_t* censP = census[num].ptr<uint64_t>(v, u);
					uint64_t cs = 0;
					int step = 0, dep = 0;
					int v_start = v + R_V - R_V;
					int v_end = v + R_V + R_V;
					int u_ct = u + R_U;
					int u_start = u + R_U - R_U;
					int u_end = u + R_U + R_U;
					int cof = 1;
					for (int v_ = v_start; v_ <= v_end; v_++)
					{
						for(int u_ = u_start; u_ != u_end; u_ += cof)
						{
							for (int c = 0; c < channels; c++)
							{
								if (step > 63)
								{
									censP[dep] = cs;
									cs = 0;
									step = 0;
									dep++;
								}
								cs <<= 1;
								if (I_B[num].ptr<uchar>(v_, u_)[c] - I_B[num].ptr<uchar>(v_, u_ + cof)[c] < 0)
									cs++;
								step++;
							}
						}
						if (v_ < v_end)
						{
							for (int c = 0; c < channels; c++)
							{
								if (step > 63)
								{
									censP[dep] = cs;
									cs = 0;
									step = 0;
									dep++;
								}
								cs <<= 1;
								if (I_B[num].ptr<uchar>(v_, u_end)[c] - I_B[num].ptr<uchar>(v_ + 1, u_end)[c] < 0)
									cs++;
								step++;
							}
						}
						u_start ^= u_end;
						u_end = u_start ^ u_end;
						u_start = u_start ^ u_end;
						cof = -cof;
					}
					if (step > 0)
						censP[dep] = cs;
				}
			}

		}
	}

	// 在传统Census编码完成后，后面再跟上最内圈的前后像素的比较编码
	void genCensusCode_NC_Sur(vector<Mat>& I, vector<Mat>& census, int R_V, int R_U)
	{
		vector<Mat> I_B(2);
		for (int i = 0; i < 2; i++)
			copyMakeBorder(I[i], I_B[i], R_V, R_V, R_U, R_U, BORDER_REFLECT_101);
		const int channels = I[0].channels();
		int dv_sur[] = { -1, -1, -1, 0, 1, 1, 1, 0, -1 };
		int du_sur[] = { -1, 0, 1, 1, 1, 0, -1, -1 ,-1};
		int sur_num = 8;
		//OMP_PARALLEL_FOR
		for (int num = 0; num < 2; num++)
		{
			for (int v = 0; v < h_; v++)
			{
				for (int u = 0; u < w_; u++)
				{
					uchar* IP = I_B[num].ptr<uchar>(v + R_V, u + R_U);
					uint64_t* censP = census[num].ptr<uint64_t>(v, u);
					uint64_t cs = 0;
					int step = 0, dep = 0;
					for (int dv = -R_V; dv <= R_V; dv++)
					{
						for (int du = -R_U; du <= R_U; du++)
						{
							//if (dv != 0 || du != 0)
							//{
								for (int c = 0; c < channels; c++)
								{
									if (step > 63)
									{
										censP[dep] = cs;
										cs = 0;
										step = 0;
										dep++;
									}
									cs <<= 1;
									if (IP[c] - I_B[num].ptr<uchar>(v + R_V + dv, u + R_U + du)[c] < 0)
										cs++;
									step++;
								}
							//}
						}
					}
					for (int i = 0; i < sur_num; i++)
					{
						uchar* pre = I_B[num].ptr<uchar>(v + R_V + dv_sur[i], u + R_U + du_sur[i]);
						uchar* aft = I_B[num].ptr<uchar>(v + R_V + dv_sur[i + 1], u + R_U + du_sur[i + 1]);
						for (int c = 0; c < channels; c++)
						{
							if (step > 63)
							{
								censP[dep] = cs;
								cs = 0;
								step = 0;
								dep++;
							}
							cs <<= 1;
							if (pre[c] - aft[c] < 0)
								cs++;
							step++;
						}
					}
					if (step > 0)
						censP[dep] = cs;
				}
			}
		}
	}

	void gen_cenVM_XOR(vector<Mat>& census, Mat& cenVm, int codeLength, float truncRat, int LOR = 0)
	{
		float DEFAULT = codeLength * truncRat;
		float norm = param_.is_censusNorm ? 1 : DEFAULT; // 1

		CV_Assert(census[0].type() == CV_64F);
		CV_Assert(cenVm.depth() == CV_32F);
		memset(cenVm.data, 0, (uint64_t)h_ * w_ * param_.numDisparities * sizeof(float));
		const int disp = param_.numDisparities;

		const int varNum = census[0].size[2]; 
		//const int varNum = 1; 

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
				float* cenvmP = cenVm.ptr<float>(v, u);
				for (int d = 0; d < disp; d++)
				{
					int lp = u + d * leftCoefficient;
					int rp = u - d * rightCoefficient;
					if (lp >= w_ || rp < 0)
						cenvmP[d] = norm;
					else
					{
						float cost = 0;
						for (int varN = 0; varN < varNum; varN++)
						{
							cost += static_cast<float>(HammingDistance(census[0].ptr<uint64_t>(v, lp)[varN], census[1].ptr<uint64_t>(v, rp)[varN]));
						}
						cenvmP[d] = min(cost, DEFAULT);
						if (param_.is_censusNorm)
							cenvmP[d] /= DEFAULT;
					}
				}
			}
		}
	}

	void gen_cenVM_XOR_From2Code(vector<vector<Mat>>& census, Mat& cenVm, int codeLength[], int varNum[], float truncRatio, int LOR = 0)
	{
		CV_Assert(census[0][0].depth() == CV_64F);
		CV_Assert(census[1][1].depth() == CV_64F);
		CV_Assert(cenVm.depth() == CV_32F);
		//memset(cenVm.data, 0, (uint64_t)h_ * w_ * param_.numDisparities * sizeof(float));
		const int disp = param_.numDisparities;

		float DEFAULT[2] = {codeLength[0] * truncRatio, codeLength[1] * truncRatio};

		int leftCoefficient = 0, rightCoefficient = 1;
		if (LOR == 1)
		{
			leftCoefficient = 1;
			rightCoefficient = 0;
		}

		Mat I;
		I_g[LOR].convertTo(I, CV_32F);
		int size[] = { 5, 15 };
		vector<Mat> var(2);
		for (int i = 0; i < 2; i++)
		{
			Mat I_mean, I_2_mean, tem;
			boxFilter(I, I_mean, CV_32F, Size(size[i], size[i]));
			multiply(I, I, tem);
			boxFilter(tem, I_2_mean, CV_32F, Size(size[i], size[i]));
			multiply(I_mean, I_mean, tem);
			subtract(I_2_mean, tem, var[i]);
			//convertScaleAbs(res[i], res[i]);
		}



		for (int v = 0; v < h_; v++)
		{
			float* var0P = var[0].ptr<float>(v);
			float* var1P = var[1].ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				float var0 = var0P[u];
				float var1 = var1P[u];
				float ratio = var1 / (var0 + var1);
				float dif = var0 - var1;
				bool isBig = true;
				if (dif > 0 && dif / var1 > 0.7)
				{
					isBig = false;
				}

				short* arm = HVL[LOR].ptr<short>(v, u);
				short* armTile = tileCrossL[LOR].ptr<short>(v, u);

				int shortest = 100000;
				for (int dir = 0; dir < 4; dir++)
				{
					if (arm[dir] < shortest)
						shortest = arm[dir];
					//if (armTile[dir] < shortest)
					//	shortest = armTile[dir];
				}
				float numerator = -1.5;
				float a = 1 - exp(numerator / (shortest + 1));
				float b = 1 - a;
				float* cenvmP = cenVm.ptr<float>(v, u);
				{
					for (int d = 0; d < disp; d++)
					{
						int lp = u + d * leftCoefficient;
						int rp = u - d * rightCoefficient;
						if (lp >= w_ || rp < 0)
							cenvmP[d] = 1;
						else
						{
							float cost[2] = {0, 0};
							for (int i = 0; i < 2; i++)
							{
								float ct = 0;
								for (int varN = 0; varN < varNum[i]; varN++)
								{
									ct += static_cast<float>(HammingDistance(census[i][0].ptr<uint64_t>(v, lp)[varN], census[i][1].ptr<uint64_t>(v, rp)[varN]));
								}
								ct = min(ct, DEFAULT[i]);
								cost[i] = ct / DEFAULT[i];
							}
							//cenvmP[d] = a * cost[0] + b * cost[1];
							
							//cenvmP[d] = ratio * cost[0] + (1 - ratio) * cost[1];
							//cenvmP[d] = isBig ? cost[0] : cost[1];
							cenvmP[d] = cost[0];
						}
					}
				}
			}
		}
	}
	
	void gen_cenVM_XOR_From2Code_tem(Mat grad, vector<vector<Mat>>& census, Mat& cenVm, int codeLength[], int varNum[], int LOR = 0)
	{
		CV_Assert(census[0][0].depth() == CV_64F);
		CV_Assert(census[1][1].depth() == CV_64F);
		CV_Assert(cenVm.depth() == CV_32F);
		//memset(cenVm.data, 0, (uint64_t)h_ * w_ * param_.numDisparities * sizeof(float));
		const int disp = param_.numDisparities;

		int leftCoefficient = 0, rightCoefficient = 1;
		if (LOR == 1)
		{
			leftCoefficient = 1;
			rightCoefficient = 0;
		}

		int thres = 10;
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				short* arm = HVL[LOR].ptr<short>(v, u);
				short* armTile = tileCrossL[LOR].ptr<short>(v, u);
				float gradV = grad.ptr<float>(v)[u];
				int shortest = 10000;
				for (int dir = 0; dir < 4; dir++)
				{
					if (arm[dir] < shortest)
						shortest = arm[dir];
					//if (armTile[dir] < shortest)
					//	shortest = armTile[dir];
				}
				//float a = 1 - exp(-1.5 / (shortest + 1));
				//float b = 1 - a;				
				float a = 1 - exp(-0.1 / (gradV + 1));
				float b = 1 - a;
				float* cenvmP = cenVm.ptr<float>(v, u);
				{
					for (int d = 0; d < disp; d++)
					{
						int lp = u + d * leftCoefficient;
						int rp = u - d * rightCoefficient;
						if (lp >= w_ || rp < 0)
							cenvmP[d] = 1;
						else
						{
							float cost[2] = { 0, 0 };
							for (int i = 0; i < 2; i++)
							{
								for (int varN = 0; varN < varNum[i]; varN++)
								{
									cost[i] += static_cast<float>(HammingDistance(census[i][0].ptr<uint64_t>(v, lp)[varN], census[i][1].ptr<uint64_t>(v, rp)[varN]));
								}
								cost[i] /= codeLength[i];
							}
							//cenvmP[d] = shortest < thres ? cost[0] : cost[1];
							//cenvmP[d] = 0.7 * cost[0] + 0.3 * cost[1];
							cenvmP[d] = b * cost[0] + a * cost[1];
						}
					}
				}
			}
		}
	}


	void genCensus(cv::Mat& img, cv::Mat& censusCode, int R_V, int R_U);
	
	void genCensusVm(vector<cv::Mat>& census, Mat& cenVm, int LOR = 0)
	{
		const int n = param_.numDisparities;
		const int DM = (param_.W_U * 2 + 1) * (param_.W_V * 2 + 1);
		int leftCoe = 0, rightCoe = -1;
		if (LOR == 1)
			leftCoe = 1, rightCoe = 0;
		for (int v = 0; v < h_; v++)
		{
			uint64_t* c0 = census[0].ptr<uint64_t>(v);
			uint64_t* c1 = census[1].ptr<uint64_t>(v);
			for (int u = 0; u < w_; u++)
			{
				float* cenVmP = cenVm.ptr<float>(v, u);
				for (int d = 0; d < n; d++)
				{
					int u0 = u + leftCoe * d;
					int u1 = u + rightCoe * d;
					if (u0 >= w_ || u1 < 0)
						cenVmP[d] = DM;
					else
						cenVmP[d] = HammingDistance(c0[u0], c1[u1]);
				}
			}
		}
	}

	template <int VIEW = 0>
	void genSymCensus(cv::Mat & src, cv::Mat & dst)  // src为读入的原图，uchar类型
	{
		CV_Assert(dst.elemSize() == 8);
		memset(dst.data, 0, (uint64_t)dst.rows * dst.cols * sizeof(uint64_t));

		const int h = src.rows;
		const int w = src.cols;

		const int W_V = param_.W_V;
		const int W_U = param_.W_U;

		for (int v = W_V; v < h - W_V; v++)
		{
			for (int u = W_U; u < w - W_U; u++)

			{
				uint64_t* dstPtr = dst.ptr<uint64_t>(v);
				uint64_t c = 0;
				for (int dv = -W_V; dv <= W_V; dv++)
				{
					for (int du = -W_U; du <= W_U; du++)
					{
						c <<= 1;
						c += src.ptr<uchar>(v + dv)[u + du] <= src.ptr<uchar>(v - dv)[u - du] ? 1 : 0;
					}
				}
				if (VIEW == 0)
					dstPtr[u] = c;
				else
					dstPtr[w - 1 - u] = c;
			}
		}

	}

	void gen_vm_from2vm_exp(cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, const float ARU0, const float ARU1, int LOR);

	void gen_vm_from3vm_exp(cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, Mat& vm2, const float ARU0, const float ARU1, float ARU2, int LOR);

	void gen_vm_from3vm_add(cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, cv::Mat& vm2, int LOR);

	void gen_vm_from2vm_expadpWgt(Mat& HVL, cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, const float ARU0, const float ARU1, int LOR);

	void gen_vm_from2vm_add(cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, int LOR);

	void gen_vm_from2vm_fixWgt(Mat& vm0, float wgt0, Mat& vm1, float wgt1, Mat& dst);

	void gen_vm_from2vm_add_wgt(cv::Mat& HVL, cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, const float weight0, const float weight1, int LOR);

	void gen_adCenZNCC_vm(cv::Mat& adVm, cv::Mat& censVm, cv::Mat& znccVm, cv::Mat& adCenZnccVm, int LOR);

	// NCC
	void ZNCC(cv::Mat& I1, cv::Mat& I2, vector<cv::Mat>& vm);

	void cal_ave_std_ncc(cv::Mat& I, cv::Mat& E_std);

	template <int LOR = 0>
	void gen_NCC_vm(cv::Mat& I1, cv::Mat& I2, cv::Mat& E_std1, cv::Mat& E_std2, cv::Mat& CVM);

	// AWS
	void AWS();
	
	template <typename T>
	void initializeMat(Mat& mat, T value)
	{
		const int h = mat.size[0];
		const int w = mat.size[1];
		const int n = mat.size[2];

		//OMP_PARALLEL_FOR
		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				T* mP = mat.ptr<T>(v, u);
				for (int d = 0; d < n; d++)
					mP[d] = value;
			}
		}
	}

	template <typename int LOR = 0>
	void updateBorder_vm(Mat& vmTem, Mat& vmTem_Border, int borderInner, int borderOut, int W_U, int W_U_init)
	{
		const int n = param_.numDisparities;
		int u0 = W_U, u1 = borderInner - 1;
		for (int i = 0; i < 2; i++)
		{
			if (i == 1)
			{
				u0 = w_ - borderInner;
				u1 = w_ - W_U;
			}
			for (int v = 0; v < h_; v++)
			{
				for (int u = u0; u < u1; u++)
				{
					float* vmTP = vmTem.ptr<float>(v, u);
					float* vmTBP = vmTem_Border.ptr<float>(v, u);
					if (u < borderOut || u >= w_ - borderOut)
						for (int d = 0; d < n; d++)
							vmTP[d] = vmTBP[d];
					else if (LOR == 0 && u < borderInner || LOR == 1 && u > w_ - 1 - borderInner)
					{
						float costBorderMin = vmTBP[LOR], costMin = vmTP[LOR];
						float dispBorder = 0, disp = 0;
						for (int d = 1; d < n; d++)
						{
							if (vmTP[d] < costMin)
							{
								costMin = vmTP[d];
								disp = d;
							}
							if (vmTBP[d] < costBorderMin)
							{
								costBorderMin = vmTBP[d];
								dispBorder = d;
							}
						}
						if (dispBorder > disp && (LOR == 0 && u - dispBorder < W_U_init && u - dispBorder >= W_U
							|| LOR == 1 && u + dispBorder > w_ - 1 - W_U_init && u + dispBorder <= w_ - 1 - W_U))
							for (int d = 0; d < n; d++)
								vmTP[d] = vmTBP[d];
					}
				}
			}
		}

	}

	template <int LOR = 0>
	void calvm_AWS(int W_V, int W_U, Mat& vmTem, Mat (&wt)[2], Mat& vm_IB)
	{
		int S = (W_V * 2 + 1) * (W_U * 2 + 1);
		int num = 0, ratio = 1;
		int leftCoefficient = 0, rightCoefficient = -1;
		if (LOR == 1)
		{
			leftCoefficient = 1;
			rightCoefficient = 0;
		}

		for (int v = 0; v < h_; v++)
		{

			for (int u = 0; u < w_; u++)
			{
				float* vmTP = vmTem.ptr<float>(v, u);

				for (int d = 0; d < d_; d++)
				{
					int u1 = u + d * leftCoefficient;
					int u2 = u + d * rightCoefficient;
					if (u1 < w_ && u2 >= 0)
					{
						float* wtP1 = wt[0].ptr<float>(v, u1);
						float* wtP2 = wt[1].ptr<float>(v, u2);
						float numer = 0, denom = 0;
					
						int step = 0;
						for (int dv = -W_V; dv <= W_V; dv++)
						{
							for (int du = -W_U; du <= W_U; du++)
							{
								float ele = wtP1[step] * wtP2[step];
								denom += ele;
								numer += ele * vm_IB.ptr<float>(v + W_V + dv, u + W_U + du)[d];
								step++;
							}
						}
						vmTP[d] = numer / denom ;
					}
				}
			}
		}
	}

	void update_AWS_vm(Mat(&census_AWS_BORDER)[2], int borderL, int W_V_cen, int W_U_cen, int W_U_AWS, Mat& vm)
	{
		const int n = param_.numDisparities;
		const int depth = census_AWS_BORDER[0].size[2];
		const float DEFAULT_MC = (W_V_cen * 2 + 1) * (W_U_cen * 2 + 1) * 3;
		Mat vm_AWS_BORDER(3, size_vm, CV_32F);
		int u_start = W_U_cen, u_end = borderL;
		for (int side = 0; side < 2; side++)
		{
			if (side == 1)
			{
				u_start = w_ - 1 - W_U_AWS;
				u_end = w_ - 1 - W_U_cen;
			}
			for (int v = W_V_cen; v < h_ - W_V_cen; v++)
			{
				for (int u = u_start; u < u_end; u++)
				{
					float* vmP = vm_AWS_BORDER.ptr<float>(v, u);
					uint64_t* cenLP = census_AWS_BORDER[0].ptr<uint64_t>(v, u);
					for (int d = 0; d < n; d++)
					{
						if (u - d < 0)
							vmP[d] = DEFAULT_MC;
						else
						{
							uint64_t* cenRP = census_AWS_BORDER[1].ptr<uint64_t>(v, u - d);
							uint64_t cost = 0;
							for (int dep = 0; dep < depth; dep++)
								cost += HammingDistance(cenLP[dep], cenRP[dep]);
							vmP[d] = static_cast<float>(cost);
						}
					}
					float* vmInitP = vm.ptr<float>(v, u);
					if (u < W_U_AWS || u >= w_ - 1 - W_U_AWS)
						for (int d = 0; d < n; d++)
							vmInitP[d] = vmP[d];
					else
					{
						float costMin = vmP[0], costInitMin = vmInitP[0];
						float disp = 0, dispInit = 0;
						for (int d = 1; d < n; d++)
						{
							if (vmP[d] < costMin)
							{
								costMin = vmP[d];
								disp = d;
							}
							if (vmInitP[d] < costInitMin)
							{
								costInitMin = vmInitP[d];
								dispInit = d;
							}
						}
						if (disp > dispInit && disp < W_U_AWS)
							for (int d = 0; d < n; d++)
								vmInitP[d] = vmP[d];
					}
				}
			}
		}
	}

	void genCensus_AWS_BORDER(vector<Mat>& I_c, Mat(&census_AWS_BORDER)[2], int borderL, int W_V_cen, int W_U_cen)
	{
		const int h = census_AWS_BORDER[0].size[0];
		const int w = census_AWS_BORDER[0].size[1];
		const int depth = census_AWS_BORDER[0].size[2];
		const int n = param_.numDisparities;

		uint64_t cs = 0;
		int u_start = W_U_cen, u_end = borderL;
		for (int i = 0; i < 2; i++)
		{
			std::memset(census_AWS_BORDER[i].data, 0, (uint64_t)h * w * depth * sizeof(uint64_t));
			u_start = W_U_cen, u_end = borderL;
			for (int side = 0; side < 2; side++)
			{
				if (side == 1)
				{
					u_start = w_ - 1 - borderL;
					u_end = w_ - W_U_cen;
				}
				for (int v = W_V_cen; v < h - W_V_cen; v++)
				{
					for (int u = u_start; u < u_end; u++)
					{
						uchar* I_anchorP = I_c[i].ptr<uchar>(v, u);
						int dep = 0;
						uint64_t* cenP = census_AWS_BORDER[i].ptr<uint64_t>(v, u);
						int step = 0;
						cs = 0;
						for (int dv = -W_V_cen; dv <= W_V_cen; dv++)
						{
							for (int du = -W_U_cen; du <= W_U_cen; du++)
							{
								uchar* I_surP = I_c[i].ptr<uchar>(v + dv, u + du);
								for (int c = 0; c < 3; c++)
								{
									cs <<= 1;
									cs += I_anchorP[c] >= I_surP[c] ? 1 : 0;
									step++;
									if (step == 64)
									{
										cenP[dep] = cs;
										cs = 0;
										step = 0;
										dep++;
									}
								}
							}
						}
						if (step > 0)
							cenP[dep] = cs;
					}
				}
			}
		}
	}

	void genWeight_AWS(int W_V, int W_U,  Mat& wt, Mat& Lab)
	{
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				uchar* LabP1 = Lab.ptr<uchar>(v + W_V, u + W_U);
				int step = 0;
				float* wtP = wt.ptr<float>(v, u);
				for (int dv = -W_V; dv <= W_V; dv++)
				{
					for (int du = -W_U; du <= W_U; du++)
					{
						uchar* LabP2 = Lab.ptr<uchar>(v + W_V + dv, u + du + W_U);
						float weight = calW4_AWS(LabP1, LabP2, dv, du);
						wtP[step] = weight;
						step++;
					}
				}
			}
		}
	}

	template <int LOR = 0>
	void genTadVm(Mat (&I_IB)[2], Mat& vm_IB)
	{
		const int h_ = I_IB[0].rows;
		const int w_ = I_IB[0].cols;
		const int n = param_.numDisparities;
		const int DEF_MC = 40;

		int leftCoefficient = 0;
		int rightCoefficient = -1;
		if (LOR == 1)
		{
			leftCoefficient = 1;
			rightCoefficient = 0;
		}
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* vmP = vm_IB.ptr<float>(v, u);
				for (int d = 0; d < n; d++)
				{
					if (u + d * leftCoefficient >= w_ || u + d * rightCoefficient < 0)
						vmP[d] = DEF_MC;
					else
					{
						uchar* I1 = I_IB[0].ptr<uchar>(v, u + d * leftCoefficient);
						uchar* I2 = I_IB[1].ptr<uchar>(v, u + d * rightCoefficient);
						int difS = 0;
						for (int c = 0; c < 3; c++)
							difS += abs(I1[c] - I2[c]);
						vmP[d] = min(difS, DEF_MC);
					}
				}
			}
		}
	}

	float calW4_AWS(uchar* p, uchar* q, int dv, int du)
	{
		const float r_c = 5;
		const float r_p = 17.5;
		float d0 = static_cast<float> (p[0]) - static_cast<float>(q[0]);
		float d1 = static_cast<float> (p[1]) - static_cast<float>(q[1]);
		float d2 = static_cast<float> (p[2]) - static_cast<float> (q[2]);

		float dif_cpq = d0 * d0 * 0.153787 + d1 * d1 + d2 * d2; // 0.153787 = (100 / 255)^2
		dif_cpq = pow(dif_cpq, 0.5);
		float dif_gpd = pow(dv * dv + du * du, 0.5);
		//float w = exp(-(dif_cpq / r_c + dif_gpd / r_p));  // 发现包含距离权重比不包含效果要差
		float w = exp(-dif_cpq / r_c);

		return w;
	}

	void calPKR(Mat& vm, Mat& mask);

	void signDp_UsingPKR(Mat& disp, Mat& PKR_Err_Mask);

	void combine2Vm(vector<Mat>& vm, vector<Mat>& vm2);

	void combine2Vm_2(vector<Mat>& vm, vector<Mat>& vm2, vector<Mat>& HVL);

	void combine2Vm_3(vector<Mat>& vm, vector<Mat>& vm2);

	void combine2Vm_4(vector<Mat>& vm, vector<Mat>& vm2);

	// cross-based cost aggregation
	void CBCA();

	void GF();

	void GFNL();

	void guideFilter(int paramNum, vector<Mat>& vm);

	void FIF();

	void FIF_Improve();

	void NL();

	Mat guideFilterCore_matlab(Mat& I, Mat p, int r, float eps);

	void guideFilterCore(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon);

	Mat CumSum(const Mat& src, const int d);

	Mat BoxFilter(const Mat& imSrc, const int r);

	void vmTrans(vector<Mat>& vm, vector<vector<Mat>>& guildFilterVm);

	void vmTrans(vector<vector<Mat>>& guideVm, vector<Mat>& vm);

	void adCensus(vector<Mat>& vm_ad, vector<Mat>& vm_census);

	void adCensuGradCombine(vector<Mat>& adVm, vector<Mat>& cenVm, vector<Mat>& gradVm);

	void adCensusZncc(vector<Mat>& vm_ad, vector<Mat>& vm_census, vector<Mat>& vm_zncc);

	void cbca_aggregate(int param_Num, vector<Mat>& vm);

	void cbca_core(vector<Mat>& vm, vector<Mat>& HVL, vector<Mat>& HVL_INTERSECTION, int ITNUM);

	void initArm();
	void initTileArm();

	template <typename T>
	void calArms(vector<Mat>& I, vector<Mat>& cross, vector<Mat>& cross_intersec, int L, int L_out, int cTresh, int cTresh_out);

	template <typename T>
	void calArms(vector<Mat>& I, vector<Mat>& cross, vector<Mat>& cross_intersec, int L, int cTresh);

	template <typename T>
	void calArms(vector<Mat>& I, vector<Mat>& cross, vector<Mat>& cross_intersec, int L0, int L1, int L2, int tresh0, int tresh1, int tresh2);

	template <typename T>
	void calArms(vector<Mat>& I, vector<Mat>& cross, vector<Mat>& cross_intersec, vector<int> L, vector<int> thres);

	void showArms(int v, int u);

	void calTileArms();

	template <typename T>
	void calHorVerDis(Mat& I, Mat& cross, int L, int L_out, int C_D, int C_D_out, int minL);

	template <typename T>
	void calHorVerDis(Mat& I, Mat& cross, int L0, int L1, int L2, int thresh0, int thresh1, int thresh2, int minL);

	template <typename T>
	void calHorVerDis(Mat& img, Mat& cross, int L, int DIF, int minL);

	template <typename T>
	void calHorVerDis(Mat& I, Mat& cross, vector<int> L, vector<int> thresh, int minL);

	void calHorVerDis2(int imgNum, int channel, uchar L, uchar L_out, uchar C_D, uchar C_D_out, uchar minL);

	void calTileDis(int imgNum, int channel, uchar L, uchar L_out, uchar C_D, uchar C_D_out, uchar minL);

	void drawArmForPoint(Mat& HVL, int* v, int* u, int num);

	void calTileNeigh(int imgNum, int channels, cv::Mat& tile_neigh, uchar DIFThres);

	void genTrueHorVerArms(vector<Mat>& HVL, vector<Mat>& HVL_INTERSECTION);

	void gen1DCumu(cv::Mat& vm, cv::Mat& area, Mat& areaIS, int dv, int du);

	// direc指示head、tail连线的方向，0代表水平或者-45+135度，1代表垂直或者45+225度
	void cal1DCost(Mat & vm, cv::Mat & HVL, cv::Mat & area, Mat& areaIS, Mat& HVL_INTERSECTION, int dv, int du, int direc)
	{
		int head_num = direc * 2 + 1;
		int tail_num = direc * 2;
		CV_Assert(HVL.depth() == CV_16U);

		const int n = param_.numDisparities;

		Mat vmTemp(3, size_vm, CV_32F);
		Mat areaISTemp, areaTemp;
		if (param_.cbca_intersect)
			areaISTemp.create(3, size_vm, CV_32S);
		else
			areaTemp.create(h_, w_, CV_32S);
		int tail_u, tail_v, head_u, head_v, pre_tailU, pre_tailV;
		bool inner;
		int* areatemPtr = NULL;
		int* areaISP = NULL;
		//OMP_PARALLEL_FOR
		for (int v = 0; v < h_; v++)
		{
			if (!param_.cbca_intersect)
				areatemPtr = areaTemp.ptr<int>(v);
			for (int u = 0; u < w_; u++)
			{
				float* vmTprt = vmTemp.ptr<float>(v, u);
				if (!param_.cbca_intersect)
				{
					ushort* HVLPtr = HVL.ptr<ushort>(v, u);
					{
						tail_u = u + du * HVLPtr[tail_num];
						tail_v = v + dv * HVLPtr[tail_num];
						head_u = u - du * HVLPtr[head_num];
						head_v = v - dv * HVLPtr[head_num];
						pre_tailU = tail_u + du;
						pre_tailV = tail_v + dv;
						inner = (pre_tailU >= 0 && pre_tailU < w_ && pre_tailV >= 0 && pre_tailV < h_);
						if (inner)
							areatemPtr[u] = area.ptr<int>(head_v)[head_u] - area.ptr<int>(pre_tailV)[pre_tailU];
						else
							areatemPtr[u] = area.ptr<int>(head_v)[head_u];
					}
				}
				if (param_.cbca_intersect)
					areaISP = areaISTemp.ptr<int>(v, u);
				//OMP_PARALLEL_FOR
				for (int d = 0; d < n; d++)
				{
					if (param_.cbca_intersect)
					{
						ushort* HVLISP = HVL_INTERSECTION.ptr<ushort>(v, u, d);
						tail_u = u + du * HVLISP[tail_num];
						tail_v = v + dv * HVLISP[tail_num];
						head_u = u - du * HVLISP[head_num];
						head_v = v - dv * HVLISP[head_num];
						pre_tailU = tail_u + du;
						pre_tailV = tail_v + dv;
						inner = (pre_tailU >= 0 && pre_tailU < w_ && pre_tailV >= 0 && pre_tailV < h_);
						if (inner)
							areaISP[d] = areaIS.ptr<int>(head_v, head_u)[d] - areaIS.ptr<int>(pre_tailV, pre_tailU)[d];
						else
							areaISP[d] = areaIS.ptr<int>(head_v, head_u)[d];
					}
					vmTprt[d] = inner ? vm.ptr<float>(head_v, head_u)[d] - vm.ptr<float>(pre_tailV, pre_tailU)[d] : vm.ptr<float>(head_v, head_u)[d];
				}
			}
		}
		vmTemp.copyTo(vm);
		if (!param_.cbca_intersect)
			areaTemp.copyTo(area);
		else
			areaISTemp.copyTo(areaIS);
	}

	void genfinalVm_cbca(Mat& vm, Mat& area, Mat& areaIS, int imgNum);

	void combine_HV_Tilt(Mat& vm_HV, Mat& vm_Tile, Mat& area_HV, Mat& area_tile, Mat& areaIS_HV, Mat& areaIS_tile, int imgNum);

	void combine_Cross_FW(Mat& vm_dst, Mat& vm_BF, Mat& area, Mat& areaIS, int imgNum);

	void wta_Co(cv::Mat& vm, cv::Mat& D1, cv::Mat& D2);

	void cal_err(cv::Mat& DT, cv::Mat& DP, FILE* save_txt);

	void saveTime(int ms, string procedure)
	{
		if (_access(param_.savePath.c_str(), 0) == -1)  // 判断文件是否存在
			createDirectory(param_.savePath.c_str());
		string addr = param_.savePath + "time.txt";
		fstream fout;
		fout.open(addr, ios::app | ios::out);
		if (fout.is_open())
		{
			fout << procedure + ": " << ms << "\r";
			fout.close();
		}
		else
		{
			cout << "time to txt failed" << endl;
			exit(1);
		}
	}

	void clearTimeTxt();

	template <typename T>
	void calErr(Mat& DP, Mat& DT, string procedure, bool calCSV = false)
	{
		string addr = param_.savePath + param_.err_name;
		ofstream fout;
		fout.open(addr, ios::app | ios::out);
		if (fout.is_open())
		{
			fout << procedure << "\t";
			const int THRES = param_.errorThreshold;
			Mat mask;
			string regionName = "nonocc";
			for (int region = 0; region < 3; region++)
			{
				if (I_mask[region].empty())
					continue;
				if (region == 1)
					regionName = "all";
				else if (region == 2)
					regionName = "disc";
				int sumNum = 0, errorNumer = 0;
				float errorValueSum = 0;
				mask = I_mask[region];
				for (int v = 0; v < h_; v++)
				{
					uchar* maskP = mask.ptr<uchar>(v);
					T* ptrDP = DP.ptr<T>(v);
					float* ptrDT = DT.ptr<float>(v);
					for (int u = 0; u < w_; u++)
					{
						if (maskP[u] == 255)
						{
							sumNum++;
							if (ptrDP[u] >= 0)
							{
								float dif = abs(ptrDT[u] - ptrDP[u]);
								errorValueSum += pow(dif, 2);
								if (dif > THRES)
									errorNumer++;
							}
							else
							{
								errorNumer++;
								errorValueSum += 2;  // 这个地方加多少合适呢？
							}
						}
					}
				}
				float PBM = (float)errorNumer / sumNum;
				float rms = sqrt(errorValueSum / sumNum);
				std::cout << endl << regionName << "\terrorRatio: " << PBM << " epe: " << rms << " " + procedure << endl;
				fout << regionName + "\tPBM: " << PBM << "\tRMS: " << rms << "\t";
				// www
				if (calCSV)
				{
					if (param_.saveErr2CsvOnlyAll)
					{
						if (regionName == "all")
							ofs_ << PBM << ",";
					}
					else if (param_.saveErr2CsvOnlyNonocc)
					{
						if (regionName == "nonocc")
							ofs_ << PBM << ",";
					}
					else
						ofs_ << PBM << ",";
				}
			}
			fout << "\n";
			fout.close();
		}
		else
		{
			cout << "can't write err to txt";
			exit(0);
		}
	}

	void clearErrTxt();


	void saveErrorMap(std::string add, Mat& DP, Mat& DT);

	template <typename T>
	void saveBiary(string method, Mat& DP, Mat& DT)
	{
		string addr = param_.savePath + method + ".png";
		Mat biaryImg(h_, w_, CV_8UC3);
		CV_Assert(DT.type() == CV_32F);
		CV_Assert(DP.size() == DT.size());

		// "tsukuba", "venus", "teddy", "cones"
		if (object == "tsukuba" || object == "venus" || object == "teddy" || object == "cones")
		{// wxy
			Mat occMask = I_mask[0];
			for (int v = 0; v < h_; v++)
			{
				T* dpPtr = DP.ptr<T>(v);
				float* dtPtr = DT.ptr<float>(v);
				for (int u = 0; u < w_; u++)
				{
					uchar* biPtr = biaryImg.ptr<uchar>(v, u);
					if (dtPtr[u] == 0)
					{
						biPtr[0] = 255;
						biPtr[1] = 255;
						biPtr[2] = 255;
					}
					else
					{
						if (abs(dpPtr[u] - dtPtr[u]) > param_.errorThreshold)
						{
							int occV = occMask.ptr<uchar>(v)[u];
							if (occV == 0)
							{
								biPtr[0] = 170;
								biPtr[1] = 170;
								biPtr[2] = 170;
							}
							else
							{
								biPtr[0] = 0;
								biPtr[1] = 0;
								biPtr[2] = 0;
							}

						}
						else
						{
							biPtr[0] = 255;
							biPtr[1] = 255;
							biPtr[2] = 255;
						}
					}
				}
			}

		}
		else
		{
			for (int v = 0; v < h_; v++)
			{
				T* dpPtr = DP.ptr<T>(v);
				float* dtPtr = DT.ptr<float>(v);
				for (int u = 0; u < w_; u++)
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
						if (abs(dpPtr[u] - dtPtr[u]) > param_.errorThreshold)
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
		imwrite(addr, biaryImg);  //save biaryImg
	}

	/*检测topDisp(即代价最小的几个代价)中是否含有正确视差，以彩色图输出，若最小代价视差正确，则此点标为绿色，若最小代价之外的其他视差正确，则此点标记为蓝色，若
	 *
	 *
	*/
	void signCorrectFromTopVm(string imgName, Mat& dispTop, Mat& DT)
	{
		string addr = param_.savePath + imgName;
		Mat biaryImg(h_, w_, CV_8UC3);
		CV_Assert(DT.type() == CV_32F);
		CV_Assert(dispTop.type() == CV_32F);
		CV_Assert(dispTop.dims == 4);
		const int Num = dispTop.size[2];

		for (int v = 0; v < h_; v++)
		{
			float* dtPtr = DT.ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				uchar* biPtr = biaryImg.ptr<uchar>(v, u);
				float dtV = dtPtr[u];
				if (dtV == 0)
				{
					biPtr[0] = 0;
					biPtr[1] = 0;
					biPtr[2] = 0;
				}
				else
				{
					bool hasTrueDisp = false;
					int num_ = dispTop.ptr<float>(v, u, Num - 1)[0];
					for (int n = 0; n < num_; n++)
					{
						float* dpPtr = dispTop.ptr<float>(v, u, n);
						if (abs(dpPtr[0] - dtV) <= param_.errorThreshold)
						{
							if (n == 0)
							{
								biPtr[0] = 0;
								biPtr[1] = 252;
								biPtr[2] = 124;
							}
							else
							{
								biPtr[0] = 238;
								biPtr[1] = 134;
								biPtr[2] = 28;
							}
							hasTrueDisp = true;
							break;
						}
					}
					if (!hasTrueDisp)
					{
						biPtr[0] = 0;
						biPtr[1] = 0;
						biPtr[2] = 255;
					}
				}
			}
		}

		//Mat signImg = biaryImg.clone();
		//for (int v = param_.sign_v_range[0]; v < param_.sign_v_range[1]; v++)
		//{
		//	for (int u = param_.sign_u_range[0]; u < param_.sign_u_range[1]; u += 3)
		//	{
		//		uchar* sP = signImg.ptr<uchar>(v, u);
		//		for (int c = 0; c < 3; c++)
		//			sP[c] = 125;
		//	}
		//}
		//imwrite(param_.savePath + "sign.png", signImg);  //save biaryImg

		imwrite(addr, biaryImg);  //save biaryImg
	}

	void errorMap(cv::Mat& DP, cv::Mat& DT, cv::Mat& errMap);

	void biaryImg(cv::Mat& DP, cv::Mat& DT, cv::Mat& biaryImg);

	template <typename T, int imgNum = 1>
	void saveDispMap(const cv::Mat & dispM, const Mat & trueM, string method, bool calErr = false)
	{
		Size size = dispM.size();
		Mat dispMap(size, CV_8UC3, Scalar::all(0));

		T disp_max = 0;
		T disp_min = numeric_limits<T>::max();
		/*	OMP_PARALLEL_FOR*/
		for (int v = 0; v < size.height; v++)
		{
			for (int u = 0; u < size.width; u++)
			{
				T val = dispM.ptr<T>(v)[u];
				if (val < disp_min && val >= 0)
					disp_min = val;
				if (val > disp_max)
					disp_max = val;
			}
		}

		float distance = disp_max - disp_min;
		float ratio = 255.0 / distance;
		//OMP_PARALLEL_FOR
		for (int v = 0; v < size.height; v++)
		{
			const T* disP = dispM.ptr<T>(v);
			for (int u = 0; u < size.width; u++)
			{
				uchar* disMP = dispMap.ptr<uchar>(v, u);
				if (disP[u] >= 0)
				{
					float disp = ratio * (disP[u] - disp_min);
					uchar d = static_cast<uchar>(disp);
					disMP[0] = d;
					disMP[1] = d;
					disMP[2] = d;
				}
				else
				{
					short dValue = disP[u];
					if (dValue == param_.DISP_OCC)
					{
						disMP[0] = 255;
						disMP[1] = 0;
						disMP[2] = 0;
					}
					else if (dValue == param_.DISP_MIS)
					{
						disMP[0] = 0;
						disMP[1] = 0;
						disMP[2] = 255;
					}
					else if (dValue == param_.DISP_PKR)
					{
						disMP[0] = 0;
						disMP[1] = 255;
						disMP[2] = 255;
					}
					else if (dValue == param_.err_ip_dispV)
					{
						disMP[0] = 255;
						disMP[1] = 0;
						disMP[2] = 255;
					}
					else if (dValue == param_.cor_ip_dispV)
					{
						disMP[0] = 255;
						disMP[1] = 255;
						disMP[2] = 0;
					}
				}
			}
		}

		string path = param_.savePath; // ******
		system(("IF NOT EXIST " + path + " (mkdir " + path + ")").c_str());
		path += method + ".png";
		imwrite(path, dispMap);

		if (calErr)
		{
			for (int v = 0; v < size.height; v++)
			{
				const T* disP = dispM.ptr<T>(v);
				const float* disT = trueM.ptr<float>(v);
				uchar* all = I_mask[1].ptr<uchar>(v);
				for (int u = 0; u < size.width; u++)
				{
					if (all[u] > 0)
					{
						uchar* disMP = dispMap.ptr<uchar>(v, u);
						if (abs(disT[u] - disP[u]) > 1)
						{
							disMP[0] = 0;
							disMP[1] = 0;
							disMP[2] = 255;
						}
					}
				}
			}
			string name = param_.savePath + method + "_err.png";
			imwrite(name, dispMap);

			//	int step2 = 0;
			//	for (int v = 0; v < size.height; v++)
			//	{
			//		const uchar* armMP = arm_Mask.ptr<uchar>(v);
			//		for (int u = 0; u < size.width; u++)
			//		{
			//			uchar* disP = dispMap.ptr<uchar>(v, u);
			//			if (armMP[u] > 0)
			//			{
			//				disP[0] = 255;
			//				disP[1] = 255;
			//				disP[2] = 51;
			//				step2++;
			//			}
			//		}
			//	}
			//	name = param_.savePath + method + "Sign.png";
			//	imwrite(name, dispMap);
			//	cout << "step2: " << step2 << endl;
			//}
		}
	}
	// 分别统计对遮挡点和误匹配点插值了多少，里面正确的是多少
	void coutInterpolaterEffect(Mat& dispBef, Mat& dispAft)
	{
		int ipoled_occ = 0;
		int ipoled_occ_right = 0;
		int ipoled_mis = 0;
		int ipoled_mis_right = 0;
		for (int v = 0; v < h_; v++)
		{
			short* aftP = dispAft.ptr<short>(v);
			short* befP = dispBef.ptr<short>(v);
			float* DTP = DT.ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				float DTV = DTP[u];
				if (DTV > 0)
				{
					short aftV = aftP[u];
					short befV = befP[u];
					if (befV < 0 && aftV >= 0)
					{
						if (befV == param_.DISP_OCC)
						{
							ipoled_occ++;
							if (abs(aftV - DTV) <= param_.errorThreshold)
								ipoled_occ_right++;
						}
						else if (befV == param_.DISP_MIS)
						{
							ipoled_mis++;
							if (abs(aftV - DTV) <= param_.errorThreshold)
								ipoled_mis_right++;
						}
					}
				}

			}
		}

		cout << "ipoled_occ: " << ipoled_occ << endl;
		cout << "ipoled_occ_right: " << ipoled_occ_right << endl;
		cout << "ipoled_mis: " << ipoled_mis << endl;
		cout << "ipoled_mis_right: " << ipoled_mis_right << endl;
		
	}

	void gen_dispFromVm(Mat& vm, Mat& dispMap);

	void transform_NCCVm(Mat& vm);

	void transform_NCCVm2(Mat& vm);

	void iterpolateBackground(cv::Mat& dst);

	void sgm(cv::Mat& vm, bool leftFist = true);

	void sgm_hori(cv::Mat& vm, bool leftFirst);

	void sgm_verti(cv::Mat& vm, bool leftFirst);

	void so(Mat& vm, Mat& disp, vector<Mat>& I);

	void so_change(Mat& vm, Mat& DP, vector<Mat>& I);

	void so_T2D(Mat& vm, Mat& DP, vector<Mat>& I);

	void so_R2L(Mat& vm, Mat& DP, vector<Mat>& I);

	void gen_sgm_vm(Mat& vm, vector<cv::Mat1f>& Lr, int numOfDirec);

	static float min4(float a, float b, float c, float d)
	{
		return std::min(std::min(a, b), std::min(c, d));
	}

	template <typename T>
	void updateCost(cv::Mat& Lr, cv::Mat& vm, int v, int u, int n, int rv, int ru, bool preIsInner, bool leftFirst)
	{
		const int corDirThres = param_.sgm_corDifThres;  //15
		const int reduCoeffi1 = param_.sgm_reduCoeffi1;  //4
		const int reduCoeffi2 = param_.sgm_reduCoeffi2;  //20 10

		int direction = 1;
		if (!leftFirst)
			direction = -1;
		T* vmPtr = vm.ptr<T>(v, u);
		if (preIsInner)
		{
			int D1 = 0;
			for (int c = 0; c < 3; c++)
			{
				if (leftFirst)
					D1 = max(D1, abs(I_c[0].ptr<uchar>(v, u)[c] - I_c[0].ptr<uchar>(v + rv, u + ru)[c]));
				else
					D1 = max(D1, abs(I_c[1].ptr<uchar>(v, u)[c] - I_c[1].ptr<uchar>(v + rv, u + ru)[c]));
			}
				
			float* LrForePtr = Lr.ptr<float>(v + rv, u + ru);
			float minC = numeric_limits<float>::max();
			for (int d = 0; d < n; d++)
				minC = min(LrForePtr[d], minC);

			for (int d = 0; d < n; d++)
			{
				float P1 = 1.0;  //1.0 
				float P2 = 3.0;  //3.0
				int D2 = 0;
				if (u - direction * d < 0 || u + ru - direction * d < 0 
					|| u - direction * d >= w_ || u + ru - direction * d >= w_)
					D2 = corDirThres + 1;
				else
				{
					if (leftFirst)
						for (int c = 0; c < 3; c++)
							D2 = max(D2, abs(I_c[1].ptr<uchar>(v, u - direction * d)[c] - I_c[1].ptr<uchar>(v + rv, u + ru - direction * d)[c]));
					else
						for (int c = 0; c < 3; c++)
							D2 = max(D2, abs(I_c[0].ptr<uchar>(v, u - direction * d)[c] - I_c[0].ptr<uchar>(v + rv, u + ru - direction * d)[c]));
				}
				
				//if (D1 > corDirThres && D2 > corDirThres)
				//{
				//	P1 /= reduCoeffi2;
				//	P2 /= reduCoeffi2;
				//}
				//else if (D1 > corDirThres || D2 > corDirThres)
				//{
				//	P1 /= reduCoeffi1;
				//	P2 /= reduCoeffi1;
				//}
				if (D1 > corDirThres)
				{
					P1 /= reduCoeffi1;
					P2 /= reduCoeffi1;
				}
					
				P1 -= minC;

				T cost = vmPtr[d];
				float S1 = LrForePtr[d] - minC;
				float S2 = d - 1 >= 0 ? LrForePtr[d - 1] + P1 : numeric_limits<float>::max();
				float S3 = d + 1 < n ? LrForePtr[d + 1] + P1 : numeric_limits<float>::max();
				float S4 = P2;

				Lr.ptr<float>(v, u)[d] = cost + min4(S1, S2, S3, S4);
			}
		}
		else
			for (int d = 0; d < n; d++)
				Lr.ptr<float>(v, u)[d] = vmPtr[d];
	}

	void costScan(cv::Mat& Lr, cv::Mat& vm, int rv, int ru, bool leftFirst);

	void showParams();

	void BF(vector<Mat>& vm);

	// refined process
	int cal_histogram_for_HV(Mat& dispImg, int v_ancher, int u_ancher, int numThres, float ratioThre, int LOR);
	int cal_histogram_for_Tile(Mat& dispImg, int v_ancher, int u_ancher, int numThres, float ratioThre);
	int compareArmL(int v, int u);

	void LRConsistencyCheck(cv::Mat& D1, cv::Mat& D2, cv::Mat& errMask, int LOR = 0);
	void LRConsistencyCheck_normal(cv::Mat& D1, cv::Mat& D2, cv::Mat& errMask, int LOR = 0);
	void LRConsistencyCheck_new(Mat& errorMask);
	void RV_combine_BG(cv::Mat& Dp, float rv_float, int rv_s);
	void regionVote_my(cv::Mat& Dp, float rv_ratio, int rv_s);
	void regionVote(cv::Mat& Dp, Mat& cross);
	void backgroundInterpolateCore(Mat& Dp, int v, int u, int* result);
	int backgroundInterpolateCore(Mat& Dp, int v, int u);
	int backgroundInterpolateCore_(Mat& Dp, int v, int u);
	int regionVoteCore(Mat& Dp, int v, int u, int SThres, float hratioThres);
	void regionVoteForWholeDispImg(Mat& Dp);
	int properIpolCore(Mat& Dp, int v, int u);
	void properIpol(cv::Mat& DP, cv::Mat& I1_c);
	void BGIpol(cv::Mat& Dp);
	void WM(Mat& disp, Mat& mask, Mat& img);
	void discontinuityAdjust(cv::Mat& ipol);
	void subpixelEnhancement(cv::Mat& disparity, Mat& floatDisp);

	void cbbi(Mat& Dp);

	int cbbi_core(Mat& Dp, Mat& cutImg, int v, int u);
	void cutImg(Mat& imgCut);



	// util
	void execCanny(Mat& imgEdg)
	{
		GaussianBlur(I_g[0], imgEdg, Size(5, 5), 4, 4);
		equalizeHist(imgEdg, imgEdg);
		blur(imgEdg, imgEdg, Size(3, 3));  // 均值滤波
		copyMakeBorder(imgEdg, imgEdg, 1, 1, 1, 1, BORDER_REPLICATE);
		Canny(imgEdg, imgEdg, 146, 181, 3);
		//limitRange(imgEdg, 60, false);
	}

	void limitRange(Mat& img, int range, bool is_scratch = true)
	{
		if (is_scratch)
			std::memset(img.data, 0, (uint64_t)img.rows * img.cols * sizeof(uchar));
		for (int v = 0; v < img.rows; v++)
		{
			for (int u = range; u < img.cols - range; u++)
			{
				*(img.data + v * img.step[0] + u * img.step[1]) = 1;
			}
		}
	}

	void symmetry_borderCopy_3D(Mat& src, Mat& dst, int top, int down, int left, int right)
	{
		CV_Assert(src.dims == 3 && dst.dims == 3);
		CV_Assert(src.type() == CV_32F && dst.type() == CV_32F);
		const int h = src.size[0];
		const int w = src.size[1];
		const int n = src.size[2];

		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				float* srcP = src.ptr<float>(v, u);
				float* dstP = dst.ptr<float>(v + top, u + left);
				for (int d = 0; d < n; d++)
					dstP[d] = srcP[d];
			}
		}

		// top、down interpolate
		for (int i = 0; i < 2; i++)
		{
			int v = top, coefficient = 1, dis = top;
			if (i == 1)
				v = top + h - 1, coefficient = -1, dis = down;

			for (int distance = 1; distance <= dis; distance++)
			{
				int v_out = v - coefficient * distance;
				int v_inner = v + coefficient * distance;
				for (int u = left; u < left + w; u++)
				{
					float* outP = dst.ptr<float>(v_out, u);
					float* innerP = dst.ptr<float>(v_inner, u);
					for (int d = 0; d < n; d++)
						outP[d] = innerP[d];
				}
			}
		}

		// left、right interpolate(contain coner)
		for (int v = 0; v < h + top + down; v++)
		{
			int u = left, dis = left, coefficient = 1;
			for (int i = 0; i < 2; i++)
			{
				if (i == 1)
					u = left + w - 1, dis = right, coefficient = -1;
				for (int distance = 1; distance <= dis; distance++)
				{
					float* innerP = dst.ptr<float>(v, u + distance * coefficient);
					float* outP = dst.ptr<float>(v, u - distance * coefficient);
					for (int d = 0; d < n; d++)
						outP[d] = innerP[d];
				}
			}
		}
	}

	/*
	** @thres 阈值百分比(高于或低于极值代价多少范围内的考虑进来）
	** @num 在thres范围内选择的代价数量
	*/
	void selectTopCostFromVolumn(Mat& vm, Mat& topDisp, float thres)
	{
		CV_Assert(topDisp.dims == 4);
		CV_Assert(topDisp.type() == CV_32F);
		int n = param_.numDisparities;
		//const float mostBigCost = 2
		//Mat vmCopy = vm.clone();
		float mostBigCost = numeric_limits<float>::max();

		int num = topDisp.size[2] - 1;
		float firstV = 0;
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* vmP = vm.ptr<float>(v, u);
				float* topDispLastP = topDisp.ptr<float>(v, u, num);
				for (int findNum = 0; findNum < num; findNum++)
				{
					float* topDP = topDisp.ptr<float>(v, u, findNum);
					float mostSmall = vmP[0];
					int disp = 0;
					for (int d = 1; d < n; d++)
					{
						float val = vmP[d];
						if (mostSmall > val)
						{
							mostSmall = val;
							disp = d;
						}
					}
	
					if (findNum == 0)
					{
						vmP[disp] = mostBigCost;
						topDispLastP[0] = 1;
						topDP[0] = disp;
						topDP[1] = mostSmall;
						firstV = mostSmall;
					}

					else
					{
						if (mostSmall < firstV * thres)
						{
							vmP[disp] = mostBigCost;
							topDispLastP[0]++;
							topDP[0] = disp;
							topDP[1] = mostSmall;
						}
						else
							break;
					}	 
				}
			}
		}
	}

	// 从topVm找寻指定范围内的正确视差的情况，包括是否含正确视差，正确视差，代价，第几顺位，和最小代价的差，差占最小代价的比重, 存于csv中
	void genExcelFromTopDisp(Mat& topDisp, Mat& DT);

	void genDispFromTopCostVm(Mat& topDisp, Mat& disp) 
	{
		const int num = topDisp.size[2];  // 设置的候选视差的个数加上一位实际候选视差个数指示位
		//int dv[] = {0, -1, -1, -1, 0, 1, 1, 1};
		//int du[] = {-1, -1, 0, 1, 1, 1, 0, -1};
		int dv[] = { 0, 0 };
		int du[] = { -1, 1 };
		const int dirNum = 2;
		//const int dirNum = param_.vmTop_thres_dirNum;

		map<int, int> disp_Num_container;
		map<int, float> disp_Cost_container;
		map<int, int>::iterator iter;
		map<int, float>::iterator iter_;

		for (int v = 0; v < h_; v++)
		{
			short* disP = disp.ptr<short>(v);
			for (int u = 0; u < w_; u++) 
			 { 
				float dispNum = topDisp.ptr<float>(v, u, num - 1)[0];
				if (dispNum == 1)
					disP[u] = topDisp.ptr<float>(v, u, 0)[0];
				else if (dispNum > 1)
				 {
					for (int i = 0; i < dispNum; i++)
					{
						int disp = topDisp.ptr<float>(v, u, i)[0];
						float cost = topDisp.ptr<float>(v, u, i)[1];
						disp_Num_container[disp]++;
						disp_Cost_container[disp] += cost;
					}

					for (int dir = 0; dir < dirNum; dir++)
					{
						int v_ = v + dv[dir], u_ = u + du[dir];
						if (v_ >= 0 && v_ < h_ && u_ >= 0 && u_ < w_)
						{
							int dispNum_ = topDisp.ptr<float>(v_, u_, num - 1)[0];
							for (int n = 0; n < dispNum_; n++)
							{
								int disp = topDisp.ptr<float>(v_, u_, n)[0];
								float cost = topDisp.ptr<float>(v_, u_, n)[1];
								iter = disp_Num_container.find(disp);
								if (iter != disp_Num_container.end())
								{
									disp_Num_container[disp]++;
									disp_Cost_container[disp] += cost;
								}
							}
						}
					}

					int disp = -1, dispNum = -1;
					float cost = numeric_limits<float>::max();
					for (iter = disp_Num_container.begin(); iter != disp_Num_container.end(); iter++)
					{
						int dNum = iter->second;
						int disp_ = iter->first;
						iter_ = disp_Cost_container.find(disp_);
						if (iter_ == disp_Cost_container.end())
						{
							cout << "mapCost and mapNum inconsistent";
							abort();
						}
						float cost_ = iter_->second;
						if (dNum > dispNum || (dNum = dispNum && cost_ < cost))
						{
							dispNum = dNum;
							cost = cost_;
							disp = disp_;
						}
					}
					disP[u] = disp;
					disp_Num_container.clear();
					disp_Cost_container.clear();
				}
			}
		}
	}

	void genDispFromTopCostVm2(Mat& topDisp, Mat& disp);

	// 块滤波（且匹配代价卷已经经过边缘扩充操作）
	void BF_BI(Mat& src, Mat& dst, int W_V, int W_U)
	{
		CV_Assert(src.dims == 3 && dst.dims == 3);
		CV_Assert(src.type() == CV_32F && dst.type() == CV_32F);
		const int h = dst.size[0];
		const int w = dst.size[1];
		const int n = dst.size[2];
		const int S = (W_V * 2 + 1) * (W_U * 2 + 1);

		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				float* dstP = dst.ptr<float>(v, u);
				for (int d = 0; d < n; d++)
				{
					float sum = 0;
					for (int dv = -W_V; dv <= W_V; dv++)
						for (int du = -W_U; du <= W_U; du++)
							sum += src.ptr<float>(v + W_V + dv, u + W_U + du)[d];
					dstP[d] = sum / S;
				}
			}
		}
	}

	// fuse two vm with weight
	void fuse_2vm(Mat& vm1, Mat& vm2, float weight1, float weight2)
	{
		CV_Assert(vm1.type() == CV_32F && vm2.type() == CV_32F);
		const int h = vm1.size[0];
		const int w = vm1.size[1];
		const int n = vm1.size[2];

		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				float* vm1P = vm1.ptr<float>(v, u);
				float* vm2P = vm2.ptr<float>(v, u);
				for (int d = 0; d < n; d++)
					vm1P[d] = vm1P[d] * weight1 + vm2P[d] * weight2;
			}
		}

	}
	// select the small value from every pos in vm1 and vm2 to fill the dst, support in-place 
	void cross2Vm(Mat& vm1, Mat& vm2, Mat& dst)
	{
		CV_Assert(vm1.type() == CV_32F && vm2.type() == CV_32F && dst.type() == CV_32F);
		const int h = vm1.size[0];
		const int w = vm1.size[1];
		const int n = vm1.size[2];
		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				float* v1P = vm1.ptr<float>(v, u);
				float* v2P = vm2.ptr<float>(v, u);
				float* dstP = dst.ptr<float>(v, u);
				for (int d = 0; d < n; d++)
					dstP[d] = min(v1P[d], v2P[d]);
			}
		}
	}

	/*标记前后两幅视差图中错误点的改变，将正确的变成错误的标记一种颜色（），错误的变成正确的标记一种颜色
	 *要用三通道的图来保存，其他没变化的，如果是3通道，则直接复制，若是1通道，则将一个通道的值复制给另外
	 *两个通道即可
	 */
	void signDispChange_forRV(Mat& dispOriginal, Mat& dispChanged, Mat& trueD, Mat& mask, Mat& result)
	{
		for (int v = 0; v < h_; v++)
		{
			short* dispOP = dispOriginal.ptr<short>(v);
			short* dispCP = dispChanged.ptr<short>(v);
			float* trueDP = trueD.ptr<float>(v);
			short* resP = result.ptr<short>(v);
			uchar* maskP = mask.ptr<uchar>(v);
			for (int u = 0; u < w_; u++)
			{
				short dispO = dispOP[u];
				if (maskP[u] == 255)
				{
					if (dispO >= 0)
						resP[u] = dispO;
					else
					{
						short dispC = dispCP[u];
						if (dispC < 0)
							resP[u] = dispC;
						else
							if (abs(trueDP[u] - dispC) > param_.errorThreshold)
								resP[u] = param_.err_ip_dispV ;
							else
								resP[u] = param_.cor_ip_dispV;
					}
				}
				else
					resP[u] = dispCP[u];
			}
		}
	}

	
	void signDispChange_for2Disp(Mat& dispOriginal, Mat& dispChanged, Mat& trueD, Mat& mask, Mat& result)
	{
		CV_Assert(dispOriginal.type() == CV_16S && dispChanged.type() == CV_16S);
		CV_Assert(trueD.type() == CV_32F && mask.type() == CV_8U);
		CV_Assert(result.type() == CV_16S);

		for (int v = 0; v < h_; v++)
		{
			float* trueP = trueD.ptr<float>(v);
			uchar* maskP = mask.ptr<uchar>(v);
			short* dispOP = dispOriginal.ptr<short>(v);
			short* dispCP = dispChanged.ptr<short>(v);
			short* resultP = result.ptr<short>(v);
			for (int u = 0; u < w_; u++)
			{
				short dispOV = dispOP[u];
				short dispCV = dispCP[u];
				float trueV = trueP[u];
				if (maskP[u] == 255 && dispOV != dispCV)
				{
					bool is_ORight = abs(trueP[u] - dispOV) <= param_.errorThreshold;
					bool is_CRight = abs(trueP[u] - dispCV) <= param_.errorThreshold;
					if (is_ORight && is_CRight)
						resultP[u] = dispCV;
					else if (is_ORight && !is_CRight)
						resultP[u] = param_.err_ip_dispV;
					else if (!is_ORight && is_CRight)
						resultP[u] = param_.cor_ip_dispV;
					else
						resultP[u] = dispCV;
				}
				else
					resultP[u] = dispCV;
			}
		}
	}

	void edgeEnhance(cv::Mat& srcImg, cv::Mat& dstImg);

	void saveFromVm(vector<Mat>& vm, string name);

	template <typename T, int LOR = 0>
	void saveFromDisp(Mat& disp, string name, bool calCsv = false, bool dispShowErr = false);

public:

	Parameters param_;
	vector<Mat> I_c;
	Mat Lab;
	Mat HSV;
	vector<Mat> I_g;
	vector<Mat> I_mask;  // 0为nonocc，1为all, 2为disc
	int h_;
	int w_;
	int d_;
	std::vector<cv::Mat1f> L;  // 因为float类型的范围暂时可以容下所有的单方向聚合值
	vector<cv::Mat1f> S;  // 设置为float类型的原因同上
	vector<Mat> census_;
	int size_vm[3];

	int HVL_num;
	int TileL_num;
	vector<Mat> HVL;
	vector<Mat> HVL_INTERSECTION;
	vector<Mat> tileCrossL;
	vector<Mat> tile_INTERSECTION;
	vector<Mat> vm;
	vector<Mat> asd_vm;
	vector<Mat> tile_neighbor;
	cv::Mat DP[2];
	cv::Mat DT;
	cv::Mat LRC_Err_Mask;
	cv::Mat PKR_Err_Mask;

	cv::Mat arm_Mask;
	cv::Mat arm_Lst;
	cv::Mat guideDisp;

	ofstream ofs_;

	vector<vector<Mat>> guideVm;

	int img_counting;
};

void SolveAll(StereoMatching**& smPyr, const int PY_LVL, const float REG_LAMBDA);

template <typename T>
static bool judgeColorDif(T* target, T* refer, int thres, int channel);