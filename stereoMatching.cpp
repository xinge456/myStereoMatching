
#include "stereoMatching.h"

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

using namespace std;

void StereoMatching::censusGrad(vector<Mat>& vm)
{
	vector<Mat> gradVm(2);
	vector<Mat> censusVm(2);
	for (int i = 0; i < 2; i++)
	{
		gradVm[i].create(3, size_vm, CV_32F);
		censusVm[i].create(3, size_vm, CV_32F);
	}
	grad(gradVm, 500);
	censusCal(censusVm, 1);

	int img_num = Do_LRConsis ? 2 : 1;
	float lam1 = param_.lamCen;
	float lam2 = param_.lamG;
	for (int i = 0; i < img_num; i++)
	{
		gen_vm_from2vm_exp(vm[i], censusVm[i], gradVm[i], lam1, lam2, i);
		//gen_vm_from2vm_add(vm[i], censusVm[i], gradVm[i], i);
	}
#ifdef DEBUG
	saveFromVm(vm, "censusGrad");
#endif // DEBUG
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
		gen_ad_sd_vm(adVm[i], i, 0, 7);
		//gen_ad_vm(adVm[i], I_c[0], I_c[1], 7, i);
	grad(gradVm, 2);
	for (int i = 0; i < 2; i++)
	{
		//addWeighted(adVm[i], 0.11, gradVm[i], 1 - 0.11, 0, vm[i]);
		gen_vm_from2vm_fixWgt(adVm[i], 0.11, gradVm[i], 1 - 0.11, vm[i]);
		//gen_vm_from2vm_add(vm[i], adVm[i], gradVm[i], 0.11, 1 - 0.11, i);
	}	
	saveFromVm(vm, "adGrad");
}

void StereoMatching::asdCal(vector<Mat>& vm_asd, string method, int imgNum, float Trunc)
{
	cout << "\n" << endl;
	cout << "start " << method << " calculation" << endl;
	clock_t start = clock();
	int type = method == "AD" ? 0 : 1;
	for (int i = 0; i < imgNum; i++)
		gen_ad_sd_vm(vm_asd[i], i, type, Trunc);
	clock_t end = clock();
	clock_t time = end - start;
	cout << time << endl;
#ifdef DEBUG
	saveTime(time, method);
	saveFromVm(vm_asd, method);
#endif
	cout << "AD vm generated" << endl;
}

void StereoMatching::bt(vector<Mat>& vm)
{
	cout << "\n" << endl;
	cout << "start " << "BT" << " calculation" << endl;
	clock_t start = clock();

	vector<Mat> grayNei(2);
	for (int i = 0; i < 2; i++)
		grayNei[i].create(h_, w_, CV_32FC2);
	calNeiMaxMin(grayNei);
	int n = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < n; i++)
	{
		calCostForBT(vm[i], grayNei, I_g, i);
	}

	clock_t end = clock();
	clock_t time = end - start;
	cout << time << endl;
#ifdef DEBUG
	saveTime(time, "bt");
	saveFromVm(vm, "bt");
#endif
	cout << "bt vm generated" << endl;
}

void StereoMatching::bt_color(vector<Mat>& vm)
{
	cout << "\n" << endl;
	cout << "start " << "BT" << " calculation" << endl;
	clock_t start = clock();

	vector<Mat> colorNei(2);
	for (int i = 0; i < 2; i++)
		colorNei[i].create(h_, w_, CV_32FC(6));
	calNeiMaxMin_color(colorNei);
	int n = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < n; i++)
	{
		calCostForBT(vm[i], colorNei, I_c, i);
	}

	clock_t end = clock();
	clock_t time = end - start;
	cout << time << endl;
#ifdef DEBUG
	saveTime(time, "bt");
	saveFromVm(vm, "bt");
#endif
	cout << "bt vm generated" << endl;
}

void StereoMatching::calCostForBT(Mat& vm, vector<Mat>& grayNei, vector<Mat>& I_g, int LOR)
{
	const int n = param_.numDisparities;
	const int cha = I_g[0].channels();
	float trunc = 20;

	int cofL = LOR == 0 ? 0 : 1;
	int cofR = LOR == 0 ? -1 : 0;
	for (int v = 0; v < h_; v++)
	{
		float* vP = vm.ptr<float>(v);
		float* nLP = grayNei[0].ptr<float>(v);
		float* nRP = grayNei[1].ptr<float>(v);
		uchar* iLP = I_g[0].ptr<uchar>(v);
		uchar* iRP = I_g[1].ptr<uchar>(v);

		for (int u = 0; u < w_; u++)
		{
			for (int d = 0; d < n; d++)
			{
				int lP = u + d * cofL;
				int rP = u + d * cofR;
				if (lP < w_ && rP >= 0)
				{
					float sum = 0;
					lP *= cha;
					rP *= cha;
					for (int c = 0; c < cha; c++)
					{
						uchar iLV = iLP[lP + c];
						uchar iRV = iRP[rP + c];
						int lsft = d * cofL * 2 * cha + c * 2;
						int rsft = d * cofR * 2 * cha + c * 2;
						float v0 = max(0.0f, max(nRP[rsft] - iLV, iLV - nRP[rsft + 1]));
						float v1 = max(0.0f, max(nLP[lsft] - iRV, iRV - nLP[lsft + 1]));
						sum += min(v0, v1);
					}
					*vP++ = min(sum / cha, trunc);
					//uchar iLV = iLP[lP];
					//uchar iRV = iRP[rP];
					//int lsft = d * cofL * 2;
					//int rsft = d * cofR * 2;
					//float v0 = max(0.0f, max(nRP[rsft] - iLV, iLV - nRP[rsft + 1]));
					//float v1 = max(0.0f, max(nLP[lsft] - iRV, iRV - nLP[lsft + 1]));
					//*vP++ = min(v0, v1);
				}
				else
					*vP++ = trunc;
			}
			nLP += 2 * cha;
			nRP += 2 * cha;
		}
	}
}

void StereoMatching::calNeiMaxMin(vector<Mat>& grayNei)
{
	for (int i = 0; i < 2; i++)
	{
		Mat I = I_g[i];
		Mat grayN = grayNei[i];
		for (int v = 0; v < h_; v++)
		{
			uchar* Ip = I.ptr<uchar>(v);
			float* nP = grayN.ptr<float>(v);
			float Il = Ip[0];
			float Ir = 0.5 * (Ip[1] + Ip[0]);
			nP[0] = min(Il, Ir);
			nP[1] = max(Il, Ir);
			nP += 2;
			for (int u = 1; u < w_ - 1; u++)
			{
				float Iv = Ip[u];
				Il = 0.5 * (Ip[u - 1] + Iv);
				Ir = 0.5 * (Ip[u + 1] + Iv);
				nP[0] = min(Iv, min(Il, Ir));
				nP[1] = max(Iv, max(Il, Ir));
				nP += 2;
			}
			Il = 0.5 * (Ip[w_ - 2] + Ip[w_ - 1]);
			Ir = Ip[w_ - 1];
			nP[0] = min(Il, Ir);
			nP[1] = max(Il, Ir);
		}
	}
}

void StereoMatching::calNeiMaxMin_color(vector<Mat>& colorNei)
{
	for (int i = 0; i < 2; i++)
	{
		Mat I = I_c[i];
		Mat colorN = colorNei[i];
		for (int v = 0; v < h_; v++)
		{
			uchar* Ip = I.ptr<uchar>(v);
			float* nP = colorN.ptr<float>(v);
			for (int c = 0; c < 3; c++)
			{
				float Il = Ip[c];
				float Ir = 0.5 * (Ip[c] + Ip[c + 3]);
				*nP++ = min(Il, Ir);
				*nP++ = max(Il, Ir);
			}
			for (int u = 1; u < w_ - 1; u++)
			{
				int pos = u * 3;
				for (int c = 0; c < 3; c++)
				{
					float Iv = Ip[pos + c];
					float Il = 0.5 * (Ip[pos - 3 + c] + Iv);
					float Ir = 0.5 * (Ip[pos + 3 + c] + Iv);
					*nP++ = min(Il, Ir);
					*nP++ = max(Il, Ir);
				}
			}
			for (int c = 0; c < 3; c++)
			{
				int pos = (w_ - 2) * 3;
				float Il = 0.5 * (Ip[pos + c] + Ip[pos + 3 + c]);
				float Ir = Ip[pos + 3 + c];
				*nP++ = min(Il, Ir);
				*nP++ = max(Il, Ir);
			}
		}
	}
}

// 计算x向梯度:(I(u+1) - I(u-1))/2 
void StereoMatching::calGrad(Mat& grad, Mat& img)
{
	int channels = img.channels();
	if (channels == 1)
	{
		for (int v = 0; v < h_; v++)
		{
			uchar* iP = img.ptr<uchar>(v);
			float* gP = grad.ptr<float>(v);
			for (int u = 1; u < w_ - 1; u++)
			{
				gP[u] = 0.5 * (iP[u + 1] - iP[u - 1]);
			}
			gP[0] = iP[1] - iP[0];
			gP[w_ - 1] = iP[w_ - 1] - iP[w_ - 2];
		}
	}
	else
	{
		uchar* iP_L = NULL;
		uchar* iP_R = NULL;
		float* gP = NULL;
		for (int v = 0; v < h_; v++)
		{
			for (int u = 1; u < w_ - 1; u++)
			{
				iP_L = img.ptr<uchar>(v, u - 1);
				iP_R = img.ptr<uchar>(v, u + 1);
				gP = grad.ptr<float>(v, u);
				for (int c = 0; c < 3; ++c)
				{
					gP[c] = 0.5 * (iP_R[c] - iP_L[c]);
				}
			}
			iP_L = img.ptr<uchar>(v, 0);
			iP_R = img.ptr<uchar>(v, 1);
			gP = grad.ptr<float>(v, 0);
			for (int c = 0; c < 3; c++)
				gP[c] = iP_R[c] - iP_L[c];
			iP_L = img.ptr<uchar>(v, w_ - 2);
			iP_R = img.ptr<uchar>(v, w_ - 1);
			gP = grad.ptr<float>(v, w_ - 1);
			for (int c = 0; c < 3; c++)
				gP[c] = iP_R[c] - iP_L[c];
		}
	}
}


void StereoMatching::calGrad_y(Mat& grad, Mat& img)
{
	int channels = img.channels();
	if (channels == 1)
	{
		float* gP = NULL;
		uchar* iP_pre = NULL;
		uchar* iP_aft = NULL;
		for (int v = 1; v < h_ - 1; v++)
		{
			iP_pre = img.ptr<uchar>(v - 1);
			iP_aft = img.ptr<uchar>(v + 1);
			gP = grad.ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				gP[u] = 0.5 * (iP_aft[u] - iP_pre[u]);
				//gP[u] = abs(iP[u + 1] - iP[u]) + abs(img.ptr<uchar>(v + 1)[u] - iP[u]);
				//gP[u] = sqrt(pow(iP[u + 1] - iP[u], 2) + pow(img.ptr<uchar>(v + 1)[u] - iP[u], 2));
			}
		}
		gP = grad.ptr<float>(0);
		iP_pre = img.ptr<uchar>(0);
		iP_aft = img.ptr<uchar>(1);
		for (int u = 0; u < w_; u++)
			gP[u] = iP_aft[u] - iP_pre[u];
		gP = grad.ptr<float>(h_ - 1);
		iP_pre = img.ptr<uchar>(h_ - 2);
		iP_aft = img.ptr<uchar>(h_ - 1);
		for (int u = 0; u < w_; u++)
			gP[u] = iP_aft[u] - iP_pre[u];
	}
	else
	{
		float* gP = NULL;
		uchar* iP_pre = NULL;
		uchar* iP_aft = NULL;
		for (int v = 1; v < h_ - 1; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				iP_pre = img.ptr<uchar>(v - 1, u);
				iP_aft = img.ptr<uchar>(v + 1, u);
				gP = grad.ptr<float>(v, u);
				for (int c = 0; c < 3; c++)
					gP[c] = 0.5 * (iP_aft[c] - iP_pre[c]);
				//gP[u] = abs(iP[u + 1] - iP[u]) + abs(img.ptr<uchar>(v + 1)[u] - iP[u]);
				//gP[u] = sqrt(pow(iP[u + 1] - iP[u], 2) + pow(img.ptr<uchar>(v + 1)[u] - iP[u], 2));
			}
		}
		for (int u = 0; u < w_; u++)
		{
			gP = grad.ptr<float>(0, u);
			iP_pre = img.ptr<uchar>(0, u);
			iP_aft = img.ptr<uchar>(1, u);
			for (int c = 0; c < 3; c++)
				gP[c] = iP_aft[c] - iP_pre[c];
		}
		for (int u = 0; u < w_; u++)
		{
			gP = grad.ptr<float>(h_ - 1, u);
			iP_pre = img.ptr<uchar>(h_ - 2, u);
			iP_aft = img.ptr<uchar>(h_ - 1, u);
			for (int c = 0; c < 3; c++)
				gP[c] = iP_aft[c] - iP_pre[c];
		}
	}
}

void StereoMatching::calgradvm(Mat& vm, vector<Mat>& grad, vector<Mat>& grad_y, int num , float Trunc)
{
	CV_Assert(grad[0].depth() == CV_32F);
	const int n = param_.numDisparities;
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
			short* arm = HVL[num].ptr<short>(v, u);
			short* armTile = tileCrossL[num].ptr<short>(v, u);
			float shortestH = 10000;
			float shortestV = 10000;
			for (int dir = 0; dir < 2; dir++)
			{
				if (arm[dir] < shortestH)
					shortestH = arm[dir];
				//if (armTile[dir] < shortest)
				//	shortest = armTile[dir];
			}
			for (int dir = 2; dir < 4; dir++)
			{
				if (arm[dir] < shortestV)
					shortestV = arm[dir];
			}
			
			if (shortestH == 0)
				shortestH++;
			if (shortestV == 0)
				shortestV++;
			float a = shortestH / (shortestH + shortestV);
			// aaaaaa
			//Trunc = 4 + 32 * shortest / (param_.cbca_crossL[0] + 1);
			//Trunc = 4 + 8 * shortest * shortest / (shortest * shortest + 25);
			//Trunc = 2 + 10 * shortest / (shortest  + 5);
			//Trunc = 4 + 8 * exp(-7 / (shortest + 1));
			//Trunc = 5; //测试Trunc为6时的结果？？？？？？
			//Trunc = 4 + 16 * exp(-);

			float* vP = vm.ptr<float>(v, u);

			for (int d = 0; d < n; d++)
			{
				int u0 = u + leftCoe * d;
				int u1 = u + rightCoe * d;
				if (u0 >= w_ || u1 < 0)
					vP[d] = sqrt(pow(Trunc, 2) * 2);
				else
				{
					//vP[d] = sqrt(pow(a * min(abs(grad0P[u0] - grad1P[u1]), Trunc), 2) + pow((1 - a) * min(abs(grad_y0P[u0] - grad_y1P[u1]), Trunc), 2));
					//vP[d] = sqrt(pow(min(abs(grad0P[u0] - grad1P[u1]), Trunc), 2) + pow(min(abs(grad_y0P[u0] - grad_y1P[u1]), Trunc), 2));
					//vp[d] = 0.5 * min(abs(grad0p[u0] - grad1p[u1]), trunc) + 0.5 * min(abs(grad_y0p[u0] - grad_y1p[u1]), trunc);
					if (param_.gradFuse_adpWgt)
						vP[d] = a * min(abs(grad0P[u0] - grad1P[u1]), Trunc) + (1 - a) * min(abs(grad_y0P[u0] - grad_y1P[u1]), Trunc);
					else
						//vP[d] = 0.5 * min(abs(grad0P[u0] - grad1P[u1]), Trunc) + 0.5 * min(abs(grad_y0P[u0] - grad_y1P[u1]), Trunc);
						vP[d] = min(abs(grad0P[u0] - grad1P[u1]), Trunc) + min(abs(grad_y0P[u0] - grad_y1P[u1]), Trunc);
				}
			}
		}
	}
}

void StereoMatching::calgradvm_mag_and_phase(Mat& vm, vector<Mat>& grad_xy, vector<Mat>& grad_atan, int num, float Trunc)
{
	const int n = param_.numDisparities;
	int leftCoe = 0, rightCoe = -1;
	if (num == 1)
		leftCoe = 1, rightCoe = 0;
	const float PI = 3.1415926;
	const float PI2 = 3.1415926 * 2;
	for (int v = 0; v < h_; v++)
	{
		
		for (int u = 0; u < w_; u++)
		{
			float* vP = vm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				int uL = u + leftCoe * d;
				int uR = u + rightCoe * d;
				if (uL >= w_ || uR < 0)
					vP[d] = sqrt(pow(Trunc, 2) * 2);
				else
				{
					float* grad_xy_LP = grad_xy[0].ptr<float>(v, uL);
					float* grad_xy_RP = grad_xy[1].ptr<float>(v, uR);
					float* grad_atan_LP = grad_atan[0].ptr<float>(v, uL);
					float* grad_atan_RP = grad_atan[1].ptr<float>(v, uR);
					float amp = 0;// 幅值
					float phase = 0; //相位
					float sum = 0;
					for (int c = 0; c < 3; c++)
					{
						amp = abs(grad_xy_LP[c] - grad_xy_LP[c]);
						phase = abs(grad_atan_LP[c] - grad_atan_RP[c]);
						phase = phase < PI ? phase : PI2 - phase;
						sum += 0.12 * amp + phase;
					}
					vP[d] = sum / (sum + 4);
					//vP[d] = min(cache, 2.4f);
				}
			}
		}
	}
}

void StereoMatching::calgradvm_1d(Mat& vm, vector<Mat>& grad, int num, float trunc)
{
	float Default = param_.is_gradNorm ? 1 : trunc;
	CV_Assert(grad[0].depth() == CV_32F);
	const int n = param_.numDisparities;
	//const float costD = 500;
	int leftCoe = 0, rightCoe = -1;
	if (num == 1)
		leftCoe = 1, rightCoe = 0;
	for (int v = 0; v < h_; v++)
	{
		float* grad0P = grad[0].ptr<float>(v);
		float* grad1P = grad[1].ptr<float>(v);
		float* vP = vm.ptr<float>(v);
		for (int u = 0; u < w_; u++)
		{
			for (int d = 0; d < n; d++)
			{
				int u0 = u + leftCoe * d;
				int u1 = u + rightCoe * d;
				if (u0 >= w_ || u1 < 0)
					*vP++ = Default;
				else
				{
					float cache = min(abs(grad0P[u0] - grad1P[u1]), trunc);
					if (param_.is_gradNorm)
						cache /= trunc;
					*vP++ = cache;
				}
			}

		}
	}
}

void StereoMatching::saveFromVm(vector<Mat>& vm, string name)
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

void StereoMatching::edgeEnhance(cv::Mat& srcImg, cv::Mat& dstImg)
{
	if (!dstImg.empty())
	{
		dstImg.release();
	}

	std::vector<cv::Mat> rgb;

	if (srcImg.channels() == 3)        // rgb image  
	{
		cv::split(srcImg, rgb);
	}
	else if (srcImg.channels() == 1)   // gray image  
	{
		rgb.push_back(srcImg);
	}

	// 分别对R、G、B三个通道进行边缘增强  
	for (size_t i = 0; i < rgb.size(); i++)
	{
		cv::Mat sharpMat8U;
		cv::Mat sharpMat;
		cv::Mat blurMat;

		// 高斯平滑  
		cv::GaussianBlur(rgb[i], blurMat, cv::Size(7, 7), 0, 0);

		// 计算拉普拉斯  
		cv::Laplacian(blurMat, sharpMat, CV_16S);

		// 转换类型  
		sharpMat.convertTo(sharpMat8U, CV_8U);
		//imshow("shape.png", sharpMat8U);
		//waitKey(0);
		cv::add(rgb[i], sharpMat8U, rgb[i]);
	}
	cv::merge(rgb, dstImg);
}

template <typename T, int LOR>
void StereoMatching::saveFromDisp(Mat& disp, string name, bool calCsv, bool dispShowErr)
{
	string img_num = to_string(img_counting);
	if (LOR == 0)
	{
		saveDispMap<T>(disp, DT, "d" + img_num + "isp_" + name + "LR", dispShowErr);
		saveBiary<short>("b" + img_num + "iary_" + name, disp, DT);
		calErr<short>(disp, DT, name, calCsv);
		img_counting++;
	}
	else
		saveDispMap<T>(disp, DT, "d" + to_string(img_counting - 1) + "isp_" + name + "RL");
}

void StereoMatching::grad(vector<Mat>& vm_grad, float Trunc)
{
	// 计算梯度
	clock_t start = clock();
	//const float Trunc = 4;
	vector<Mat> grad(2);
	vector<Mat> grad_y(2);
	vector<Mat> grad_vm(2);
	vector<Mat> grad_y_vm(2);
	for (int i = 0; i < 2; i++)
	{
		grad[i].create(h_, w_, CV_32F);
		grad_y[i].create(h_, w_, CV_32F);

		grad_vm[i].create(3, size_vm, CV_32F);
		grad_y_vm[i].create(3, size_vm, CV_32F);
		//Sobel(I_g[i], grad[i], CV_32F, 1, 0);
		//Sobel(I_g[i], grad_y[i], CV_32F, 0, 1);
		calGrad(grad[i], I_g[i]);
		calGrad_y(grad_y[i], I_g[i]);
	
		saveDispMap<float>(grad[i], DT, to_string(i) + "grad");
		saveDispMap<float>(grad_y[i], DT, to_string(i) + "grad_y");

	}

	// 计算CBCA的臂长，用于下面控制截断值
	if (!param_.has_initArm)
		initArm();
	if (!param_.has_calArms)
		calArms<uchar>(I_c, HVL, HVL_INTERSECTION, param_.cbca_crossL[0], param_.cbca_crossL_out[0], param_.cbca_cTresh[0], param_.cbca_cTresh_out[0]);

	// 计算代价卷
	int imgNum = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < imgNum; i++)
	{
		//calgradvm_mag_and_phase(vm_grad[i], grad_xy, grad_atan, i, Trunc);
		if (param_.grad_use2direc)
			calgradvm(vm_grad[i], grad, grad_y, i, Trunc);
		else
			calgradvm_1d(vm_grad[i], grad, i, Trunc);

		//calgradvm_1d(vm_grad[i], grad_xy, i, Trunc);
		//calgradvm_1d(vm_grad[i], grad, i, Trunc);
		//calgradvm_1d(vm_grad[i], grad_y, i, Trunc);
		//addWeighted(grad_vm[i], 0.5, grad_y_vm[i], 0.5, 0, vm_grad[i]);
	}
	clock_t end = clock();
	clock_t time = end - start;
#ifdef DEBUG
	saveTime(time, "grad");
	saveFromVm(vm_grad, "grad");
#endif // DEBUG
}

void StereoMatching::grad_color(vector<Mat>& vm_grad, float Trunc)
{
	// 计算梯度
	clock_t start = clock();
	//const float Trunc = 4;
	vector<Mat> grad(2);
	vector<Mat> grad_y(2);
	vector<Mat> grad_xy(2);
	vector<Mat> grad_atan(2);
	vector<Mat> grad_vm(2);
	vector<Mat> grad_y_vm(2);
	for (int i = 0; i < 2; i++)
	{
		grad[i].create(h_, w_, CV_32FC3);
		grad_y[i].create(h_, w_, CV_32FC3);
		grad_xy[i].create(h_, w_, CV_32FC3);
		grad_atan[i].create(h_, w_, CV_32FC3);
		grad_vm[i].create(3, size_vm, CV_32FC3);
		grad_y_vm[i].create(3, size_vm, CV_32FC3);
		//Sobel(I_g[i], grad[i], CV_32F, 1, 0);
		//Sobel(I_g[i], grad_y[i], CV_32F, 0, 1);
		calGrad(grad[i], I_c[i]);
		calGrad_y(grad_y[i], I_c[i]);
		combineXYGrad(grad[i], grad_y[i], grad_xy[i]);
		getAtanGrad(grad[i], grad_y[i], grad_atan[i]);
		/*saveDispMap<float>(grad[i], DT, to_string(i) + "grad");
		saveDispMap<float>(grad_y[i], DT, to_string(i) + "grad_y");
		saveDispMap<float>(grad_atan[i], DT, to_string(i) + "grad_atan");
		saveDispMap<float>(grad_xy[i], DT, to_string(i) + "grad_xy");*/
	}

	// 计算CBCA的臂长，用于下面控制截断值
	//if (!param_.has_initArm)
	//	initArm();
	//if (!param_.has_calArms)
	//	calArms<uchar>(I_c, HVL, HVL_INTERSECTION, param_.cbca_crossL[0], param_.cbca_crossL_out[0], param_.cbca_cTresh[0], param_.cbca_cTresh_out[0]);

	// 计算代价卷
	int imgNum = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < imgNum; i++)
	{
		calgradvm_mag_and_phase(vm_grad[i], grad_xy, grad_atan, i, Trunc);
		//if (param_.grad_use2direc)
		//	calgradvm(vm_grad[i], grad, grad_y, i, Trunc);
		//else
		//	calgradvm_1d(vm_grad[i], grad, i, Trunc);

		//calgradvm_1d(vm_grad[i], grad_xy, i, Trunc);
		//calgradvm_1d(vm_grad[i], grad, i, Trunc);
		//calgradvm_1d(vm_grad[i], grad_y, i, Trunc);
		//addWeighted(grad_vm[i], 0.5, grad_y_vm[i], 0.5, 0, vm_grad[i]);
	}
	clock_t end = clock();
	clock_t time = end - start;
	saveTime(time, "grad_color");
#ifdef DEBUG
	saveTime(time, "grad");
	saveFromVm(vm_grad, "grad");
#endif // DEBUG
}

void StereoMatching::combineXYGrad(Mat& grad_x, Mat& grad_y, Mat& grad_xy)
{
	if (grad_x.channels() == 1)
	{
		for (int v = 0; v < h_; v++)
		{
			float* xP = grad_x.ptr<float>(v);
			float* yP = grad_y.ptr<float>(v);
			float* xyP = grad_xy.ptr<float>(v);
			for (int u = 0; u < w_; u++)
				xyP[u] = sqrt(pow(xP[u], 2) + pow(yP[u], 2));
		}
	}
	else
	{
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* xP = grad_x.ptr<float>(v, u);
				float* yP = grad_y.ptr<float>(v, u);
				float* xyP = grad_xy.ptr<float>(v, u);
				for (int c = 0; c < 3; c++)
					xyP[c] = sqrt(pow(xP[c], 2) + pow(yP[c], 2));
			}
			
		}
	}

}

void StereoMatching::getAtanGrad(Mat& grad_x, Mat& grad_y, Mat& grad_atan)
{
	if (grad_x.channels() == 1)
	{
		for (int v = 0; v < h_; v++)
		{
			float* xP = grad_x.ptr<float>(v);
			float* yP = grad_y.ptr<float>(v);
			float* atanP = grad_atan.ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				float atanV = atan(yP[u] / xP[u]);
				atanP[u] = atanV;
			}
		}
	}
	else
	{
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* xP = grad_x.ptr<float>(v, u);
				float* yP = grad_y.ptr<float>(v, u);
				float* atanP = grad_atan.ptr<float>(v, u);
				for (int c = 0; c < 3; c++)
				{
					float atanV = atan(yP[c] / xP[c]);
					atanP[c] = atanV;
				}
			}
			
		}

	}

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

void StereoMatching::censusCal(vector<Mat>& vm_census, float truncRatio)
{
	cout << "start censusCal" << endl;
	clock_t start = clock();
	const int win_N = 1;

	vector<vector<int>> census_W(win_N);
	
	census_W[0] = {3, 4};
	//census_W[0] = {2, 2};
	//census_W[0] = {2, 2};
	//census_W[1] = { 5, 5 };

	int imgNum = Do_LRConsis ? 2 : 1;
	Mat dispMap(h_, w_, CV_16S);
	int channels = param_.census_channel;  // 1
	int codeLength[win_N], varNum[win_N], ** size_c;
	size_c = new int* [2];
	vector<vector<Mat>> censusCode(win_N);

	for (int i = 0; i < win_N; i++)
	{
		codeLength[i] = (census_W[i][0] * 2 + 1) *(census_W[i][1] * 2 + 1) * channels;
		if (param_.censusFunc == 3)
			codeLength[i] += 8 * channels;
		varNum[i] = ceil(codeLength[i] / 64.);
		size_c[i] = new int[3];
		size_c[i][0] = h_;
		size_c[i][1] = w_;
		size_c[i][2] = varNum[i];
		censusCode[i].resize(2);
		censusCode[i][0].create(3, size_c[i], CV_64F);
		censusCode[i][0] = 0;
		censusCode[i][1].create(3, size_c[i], CV_64F);
		censusCode[i][1] = 0;
		delete[] size_c[i];
		size_c[i] = NULL;
	}
	delete[] size_c;
	size_c = NULL;

	//gen_cen_vm_<0>(vm_census[0]);
	//gen_cen_vm_<1>(vm_census[1]);
	if (channels == 3)
	{
		//genCensusCode(I_c, censusCode, param_.W_V, param_.W_U);
	}
	else
	{
		for (int i = 0; i < win_N; i++)
		{
			if (param_.censusFunc == 0)
			{
				/*	genCensus(I_g[0], censusCode[0], param_.W_V, param_.W_U);
					genCensus(I_g[1], censusCode[1], param_.W_V, param_.W_U); *///编码0
				genCensusCode<uchar>(I_g, censusCode[i], census_W[i][0], census_W[i][1]);
			}
			//else if (param_.censusFunc == 1)
			//{
			//	genCensusCode_neighC1(I_g, censusCode, param_.W_V, param_.W_U); //
			//}
			//else if (param_.censusFunc == 2)
			//	genCensusCode_neighC2(I_g, censusCode, param_.W_V, param_.W_U); //
			else if (param_.censusFunc == 3)
				genCensusCode_NC_Sur(I_g, censusCode[i], census_W[i][0], census_W[i][1]);
		}
		//genCensusCode<float>(grad, censusCode[2], census_W[2][0], census_W[2][1]);
	}

	for (int i = 0; i < imgNum; i++)
	{
		gen_cenVM_XOR(censusCode[0], vm_census[i], codeLength[0], truncRatio, i);  // 0
		//gen_cenVM_XOR_From2Code(censusCode, vm_census[i], codeLength, varNum, truncRatio, i);
		//gen_cenVM_XOR_From2Code_tem(grad[i], censusCode, vm_census[i], codeLength, varNum, i);
	}

		//genCensusVm(censusCode, vm_census[i], i);  // 1

	cout << "finish census cal" << endl;
	clock_t end = clock();
	clock_t time = end - start;
#ifdef DEBUG 
	saveTime(time, "Census");
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
	asdCal(vm_asd, "AD", imgNum, 1000);
	//bt_color(vm_asd);
	//BF(vm_asd, DT);
	censusCal(vm_census, 1);
	adCensus(vm_asd, vm_census);
	cout << "finish ADCensus cal" << endl;
	clock_t end = clock();
	clock_t time = end - start;
	cout << "ADCensus time：" << time << endl;
	saveTime(time, "ADCensus");
}

void StereoMatching::ADCensusGrad()
{
	cout << "start ADCensusGrad cal" << endl;
	clock_t start = clock();
	int imgNum = Do_LRConsis ? 2 : 1;
	vector<Mat> vm_asd(imgNum), vm_census(imgNum), vm_grad(imgNum);
	for (int num = 0; num < imgNum; num++)
	{
		vm_asd[num].create(3, size_vm, CV_32F);
		vm_census[num].create(3, size_vm, CV_32F);
		vm_grad[num].create(3, size_vm, CV_32F);
	}
	//asdCal(vm_asd, "AD", imgNum, 10); 
	asdCal(vm_asd, "AD", imgNum, 255); 
	//BF(vm_asd, DT);
	//censusCal(vm_census, 0.3);
	censusCal(vm_census, 1);
	//grad(vm_grad, 2); //xxxx
	grad(vm_grad, 500); //xxxx
	adCensuGradCombine(vm_asd, vm_census, vm_grad);
	//adCensus(vm_asd, vm_census);
	cout << "finish ADCensusGrad cal" << endl;
	clock_t end = clock();
	clock_t time = end - start;
	cout << "ADCensus time：" << time << endl;
	saveTime(time, "ADCensusGrad");
}

void StereoMatching::costCalculate()
{
	clearErrTxt();  // 清空误差txt
	clearTimeTxt();  // 清空时间txt

	string::size_type idx;
	string::size_type idy;
	int imgNum = Do_LRConsis ? 2 : 1;
	// AD、SD
	if (costcalculation == "AD" || costcalculation == "SD")
		asdCal(vm, costcalculation, imgNum, 20);

	// BT
	if (costcalculation == "BT")
		bt(vm);

	else if (costcalculation == "grad")
		grad(vm, 500);
		//grad_color(vm, 7.2);

	else if (costcalculation == "TruncAD")
		truncAD(vm);

	else if (costcalculation == "adGrad")
		adGrad(vm);

	else if (costcalculation == "censusGrad")
		censusGrad(vm);

	// census
	else if (costcalculation == "Census")
		censusCal(vm, 1);

	// ZNCC
	else if (costcalculation == "ZNCC")
		ZNCC(I_g[0], I_g[1], vm);

	// adCensus
	else if (costcalculation == "ADCensus")
		ADCensusCal();

	else if (costcalculation == "ADCensusGrad")
		ADCensusGrad();

	// BF 盒式滤波
	//idx = aggregation.find("BF");
	//if (idx != string::npos)
	//	BF(vm);
	if (aggregation == "BF")
		BF(vm);

	//AWS
	else if (aggregation == "AWS")
		AWS();

	// cost aggregation
	//CBCA
	else if (aggregation == "CBCA")
		CBCA();

	else if (aggregation == "GF")
	{
		GF();
	}

	else if (aggregation == "FIF")
		//FIF();
		FIF_Improve();

	else if (aggregation == "NL")
		NL();

	else if (aggregation == "GFNL")
		GFNL();
	std::cout << "finish cost aggregation" << endl;
	std::cout << endl;
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
		//symmetry_borderCopy_3D(vm[i], vm_borderIp, W_V_B, W_V_B, W_U_B, W_U_B);  // 
		//BF_BI(vm_borderIp, vm[i], W_V_B, W_U_B);
		boxFilter(vm[i], vm[i], -1, Size(12, 12));
	}
#ifdef DEBUG
	saveFromVm(vm, "BF");
#endif // DEBUG
}


void StereoMatching::dispOptimize()
{
	DP[0].create(h_, w_, CV_16S);
	DP[1].create(h_, w_, CV_16S);

	if (optimization == "sgm")
	{
		int num = 1;
		if (Do_refine && Do_LRConsis)
			num = 2;
		for (int i = 0; i < num; i++)
		{
			//clock_t start = clock();
			bool leftFirst = i == 0 ? true : false;
			sgm(vm[i], leftFirst);
			//Mat vm_res = vm[i].clone();
			//sgm_verti(vm[i], leftFirst);
			//sgm_hori(vm[i], leftFirst);

			//for (int v = 0; v < h_; v++)
			//{
			//	for (int u = 0; u < w_; u++)
			//	{
			//		float* vmP = vm[i].ptr<float>(v, u);
			//		float min = numeric_limits<float>::max();
			//		for (int d = 0; d < d_; d++)
			//		{
			//			if (vmP[d] < min)
			//				min = vmP[d];
			//		}
			//		for (int d = 0; d < d_; d++)
			//		{
			//			vmP[d] -= min;
			//		}
			//	}
			//}
			//vm[i] = vm[i] * 0.025 + vm_res;
			//sgm_hori(vm[i], leftFirst);
			//sgm_verti(vm[i], leftFirst);
			//clock_t end = clock();
			//clock_t time = end - time;
			//saveTime(time, "sgm" + to_string(i));
		}
	}

	else if(optimization == "so")
	{
		int num = Do_LRConsis ? 2 : 1;
		for (int i = 0; i < num; i++)
		{
			Mat vm_res = vm[i].clone();
			//so_T2D(vm_res, DP[i], I_c);
			so(vm[i], DP[i], I_c);
			//so_change(vm[i], DP[i], I_c);
			//so_R2L(vm[i], DP[i], I_c);
			if (i == 0)
				saveFromDisp<short, 0>(DP[0], "so", true, true);
			else
				saveFromDisp<short, 1>(DP[1], "so");
		}
	}

	if (optimization == "sgm" || optimization == "")
	{
		int num = Do_refine && Do_LRConsis ? 2 : 1;
		if (param_.Do_vmTop)
		{
			clock_t start = clock();
			int sizeVmTop[] = { h_, w_, param_.vmTop_Num + 1, 2 };
			Mat topDisp(4, sizeVmTop, CV_32F);
			for (int i = 0; i < num; i++)
			{
				Mat vm_copy = vm[i].clone();
				selectTopCostFromVolumn(vm_copy, topDisp, param_.vmTop_thres);
				genDispFromTopCostVm2(topDisp, DP[i]);
			}
			clock_t end = clock();
			clock_t time = end - start;
			saveTime(time, "topVm");
		}
		else
			for (int i = 0; i < num; i++)
				gen_dispFromVm(vm[i], DP[i]);
#ifdef DEBUG
		string name = param_.Do_vmTop ? "vmTop" : "wta";
		saveFromDisp<short, 0>(DP[0], name, true, true);
		saveFromDisp<short, 1>(DP[1], name);
#endif // DEBUG
	}
	cout << "disparity computation: " << optimization << " done" << endl;
}

void StereoMatching::refine()
{	
	// 基于“Fast stereo matching using adaptive guided filtering”中的方法
#ifdef USE_RECONCV

	if (true)
	{
		// 左右一致性检测，找出错误点
		Mat errMask(h_, w_, CV_8U, Scalar::all(1));
		LRConsistencyCheck_new(errMask);
		// 重新构造匹配代价卷
		for (int v = 0; v < h_; v++)
		{
			uchar* errP = errMask.ptr<uchar>(v);
			for (int u = 0; u < w_; u++)
			{
				float* cP = vm[0].ptr<float>(v, u);
				if (errP[u] == 0)
				{
					for (int d = 0; d < d_; d++)
						cP[d] = 0;
				}
				else
				{
					float cMin = numeric_limits<float>::max();
					for (int d = 0; d < d_; d++)
					{
						if (cP[d] < cMin)
							cMin = cP[d];
					}
					for (int d = 0; d < d_; d++)
					{
						cP[d] -= cMin;
					}
				}
			}
		}

		Mat Img_tem;
		I_c[0].convertTo(Img_tem, CV_32F, 1 / 255.0);
		Mat wgt_hor(h_ - 1, w_ - 1, CV_32F);
		Mat wgt_ver(h_ - 1, w_ - 1, CV_32F);
		// 计算水平和垂直两两点间的权重
		for (int v = 0; v < h_ - 1; v++)
		{
			float* wgt_hP = wgt_hor.ptr<float>(v);
			float* wgt_vP = wgt_ver.ptr<float>(v);
			for (int u = 0; u < w_ - 1; u++)
			{
				float* IP = Img_tem.ptr<float>(v, u);
				float* IP_nextU = Img_tem.ptr<float>(v, u + 1);
				float* IP_nextV = Img_tem.ptr<float>(v + 1, u);
				//float clrDif_h = 0;
				//float clrDif_v = 0;
				//for (int c = 0; c < 3; c++)
				//{
				//	clrDif_h = max(abs(IP_nextU[c] - IP[c]), clrDif_h);
				//	clrDif_v = max(abs(IP_nextV[c] - IP[c]), clrDif_v);
				//}
				//wgt_hP[u] = exp(-clrDif_h / 0.2);
				//wgt_vP[u] = exp(-clrDif_v /  0.2);
				wgt_hP[u] = pow(IP_nextU[0] - IP[0], 2) + pow(IP_nextU[1] - IP[1], 2) + pow(IP_nextU[2] - IP[2], 2);
				wgt_vP[u] = pow(IP_nextV[0] - IP[0], 2) + pow(IP_nextV[1] - IP[1], 2) + pow(IP_nextV[2] - IP[2], 2);
				wgt_hP[u] = exp(-wgt_hP[u] / (0.08 * 0.08));  //0.08 “Full-Image Guided Filtering for Fast Stereo Matching”
				wgt_vP[u] = exp(-wgt_vP[u] / (0.08 * 0.08));
			}
		}

		Mat* calcu1 = new Mat[d_];
		Mat* calcu2 = new Mat[d_];
		Mat* calcu3 = new Mat[d_];
		for (int d = 0; d < d_; d++)
		{
			calcu1[d] = Mat(h_, w_, CV_32F, Scalar::all(0));
			calcu2[d] = Mat(h_, w_, CV_32F, Scalar::all(0));
			calcu3[d] = Mat(h_, w_, CV_32F, Scalar::all(0));
		}
		// 水平代价积累（从左到右）
		for (int d = 0; d < d_; d++)
		{
			Mat calcu = calcu1[d];
			for (int v = 0; v < h_; v++)
			{
				float* wgt_h = wgt_hor.ptr<float>(v);
				float* cP = calcu.ptr<float>(v);
				cP[0] = vm[0].ptr<float>(v, 0)[d];
				for (int u = 1; u < w_; u++)
				{
					cP[u] = vm[0].ptr<float>(v, u)[d] + cP[u - 1] * wgt_h[u - 1];
				}
			}
		}
		// 水平代价积累（从右到左）
		for (int d = 0; d < d_; d++)
		{
			Mat calcu = calcu2[d];
			for (int v = 0; v < h_; v++)
			{
				float* wgt_h = wgt_hor.ptr<float>(v);
				float* cP = calcu.ptr<float>(v);
				cP[w_ - 1] = vm[0].ptr<float>(v, w_ - 1)[d];
				for (int u = w_ - 2; u >= 0; u--)
				{
					cP[u] = vm[0].ptr<float>(v, u)[d] + cP[u + 1] * wgt_h[u];
				}
			}
		}
		// 水平和
		for (int d = 0; d < d_; d++)
		{
			for (int v = 0; v < h_; v++)
			{
				float* c1 = calcu1[d].ptr<float>(v);
				float* c2 = calcu2[d].ptr<float>(v);
				for (int u = 0; u < w_; u++)
				{
					c1[u] = c1[u] + c2[u] - vm[0].ptr<float>(v, u)[d];
				}
			}
		}

		// 垂直代价积累（从上到下）
		for (int d = 0; d < d_; d++)
		{
			Mat calcu_ori = calcu1[d];
			Mat calcu_new = calcu2[d];
			for (int u = 0; u < w_; u++)
				calcu_new.ptr<float>(0)[u] = calcu_ori.ptr<float>(0)[u];
			for (int v = 1; v < h_; v++)
			{
				float* wgt_v = wgt_ver.ptr<float>(v - 1);
				float* cP_ori = calcu_ori.ptr<float>(v);
				float* cP_new_pre = calcu_new.ptr<float>(v - 1);
				float* cP_new = calcu_new.ptr<float>(v);
				for (int u = 0; u < w_; u++)
					cP_new[u] = cP_new_pre[u] * wgt_v[u] + cP_ori[u];
			}
		}
		// 垂直代价积累（从下到上）
		for (int d = 0; d < d_; d++)
		{
			Mat calcu_ori = calcu1[d];
			Mat calcu_new = calcu3[d];
			for (int u = 0; u < w_; u++)
				calcu_new.ptr<float>(h_ - 1)[u] = calcu_ori.ptr<float>(h_ - 1)[u];
			for (int v = h_ - 2; v >= 0; v--)
			{
				float* wgt_v = wgt_ver.ptr<float>(v);
				float* cP_ori = calcu_ori.ptr<float>(v);
				float* cP_new_pre = calcu_new.ptr<float>(v + 1);
				float* cP_new = calcu_new.ptr<float>(v);
				for (int u = 0; u < w_; u++)
					cP_new[u] = cP_new_pre[u] * wgt_v[u] + cP_ori[u];
			}
		}
		// 垂直加和
		for (int d = 0; d < d_; d++)
		{
			for (int v = 0; v < h_; v++)
			{
				float* c1 = calcu2[d].ptr<float>(v);
				float* c2 = calcu3[d].ptr<float>(v);
				for (int u = 0; u < w_; u++)
				{
					c1[u] = c1[u] + c2[u] - calcu1[d].ptr<float>(v)[u];
				}
			}
		}
		// 执行WTA
		Mat dp_tem(h_, w_, CV_16S);
		for (int v = 0; v < h_; v++)
		{
			short* dP = dp_tem.ptr<short>(v);
			for (int u = 0; u < w_; u++)
			{
				float cmin = numeric_limits<float>::max();
				int d_tar = -1;
				for (int d = 0; d < d_; d++)
				{
					float c = calcu2[d].ptr<float>(v)[u];
					if (c < cmin)
					{
						cmin = c;
						d_tar = d;
					}
				}
				dP[u] = d_tar;
			}
		}

		for (int v = 0; v < h_; v++)
		{
			uchar* eP = errMask.ptr<uchar>(v);
			short* dP = DP[0].ptr<short>(v);
			short* dp_temP = dp_tem.ptr<short>(v);
			for (int u = 0; u < w_; u++)
			{
				if (eP[u] == 0)
					dP[u] = dp_temP[u];
			}
		}
		delete[] calcu1;
		delete[] calcu2;
		delete[] calcu3;
		calcu1 = NULL;
		calcu2 = NULL;
		calcu3 = NULL;
		saveFromDisp<short>(DP[0], "reconstrunct cost vm and tree aggregation");
	}
#else
// 视差细化
Mat Dp_res;

//clock_t start = clock();
//for (int i = 0; i < 3; i++)
//{
	//clock_t start_ = clock();
	//regionVote(DP[0], HVL[0]);
	//regionVote(DP[1], HVL[1]);
	//clock_t end_ = clock();
	//saveTime(end_ - start_, "rv" + to_string(i));
//}
//clock_t end = clock();
//clock_t time = end - start;
//saveTime(time, "rv");

if (StereoMatching::Do_LRConsis)  // LR consisency check
{
	clock_t start = clock();
	//LRConsistencyCheck(DP[0], DP[1], LRC_Err_Mask);
	LRConsistencyCheck_normal(DP[0], DP[1], LRC_Err_Mask);
	clock_t end = clock();
	clock_t time = end - start;
#ifdef DEBUG
	saveTime(time, "LRC");
	saveFromDisp<short>(DP[0], "od");
#endif
	std::cout << "LRConsistencyCheck finished" << endl;
}

if (StereoMatching::Do_calPKR)
{
	calPKR(vm[0], PKR_Err_Mask);
	signDp_UsingPKR(DP[0], PKR_Err_Mask);
	saveFromDisp<short>(DP[0], "PKR");
}

string filename_P;
Mat dispChange(h_, w_, CV_16S);



if (StereoMatching::Do_regionVote)
{
	Dp_res = DP[0].clone();
	if (!param_.has_initArm)
		initArm();
	if (!param_.has_calArms)
		calArms<uchar>(I_c, HVL, HVL_INTERSECTION, param_.cbca_crossL[0], param_.cbca_crossL_out[0], param_.cbca_cTresh[0], param_.cbca_cTresh_out[0]);

	//float rv_ratio[] = {0.4, 0.5, 0.6, 0.7};
	//float rv_ratio[] = {0.4, 0.7, 0.5, 0.4};
	//int rv_s[] = {RV_S, RV_S, RV_S, RV_S};
	float rv_ratio[] = { 0.4, 0.4, 0.4, 0.4 };
	int rv_s[] = { 20, 20, 20, 20 };

	for (int i = 0; i < param_.region_vote_nums; i++)  // region vote
	{
		clock_t start = clock();
		regionVote_my(DP[0], rv_ratio[i], rv_s[i]);
		//RV_combine_BG(DP[0], rv_ratio[i], rv_s[i]);
		//signDispChange_forRV(Dp_res, DP[0], DT, I_mask[0], dispChange);
		//filename_P = "dispRVChange" + to_string(i + 1);
		//saveDispMap<short>(dispChange, filename_P);
		clock_t end = clock();
		clock_t time = end - start;
		cout << "RV" + to_string(i) << " time" << (int)time << endl;
		string name = "RV" + to_string(i);
		saveTime(time, name);
#ifdef DEBUG
		//string name = "RV" + to_string(i);
		//saveTime(time, name);
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
	signDispChange_forRV(Dp_res, DP[0], DT, I_mask[1], dispChange);
	saveDispMap<short>(dispChange, DT, "dispCBBIChange.png");
	saveFromDisp<short>(DP[0], "CBBI");
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

if (StereoMatching::Do_bgIpol)
{
	//Dp_res = DP[0].clone();
	//for (int i = 0; i < param_.region_vote_nums; i++)  // region vote
	BGIpol(DP[0]);
	saveFromDisp<short>(DP[0], "BG", false, false);
#ifdef DEBUG
	saveFromDisp<short>(DP[0], "BG", false, false);
#endif // DEBUG

	//coutInterpolaterEffect(Dp_res, DP[0]);
}

if (StereoMatching::Do_WM)
{
	WM(DP[0], LRC_Err_Mask, I_c[0]);
	saveFromDisp<short>(DP[0], "WM", false, false);
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
	//regionVoteForWholeDispImg(DP[0]);
	saveFromDisp<short>(DP[0], "mb", false, false);
#ifdef DEBUG
	
#endif
	cout << "medianBlur finished" << endl;
}

cout << "disparity refine finished" << endl;
cout << endl;
#endif // USE_RECONCV
}

// 算法描述见论文
void StereoMatching::genDispFromTopCostVm2(Mat& topDisp, Mat& disp)
{
	int num_ = topDisp.size[2];
	if (param_.vmTop_method == 0)
	{
		clock_t start = clock();
		const int disp_DifThres = param_.ts; // 10
		const int disp_DifThres2 = 14; // 14
		const int disp_DifThres3 = 1000; // 5
		const int num_candi = topDisp.size[2];  // 设定的候选视差数加一个实际候选视差指示位
		map<float, int> dispCost_container;
		map<float, int>::iterator iter_dispCost;
		map<int, int> disp__num;
		map<int, int>::iterator iter_disp_num;
		map<int, float> disp__cost;
		map<int, float>::iterator iter_disp_cost;

		// l,u,r,d,lu,rd,ru,ld
		int neigh_v[] = { 0, -1, 0, 1, -1, 1, -1, 1};
		int neigh_u[] = { -1, 0, 1, 0, -1, 1, 1, -1};
		int neigh_len = 8;
	
		for (int v = 0; v < h_; v++)
		{
			short* disP = disp.ptr<short>(v);
			for (int u = 0; u < w_; u++)
			{
				if (u == 0 || v == 0)  // 情况1
					disP[u] = topDisp.ptr<float>(v, u, 0)[0];
				else
				{
					int n = topDisp.ptr<float>(v, u, num_candi - 1)[0];
					if (n == 1) // 情况1
						disP[u] = topDisp.ptr<float>(v, u, 0)[0];
					else if (n > 1)
					{ 
						//bool use_neigh[] = { false, false, false, false, false, false, false, false };
						//bool use_neigh[] = { false, false, false, false, true, true, true, true };
						bool use_neigh[] = { true, true, true, true, true, true, true, true };
						//bool use_neigh[] = { true, true, true, true, false, false, false, false };
						int disp_pre1 = disP[u - 1];
						int disp_pre2 = disp.ptr<short>(v - 1)[u];
						int disp_r = 10000;
						int disp_d = 10000;
						int disp_lt = disp.ptr<short>(v - 1)[u - 1];
						int disp_rd = 10000;
						int disp_rt = 10000;
						int disp_ld = 10000;
						if (u != w_ - 1)
						{
							disp_r = topDisp.ptr<float>(v, u + 1, 0)[0];
							//disp_rt = topDisp.ptr<float>(v - 1, u + 1, 0)[0];
							disp_rt = disp.ptr<short>(v - 1)[u + 1];
						}
						if (v != h_ - 1)
						{
							disp_d = topDisp.ptr<float>(v + 1, u, 0)[0];
							disp_ld = topDisp.ptr<float>(v + 1, u - 1, 0)[0];
						}
						if (u != w_ - 1 && v != h_ - 1)
							disp_rd = topDisp.ptr<float>(v + 1, u + 1, 0)[0];
						clock_t start1 = clock();
						for (int i = 0; i < n; i++)
						{
							int d0 = topDisp.ptr<float>(v, u, i)[0];
							float c0 = topDisp.ptr<float>(v, u, i)[1];
							if (!param_.vmTop_hasCir2)
								dispCost_container.insert(pair<float, int>(c0, d0));
							else
							{
								for (int j = i + 1; j < n; j++)
								{
									int d1 = topDisp.ptr<float>(v, u, j)[0];
									float c1 = topDisp.ptr<float>(v, u, j)[1];
									if (abs(d0 - d1) < disp_DifThres)
									{
										dispCost_container.insert(pair<float, int>(c0, d0));
										dispCost_container.insert(pair<float, int>(c1, d1));
									}
								}
							}
						}

						if (dispCost_container.empty())  // 情况2
						{
							clock_t start3 = clock();
							int difMostSmall1 = numeric_limits<int>::max();
							int disp1 = -1;
							int difMostSmall2 = numeric_limits<int>::max();
							int disp2 = -1;
							int difMostSmall3 = numeric_limits<int>::max();
							int disp3 = -1;
							int difMostSmall4 = numeric_limits<int>::max();
							int disp4 = -1;
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
								int dif3 = abs(disp__ - disp_rt);
								if (dif3 < difMostSmall3)
								{
									difMostSmall3 = dif3;
									disp3 = disp__;
								}
								int dif4 = abs(disp__ - disp_lt);
								if (dif4 < difMostSmall4)
								{
									difMostSmall4 = dif4;
									disp4 = disp__;
								}
							}
							int difMostSmall = min(min(difMostSmall3, difMostSmall4), min(difMostSmall1, difMostSmall2));
							int disp = -1;
							if (difMostSmall == difMostSmall4)
								disp = disp4;
							else if (difMostSmall == difMostSmall1)
								disp = disp1;
							else if (difMostSmall == difMostSmall2)
								disp = disp2;
							else if (difMostSmall == difMostSmall3)
								disp = disp3;

							//int disp = difMostSmall == difMostSmall1 ? disp1 : disp2;
							if (difMostSmall < disp_DifThres3)
								disP[u] = disp;
							else
								disP[u] = topDisp.ptr<float>(v, u, 0)[0];
						}
						else
						{  // 情况3
							clock_t start4 = clock();
							int difMostSmall = numeric_limits<int>::max();
							int disp = -1;
							bool has_find1 = false;
							bool has_find2 = false;
							bool has_find3 = false;
							bool has_find4 = false;
							for (iter_dispCost = dispCost_container.begin(); iter_dispCost != dispCost_container.end(); iter_dispCost++)
							{
								int disp = iter_dispCost->second;
								float cost = iter_dispCost->first;
								disp__num[disp]++;
								disp__cost[disp] += cost;

								//if (!use_neigh[0] && abs(disp - disp_pre1) < disp_DifThres2)
								//{
								//	has_find1 = true;
								//	use_neigh[0] = true;
								//}
								//if (!use_neigh[1] && abs(disp - disp_pre2) < disp_DifThres2)
								//{
								//	has_find2 = true;
								//	use_neigh[1] = true;
								//}
								//if (!use_neigh[2] && abs(disp - disp_r) < disp_DifThres2)
								//{
								//	has_find3 = true;
								//	use_neigh[2] = true;
								//}
								//if (!use_neigh[3] && abs(disp - disp_d) < disp_DifThres2)
								//{
								//	has_find4 = true;
								//	use_neigh[3] = true;
								//}
								//if (!use_neigh[4] && abs(disp - disp_lt) < disp_DifThres2)
								//{
								//	use_neigh[4] = true;
								//}
								//if (!use_neigh[5] && abs(disp - disp_rd) < disp_DifThres2)
								//{
								//	use_neigh[5] = true;
								//}
								//if (!use_neigh[6] && abs(disp - disp_rt) < disp_DifThres2)
								//{
								//	use_neigh[6] = true;
								//}
								//if (!use_neigh[7] && abs(disp - disp_ld) < disp_DifThres2)
								//{
								//	use_neigh[7] = true;
								//}
							}
							int start = 2, end = 2;
							if (has_find1)
								start = 0;
							if (has_find2)
								end = 4;
							//if (true)
							if (true)
							{
								clock_t start2 = clock();
								uchar* tarP = I_c[0].ptr<uchar>(v, u);
								bool isAddNei = true;
								if (param_.vmTop_cir3_doColorLimit)
									isAddNei = false;
								for (int i = 0; i < neigh_len; i++)
								{
									if (use_neigh[i])
									{
										int v_ = v + neigh_v[i];
										int u_ = u + neigh_u[i];
										if (v_ >= 0 && v_ < h_ && u_ >= 0 && u_ < w_)
										{
											uchar* neiP = I_c[0].ptr<uchar>(v_, u_);
							
											if (isAddNei || judgeColorDif(tarP, neiP, 10, 3)) // 增加颜色限制
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
							}
							else
							{
								iter_dispCost = dispCost_container.begin();
								disP[u] = iter_dispCost->second;
							}
							disp__cost.clear();
							disp__num.clear();
							dispCost_container.clear();
						}
					}
				}
			}
		}
		clock_t end = clock();
		cout << "genDispFromTopVm2 time is: " << end - start << endl;
	}
	
	if (param_.vmTop_method == 1)
	{
		for (int v = 0; v < h_; v++)
		{
			short* dP = disp.ptr<short>(v);
			float* tP = topDisp.ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				int num_d = tP[(num_ - 1) * 2];
				if (u == 0 || num_d == 1)
					dP[u] = tP[0];
				else
				{
					int dp_ = -1;
					short dpV_pre = dP[u - 1];
					int d_ldis = 10000;
					float c_ldis = 10000000;
					for (int d = 0; d < num_d; d++)
					{
						int pos = d * 2;
						int s = abs(dpV_pre - tP[pos]);
						if (s < 2 && s < d_ldis)
						{
							d_ldis = s;
							dp_ = tP[pos];
						}
					}
					dP[u] = dp_ == -1 ? tP[0] : dp_;
				}
				tP += num_ * 2;
			}
		}
	}

	if (param_.vmTop_method == 2)
	{
		for (int v = 0; v < h_; v++)
		{
			float* tP = topDisp.ptr<float>(v);
			short* dP = disp.ptr<short>(v);
			for (int u = 0; u < w_; u++)
			{
				int num_t = tP[(num_ - 1) * 2];
				if (u == 0 || num_t == 1)
					dP[u] = tP[0];
				else
				{
					int dif_lst_pre = 1000000;
					int dif_lst_aft = 1000000;
					int d0 = -1;
					int d_pre = dP[u - 1];
					int d1 = -1;
				
					for (int n = 0; n < num_t; n++)
					{
						int dif = abs(tP[n * 2] - d_pre);
						if (dif < 2 && dif < dif_lst_pre)
						{
							dif_lst_pre = dif;
							d0 = tP[n * 2];
						}
					}
					if (u < w_ - 1)
					{
						int d_aft = tP[num_ * 2];
						for (int n = 0; n < num_t; n++)
						{
							int dif = abs(tP[n * 2] - d_aft);
							if (dif < 2 && dif < dif_lst_aft)
							{
								dif_lst_aft = dif;
								d1 = tP[n * 2];
							}
						}
					}
					if (d0 != -1 && d1 == -1)
						dP[u] = d0;
					else if (d0 == -1 && d1 != -1)
						dP[u] = d1;
					else if (d0 == -1 && d1 == -1)
						dP[u] = tP[0];
					else if (d0 != -1 && d1 != -1)
					{
						int cdif_pre = 0;
						int cdif_atf = 0;
						uchar* cP = I_c[0].ptr<uchar>(v, u);
						for (int c = 0; c < 3; c++)
						{
							cdif_pre += abs(cP[c] - cP[c - 3]);
							cdif_atf += abs(cP[c] - cP[c + 3]);
						}
						dP[u] = cdif_pre <= cdif_atf ? d0 : d1;
					}
				}
				tP += num_ * 2;
			}
		}

	}
}


void StereoMatching::clearErrTxt()
{
	string addr = param_.savePath + param_.err_name;
	if (_access(param_.savePath.c_str(), 0) == -1)  // 判断文件是否存在
		createDirectory(param_.savePath.c_str());
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
	if (_access(param_.savePath.c_str(), 0) == -1)  // 判断文件是否存在
		createDirectory(param_.savePath.c_str());
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

void StereoMatching::openCSV()
{
	ofs_.open(root + param_.errCsvName, ios::out | ios::app);
}

void StereoMatching::closeCSV()
{
	ofs_.close();
}

void StereoMatching::pipeline()
{
	const auto t1 = std::chrono::system_clock::now();
	ofs_.open(root + param_.errCsvName, ios::out | ios::app);

	//clearErrTxt();  // 清空误差txts
	//clearTimeTxt();  // 清空时间txt
	
	costCalculate(); // 代价计算（含代价聚合）

	//if (object == "")
		//showArms(248, 325);
		//showArms(347, 302);
		//showArms(310, 378); Plastic
		showArms(37, 166); //Teddy
		//showArms(22, 313);
		//showArms(78, 299);
		//showArms(257, 136);
		//showArms(257, 121); // y
		//showArms(257, 105);  // z Laundry
	if (Do_dispOptimize)
		dispOptimize(); // 视差优化

	if (Do_refine)
		refine();// 视差细化

	const auto t2 = std::chrono::system_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "disparity computation time: " << duration << "[msec]" << std::endl;
	saveTime(duration, "ALL");
	ofs_.close();
}

void StereoMatching::costScan(cv::Mat& Lr, cv::Mat& vm, int rv, int ru, bool leftFirst)
{
	//CV_Assert(Lr.size == vm.size);

	int h = vm.size[0];
	int w = vm.size[1];
	int n = param_.numDisparities;

	int v0 = 0, v1 = h, u0 = 0, u1 = w, dv = +1, du = +1;
	if ((rv > 0) || (rv == 0 && ru > 0))
	{
		v0 = h - 1; v1 = -1; u0 = w - 1; u1 = -1; dv = -1; du = -1;
	}

	bool preIsInner = true;
	for (int v = v0; v != v1; v += dv)
	{
		for (int u = u0; u != u1; u += du)
		{
			preIsInner = true;
			if (v + rv > h - 1 || v + rv < 0 || u + ru > w - 1 || u + ru < 0)
				preIsInner = false;
			try
			{
				switch (vm.depth())
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
				//vmP[d] = sum / numOfDirec;
				vmP[d] = sum;
			}
		}
	}
}

StereoMatching::StereoMatching(cv::Mat& I1_c, cv::Mat& I2_c, cv::Mat& I1_g, cv::Mat& I2_g, cv::Mat& DT, cv::Mat& all_mask, cv::Mat& nonocc_mask, cv::Mat& disc_mask, const Parameters& param) : param_(param)
{
	this->h_ = I1_c.rows;
	this->w_ = I1_c.cols;
	this->d_ = param_.numDisparities;
	this->I_c.resize(2);
	this->I_g.resize(2);
	this->I_mask.resize(3);
	this->I_c[0] = I1_c;
	/*cv::cvtColor(I1_c, this->Lab, COLOR_BGR2Lab);
	cv::cvtColor(I1_c, this->HSV, COLOR_BGR2HSV);*/
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
		param_.sgm_P1 = 1.0;
		param_.sgm_P2 = 3.0;
	}

	idx = aggregation.find("AWS");
	if (idx != string::npos)
		param_.sgm_P1 = 0.5, param_.sgm_P2 = 1.0;

	idx = aggregation.find("GF");
	if (idx != string::npos)
	{
		guideVm.resize(2);
		for (int i = 0; i < 2; i++)
		{
			guideVm[i].resize(param_.numDisparities);
			for (int d = 0; d < param_.numDisparities; d++)
				guideVm[i][d].create(h_, w_, CV_32F);
		}
		param_.sgm_P1 = 1.0, param_.sgm_P2 = 3.0;
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

template<class T>
void PrintMat(const Mat& mat)
{
	int rows = mat.rows;
	int cols = mat.cols;
	printf("\n%d x %d Matrix\n", rows, cols);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			cout << mat.at<T>(r, c) << "\t";
		}
		printf("\n");
	}
	printf("\n");
}

void SolveAll(StereoMatching**& smPyr, const int PY_LVL, const float REG_LAMBDA)
{
	printf("\n\t\tSolve All");
	printf("\n\t\tReg param: %.4lf\n", REG_LAMBDA);
	// construct regularization matrix
	Mat regMat = Mat::zeros(PY_LVL, PY_LVL, CV_32FC1);
	for (int s = 0; s < PY_LVL; s++) {
		if (s == 0) {
			regMat.at<float>(s, s) = 1 + REG_LAMBDA;
			if (PY_LVL > 1)
				regMat.at<float>(s, s + 1) = -REG_LAMBDA;
		}
		else if (s == PY_LVL - 1) {
			regMat.at<float>(s, s) = 1 + REG_LAMBDA;
			regMat.at<float>(s, s - 1) = -REG_LAMBDA;
		}
		else {
			regMat.at<float>(s, s) = 1 + 2 * REG_LAMBDA;
			regMat.at<float>(s, s - 1) = -REG_LAMBDA;
			regMat.at<float>(s, s + 1) = -REG_LAMBDA;
		}
	}
	Mat regInv = regMat.inv();
	float* invWgt = new float[PY_LVL];
	for (int s = 0; s < PY_LVL; s++) {
		invWgt[s] = regInv.at<float>(0, s);
	}
	PrintMat<float>(regInv);
	int hei = smPyr[0]->h_;
	int wid = smPyr[0]->w_;
	int disp = smPyr[0]->d_;


	//
	// Left Cost Volume
	//
	int img_n = StereoMatching::Do_refine ? 2 : 1;
	for (int n = 0; n < img_n; n++)
	{
		for (int y = 0; y < hei; y++) {
			for (int x = 0; x < wid; x++) {
				for (int d = 0; d < disp; d++) // 原代码d是从1开始的
				{
					int curY = y;
					int curX = x;
					int curD = d;
					float sum = 0;
					for (int s = 0; s < PY_LVL; s++) {
						float curCost = smPyr[s]->vm[n].ptr<float>(curY, curX)[curD];
#ifdef _DEBUG
						if (y == 160 && x == 160) {
							printf("\ns=%d(%d,%d)\td=%d\tcost=%.4lf", s, curY, curX, curD, curCost);
				}
#endif
						sum += invWgt[s] * curCost;
						curY = curY / 2;
						curX = curX / 2;
						curD = (curD + 1) / 2;
			}
					smPyr[0]->vm[n].ptr<float>(y, x)[d] = sum;
		}
	}
		}

	}
	delete[] invWgt;
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

void StereoMatching::LRConsistencyCheck_normal(cv::Mat& D1, cv::Mat& D2, Mat& errMask, int LOR)
{
	const int n = param_.numDisparities;

	//OMP_PARALLEL_FOR
	if (LOR == 0)
	{
		for (int v = 0; v < h_; v++)
		{
			short* _D1 = D1.ptr<short>(v);
			short* _D2 = D2.ptr<short>(v);
			float* DTP = DT.ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				const short d = _D1[u];
				if (d < 0 || u - d < 0 || abs(d - _D2[u - d]) > param_.LRmaxDiff)
					_D1[u] = -1;
			}
		}
	}
}

void StereoMatching::LRConsistencyCheck(cv::Mat& D1, cv::Mat& D2, Mat& errMask, int LOR)
{
	const int n = param_.numDisparities;
	int dis_occ = 0;
	int dis_mis = 0;
	int dis_err = 0;

	errMask.create(h_, w_, CV_8U);
	Mat errMask1 = Mat::zeros(h_, w_, CV_8U);
	errMask = 0;

	//OMP_PARALLEL_FOR
	if (LOR == 0)
	{
		for (int v = 0; v < h_; v++)
		{
			short* _D1 = D1.ptr<short>(v);
			short* _D2 = D2.ptr<short>(v);
			float* DTP = DT.ptr<float>(v);
			uchar* errP = errMask.ptr<uchar>(v);
			for (int u = 0; u < w_; u++)
			{
				const short d = _D1[u];
				if (d < 0 || u - d < 0 || abs(d - _D2[u - d]) > param_.LRmaxDiff)
				{
					errP[u] = 255;
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
#ifdef DEBUG
		imwrite(param_.savePath + "LR0.png", errMask);
#endif // DEBUG

	}
	else if (LOR == 1)
	{
		for (int v = 0; v < h_; v++)
		{
			short* _D1 = D1.ptr<short>(v);
			short* _D2 = D2.ptr<short>(v);
			uchar* errP = errMask1.ptr<uchar>(v);
			for (int u = 0; u < w_; u++)
			{
				const short d = _D2[u];
				if (d < 0 || u + d >= w_ || abs(d - _D1[u + d]) > param_.LRmaxDiff)
				{
					int disp = param_.DISP_OCC;
					for (int d = 0; d < n && u + d <= w_ - 1; d++)
					{
						if (_D1[u + d] == d)
						{
							disp = param_.DISP_MIS;
							break;
						}
					}
					_D2[u] = disp;
					errP[u] = 255;
				}
			}
		}
		imwrite(param_.savePath + "LR1.png", errMask1);
	}
}


void StereoMatching::LRConsistencyCheck_new(Mat& errorMask)
{
	const int Thres = 0;
	for (int v = 0; v < h_; v++)
	{
		uchar* eP = errorMask.ptr<uchar>(v);
		short* d1 = DP[0].ptr<short>(v);
		short* d2 = DP[1].ptr<short>(v);
		for (int u = 0; u < w_; u++)
		{
			int d1V = d1[u];
			if (d1V < 0 || u - d1V < 0 || abs(d1V - d2[u - d1V]) > Thres)
				eP[u] = 0;
		}
	}
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
void StereoMatching::gen_ad_sd_vm(Mat& vm_asd, int LOR, int AOS, float trunc)
{
	float DEFAULT = param_.is_adNorm ? 1 : trunc;
	const int channels = param_.SD_AD_channel;
	Mat I0 = channels == 3 ? I_c[0] : I_g[0];
	Mat I1 = channels == 3 ? I_c[1] : I_g[1];

	const int n = param_.numDisparities;
	const int pow_index = AOS == 0 ? 1 : 2;
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
					float sum = 0;
					for (int cha = 0; cha < channels; cha++)
						sum += pow(abs((float)lPtr[cha] - (float)rPtr[cha]), pow_index);
					vPtr[d] = min(sum / channels, trunc);
					if (param_.is_adNorm)
						vPtr[d] /= trunc;
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

void StereoMatching::gen_ad_vm(Mat& vm, Mat& I_l, Mat& I_r, float Trunc, int LOR)
{
	float DEFAULT = param_.is_adNorm ? Trunc / 255 : Trunc;

	const int n = param_.numDisparities;
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
			float* vPtr = vm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				int uL = u + d * leftCoefficient;
				int uR = u + d * rightCoefficient;
				if (uL >= w_ || uR < 0)
					vPtr[d] = DEFAULT;
				else
				{
					uchar* lP = I_l.ptr<uchar>(v, uL);
					uchar* rP = I_r.ptr<uchar>(v, uR);
					float sum = 0;
					for (int cha = 0; cha < I_l.channels(); cha++)
						sum += pow(lP[cha] - rP[cha], 2);
					vPtr[d] = min(sqrt(sum), DEFAULT);
					if (param_.is_adNorm)
						vPtr[d] /= DEFAULT;
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

void StereoMatching::genTrueHorVerArms(vector<Mat>& HVL, vector<Mat>& HVL_INTERSECTION)
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

template <typename T>
static bool judgeColorDif(T* target, T* refer, int thres, int channel)
{
	for (int c = 0; c < channel; c++)
	{
		if (abs(target[c] - refer[c]) > thres)
			return false;
	}
	return true;
}
//  L, L_out, C_D, C_D_out, minL

template <typename T>
void StereoMatching::calHorVerDis(Mat& I, Mat& cross, int L, int DIF, int minL)
{
	int channels = I.channels();
	clock_t start = clock();
	cout << "start cal CrossArm" << endl;
	Mat I_enhance;
	//edgeEnhance(I, I_enhance);
	//I = I_enhance;
	if (param_.doGF_bef_calArm)
		ximgproc::guidedFilter(I, I, I, 9, 50);
	//xxxxx
	//medianBlur(I, I, 3);
	const int h = I.rows;
	const int w = I.cols;

	int du_[] = { -1, 0 }, dv_[] = { 0, -1 }, du, dv;
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

		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				T* IPtr = I.ptr<T>(v, u);
				ushort* cPtr = cross.ptr<ushort>(v, u);
				ushort arm = 1;
				for (; arm <= L; arm++)
				{
					int v_arm = v + arm * dv, u_arm = u + arm * du;
					if (v_arm < 0 || v_arm >= h || u_arm < 0 || u_arm >= w)
						break;
					T* armPtr = I.ptr<T>(v_arm, u_arm);
					if (armPtr[0] < 0)
						break;
					T* armPrePtr = I.ptr<T>(v + (arm - 1) * dv, u + (arm - 1) * du);
					float DIF_ = DIF - (arm / L) * DIF;
					//bool neighborDifInThres = judgeColorDif<T>(armPtr, armPrePtr, DIF, channels);
					bool initPresDifInThres = judgeColorDif<T>(IPtr, armPtr, DIF, channels);
					//if (!neighborDifInThres || !initPresDifInThres)
					if (!initPresDifInThres)
						break;
				}
				if (--arm >= minL)
					cPtr[direc] = arm;  // l已经被减过1了
				else
				{
					for (int len = minL; len >= 0; len--)  // 动态求取边框区像素点的外部臂长，使其能取到min(minL，dis2border)
					{
						if (u + len * du >= 0 && u + len * du <= w - 1 && v + len * dv >= 0 && v + len * dv <= h - 1)
						{
							cPtr[direc] = len;
							break;
						}
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


	clock_t end = clock();
	clock_t time = end - start;
	cout << "finish cal CrossArm" << endl;
#ifdef DEBUG
	saveTime(time, "genArmForImg_L");
#endif // DEBUG
}
template <typename T>
void StereoMatching::calHorVerDis(Mat& I, Mat& cross, int L, int L_out, int C_D, int C_D_out, int minL)
{
	clock_t start = clock();
	cout << "start cal CrossArm" << endl;
	Mat I_enhance;
	//edgeEnhance(I, I_enhance);
	//I = I_enhance;
	if (param_.doGF_bef_calArm)
		ximgproc::guidedFilter(I, I, I, 9, 50);
	//xxxxx
	//medianBlur(I, I, 3);
	const int h = I.rows;
	const int w = I.cols;
	const int channel = I.channels();

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

		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				T* IPtr = I.ptr<T>(v, u);
				ushort* cPtr = cross.ptr<ushort>(v, u);
				ushort arm = 1;
				for (; arm <= L_out; arm++)
				{
					int v_arm = v + arm * dv, u_arm = u + arm * du;
					if (v_arm < 0 || v_arm >= h || u_arm < 0 || u_arm >= w)
						break;
					T* armPtr = I.ptr<T>(v_arm, u_arm);
					T* armPrePtr = I.ptr<T>(v + (arm - 1) * dv, u + (arm - 1) * du);
					bool neighborDifInThres = judgeColorDif<T>(armPtr, armPrePtr, C_D, channel);
					bool initPresDifInThres = arm <= L ? judgeColorDif<T>(IPtr, armPtr, C_D, channel) : judgeColorDif<T>(IPtr, armPtr, C_D_out, channel);
					if (!neighborDifInThres || !initPresDifInThres)
						break;
				}
				if (--arm >= minL)
					cPtr[direc] = arm;  // l已经被减过1了
				else
				{
					for (int len = minL; len >= 0; len--)  // 动态求取边框区像素点的外部臂长，使其能取到min(minL，dis2border)
					{
						if (u + len * du >= 0 && u + len * du <= w - 1 && v + len * dv >= 0 && v + len * dv <= h - 1)
						{
							cPtr[direc] = len;
							break;
						}
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

	clock_t end = clock();
	clock_t time = end - start;
#ifdef DEBUG
	saveTime(time, "genArmForImg_L");
#endif // DEBUG
}

template <typename T>
void StereoMatching::calHorVerDis(Mat& I, Mat& cross, int L0, int L1, int L2, int thresh0, int thresh1, int thresh2, int minL)
{
	clock_t start = clock();
	cout << "start cal CrossArm" << endl;
	Mat I_enhance;
	//edgeEnhance(I, I_enhance);
	//I = I_enhance;
	if (param_.doGF_bef_calArm)
		ximgproc::guidedFilter(I, I, I, 9, 50);
	//xxxxx
	//medianBlur(I, I, 3);
	const int h = I.rows;
	const int w = I.cols;
	const int channel = I.channels();

	int du_[] = { -1, 0 }, dv_[] = { 0, -1 }, du, dv;
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

		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				T* IPtr = I.ptr<T>(v, u);
				ushort* cPtr = cross.ptr<ushort>(v, u);
				ushort arm = 1;
				for (; arm <= L2; arm++)
				{
					int v_arm = v + arm * dv, u_arm = u + arm * du;
					if (v_arm < 0 || v_arm >= h || u_arm < 0 || u_arm >= w)
						break;
					T* armPtr = I.ptr<T>(v_arm, u_arm);
					T* armPrePtr = I.ptr<T>(v + (arm - 1) * dv, u + (arm - 1) * du);
					bool neighborDifInThres = judgeColorDif<T>(armPtr, armPrePtr, thresh0, channel);
					bool initPresDifInThres = true;
					if (arm <= L0)
						initPresDifInThres = judgeColorDif<T>(IPtr, armPtr, thresh0, channel);
					else if (arm <= L1)
						initPresDifInThres = judgeColorDif<T>(IPtr, armPtr, thresh1, channel);
					else 
						initPresDifInThres = judgeColorDif<T>(IPtr, armPtr, thresh2, channel);
					if (!neighborDifInThres || !initPresDifInThres)
						break;
				}
				if (--arm >= minL)
					cPtr[direc] = arm;  // l已经被减过1了
				else
				{
					for (int len = minL; len >= 0; len--)  // 动态求取边框区像素点的外部臂长，使其能取到min(minL，dis2border)
					{
						if (u + len * du >= 0 && u + len * du <= w - 1 && v + len * dv >= 0 && v + len * dv <= h - 1)
						{
							cPtr[direc] = len;
							break;
						}
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

	clock_t end = clock();
	clock_t time = end - start;

	saveTime(time, "genArmForImg_L");
}

template <typename T>
void StereoMatching::calHorVerDis(Mat& I, Mat& cross, vector<int> L, vector<int> thresh, int minL)
{
	clock_t start = clock();
	cout << "start cal CrossArm" << endl;
	Mat I_enhance;
	//edgeEnhance(I, I_enhance);
	//I = I_enhance;
	if (param_.doGF_bef_calArm)
		ximgproc::guidedFilter(I, I, I, 9, 50);
	//xxxxx
	//medianBlur(I, I, 3);
	const int h = I.rows;
	const int w = I.cols;
	const int channel = I.channels();

	int du_[] = { -1, 0 }, dv_[] = { 0, -1 }, du, dv;
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

		int len = L.size();

		for (int v = 0; v < h; v++)
		{
			for (int u = 0; u < w; u++)
			{
				T* IPtr = I.ptr<T>(v, u);
				ushort* cPtr = cross.ptr<ushort>(v, u);
				ushort arm = 1;
				for (; arm <= L[len - 1]; arm++)
				{
					int v_arm = v + arm * dv, u_arm = u + arm * du;
					if (v_arm < 0 || v_arm >= h || u_arm < 0 || u_arm >= w)
						break;
					T* armPtr = I.ptr<T>(v_arm, u_arm);
					T* armPrePtr = I.ptr<T>(v + (arm - 1) * dv, u + (arm - 1) * du);
					bool neighborDifInThres = judgeColorDif<T>(armPtr, armPrePtr, 20, channel);
					bool initPresDifInThres = true;
					for (int n_L = 0; n_L < len; n_L++)
					{
						if (arm <= L[n_L])
						{
							initPresDifInThres = judgeColorDif<T>(IPtr, armPtr, thresh[n_L], channel);
							break;
						}
					}

					if (!neighborDifInThres || !initPresDifInThres)
						break;
				}
				if (--arm >= minL)
					cPtr[direc] = arm;  // l已经被减过1了
				else
				{
					for (int len = minL; len >= 0; len--)  // 动态求取边框区像素点的外部臂长，使其能取到min(minL，dis2border)
					{
						if (u + len * du >= 0 && u + len * du <= w - 1 && v + len * dv >= 0 && v + len * dv <= h - 1)
						{
							cPtr[direc] = len;
							break;
						}
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

	clock_t end = clock();
	clock_t time = end - start;

	saveTime(time, "genArmForImg_L");
}

void StereoMatching::calHorVerDis2(int imgNum, int channel, uchar L, uchar L_out, uchar C_D, uchar C_D_out, uchar minL)
{
	clock_t start = clock();
	cout << "start cal CrossArm" << endl;
	Mat I = channel == 3 ? I_c[imgNum].clone() : I_g[imgNum].clone();
	if (param_.doGF_bef_calArm)
		ximgproc::guidedFilter(I, I, I, 9, 50);
	//xxxxx
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
					int c_d = C_D - ((float)C_D / L * arm) + 10;
					//c_d = max(c_d, 10);
					bool neighborDifInThres = judgeColorDif(armPtr, armPrePtr, C_D, channel);
					bool initPresDifInThres = arm <= L ? judgeColorDif(IPtr, armPtr, c_d, channel) : judgeColorDif(IPtr, armPtr, C_D_out, channel);
					if (!initPresDifInThres || !neighborDifInThres)
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

//  L, L_out, C_D, C_D_out, minL xxxxxx
void StereoMatching::calTileDis(int imgNum, int channel, uchar L, uchar L_out, uchar C_D, uchar C_D_out, uchar minL)
{
	clock_t start = clock();
	cout << "start cal CrossArm" << endl;
	Mat I = channel == 3 ? I_c[imgNum].clone() : I_g[imgNum].clone();
	//ximgproc::guidedFilter(I, I, I, 9, 1);
	//xxxxx
	//medianBlur(I, I, 3);
	const int h = I.rows;
	const int w = I.cols;

	int du_[] = { -1, 1 }, dv_[] = { -1, -1 }, du, dv;
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

		Mat cross = tileCrossL[imgNum];
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

void StereoMatching::drawArmForPoint(Mat& HVL, int* v, int* u, int num)
{
	Mat I_out = I_c[0].clone();
	for (int n = 0; n < num; n++)
	{
		int v_c = *v++;
		int u_c = *u++;
		ushort* lP = HVL.ptr<ushort>(v_c, u_c);
		uchar* I_outP = I_out.ptr<uchar>(v_c) + 3 * (u_c - lP[0]);
		for (int u_ = -lP[0]; u_ <= lP[1]; u_++)
		{
			I_outP[0] = 34;
			I_outP[1] = 34;
			I_outP[2] = 178;
			I_outP += 3;
		}
		I_outP = I_out.ptr<uchar>(v_c - lP[2]) + 3 * u_c;
		for (int v_ = -lP[2]; v_ <= lP[3]; v_++)
		{
			I_outP[0] = 34;
			I_outP[1] = 34;
			I_outP[2] = 178;
			I_outP += 3 * w_;
		}
	}
	string path = param_.savePath; // ******
	system(("IF NOT EXIST " + path + " (mkdir " + path + ")").c_str());
	path += "armTeddy.png";
	imwrite(path, I_out);
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
	
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vm0Ptr = vm0.ptr<float>(v, u);
			float* vm1Ptr = vm1.ptr<float>(v, u);
			float* combinedVmPtr = combinedVm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				//combinedVmPtr[d] = (1 - a) * (1 - exp(-vm0Ptr[d] / ARU0)) + a * (1 - exp(-vm1Ptr[d] / ARU1));
				combinedVmPtr[d] = 2 - exp(-vm0Ptr[d] / ARU0) - exp(-vm1Ptr[d] / ARU1);
			}
				
		}
	}
}

void StereoMatching::gen_vm_from3vm_exp(cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, Mat& vm2, const float ARU0, const float ARU1, float ARU2, int LOR)
{
	CV_Assert(vm0.depth() == CV_32F);
	CV_Assert(vm1.depth() == CV_32F);
	CV_Assert(combinedVm.depth() == CV_32F);
	Mat dispMap(h_, w_, CV_16S);

	const int n = param_.numDisparities;

	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			short* armP = HVL[LOR].ptr<short>(v, u);
			int shortest = 10000;
			for (int dir = 0; dir < 4; dir++)
			{
				if (armP[dir] < shortest)
					shortest = armP[dir];
			}
			float a = 1 - exp(-1.0f / shortest);
			//float cache = 1 - exp(-1.5f / shortest);
			//float a = cache * 0.11 * 2;
			//float c = cache * 0.89 * 2;
			//float b = 1 - cache;
			//a = a / (a + b + c);
			//c = a / 0.11 * 0.89;
			//b = 1 - a - c;
			//float a = 0.333333;
			//float c = 0.333333;
			//float b = 0.333333;
			//float c = (1 - a) * 0.89;

			float* vm0Ptr = vm0.ptr<float>(v, u);
			float* vm1Ptr = vm1.ptr<float>(v, u);
			float* vm2Ptr = vm2.ptr<float>(v, u);
			float* combinedVmPtr = combinedVm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
			{
				//float adGradv = 0.7 * vm0Ptr[d] + 0.3 * vm2Ptr[d];
				//combinedVmPtr[d] = a * (1 - exp(-adGradv / 2)) + (1 - a) * (1 - exp(-vm1Ptr[d] / 10));
				//combinedVmPtr[d] = 2 - exp(-adGradv / 2) - exp(-vm1Ptr[d] / 10);
				//float adCenv = a * vm0Ptr[d] + (1 - a) * vm1Ptr[d];
				float adCenv = 0.5 * vm0Ptr[d] + 0.5 * vm1Ptr[d];
				//float adCenv = a * (1 - exp(-vm0Ptr[d] / 5)) + (1 - a) * (1 - exp(-vm1Ptr[d] / 10));
				//combinedVmPtr[d] = 0.11 * adCenv + 0.89 *(1 - exp(-vm2Ptr[d] / 2));
				//combinedVmPtr[d] = 0.7 * adCenv + 0.3 * vm2Ptr[d];
				//combinedVmPtr[d] = 2 - exp(-adCenv / 0.5) - exp(-vm2Ptr[d] / 2);
				combinedVmPtr[d] = 0.7 * (1 - exp(-adCenv / 1)) + 0.3 * (1 - exp(-vm2Ptr[d] / 2));

				//combinedVmPtr[d] = a * adGradv + (1 - a) * vm1Ptr[d];
				//combinedVmPtr[d] = 0.3 * adGradv + 0.7 * vm1Ptr[d];
				//combinedVmPtr[d] = 3 - exp(-vm0Ptr[d] / ARU0) - exp(-vm1Ptr[d] / ARU1) - exp(-vm2Ptr[d] / ARU2);
			}
				//combinedVmPtr[d] = 3 - exp(-vm0Ptr[d] / ARU0) - exp(-vm1Ptr[d] / ARU1) - exp(-vm2Ptr[d] / ARU2);
				//combinedVmPtr[d] = a * (1 - exp(-vm0Ptr[d] / ARU0)) + b * (1 - exp(-vm1Ptr[d] / ARU1)) + c * (1 - exp(-vm2Ptr[d] / ARU2));
				
				//combinedVmPtr[d] = 3 - exp(-vm0Ptr[d] / ARU0) - exp(-vm1Ptr[d] / ARU1);
		}
	}
}

void StereoMatching::gen_vm_from3vm_add(cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, cv::Mat& vm2, int LOR)
{
	CV_Assert(vm0.depth() == CV_32F);
	CV_Assert(vm1.depth() == CV_32F);
	CV_Assert(combinedVm.depth() == CV_32F);
	Mat dispMap(h_, w_, CV_16S);

	const int n = param_.numDisparities;

	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vm0Ptr = vm0.ptr<float>(v, u);
			float* vm1Ptr = vm1.ptr<float>(v, u);
			float* vm2Ptr = vm2.ptr<float>(v, u);
			float* combinedVmPtr = combinedVm.ptr<float>(v, u);

			short* armP = HVL[LOR].ptr<short>(v, u);
			int shortest = 10000;
			for (int dir = 0; dir < 4; dir++)
			{
				if (armP[dir] < shortest)
					shortest = armP[dir];
			}

			float b = exp(-0.5f / shortest);
			float a = (1 - b) * 0.11;
			float c = (1 - b) * 0.89;
			//float a = 0.2;
			//float b = 0.4;
			//float c = 1.0 - a - b;

			for (int d = 0; d < n; d++)
			{
				//combinedVmPtr[d] = 0.4 * (vm0Ptr[d] * a + vm1Ptr[d] * (1 - a)) + 0.6 * vm2Ptr[d];
				combinedVmPtr[d] = vm0Ptr[d] * a + vm1Ptr[d] * b  + vm2Ptr[d] * c;
			}
		}
	}
}

void StereoMatching::gen_vm_from2vm_expadpWgt(Mat& HVL, cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, const float ARU0, const float ARU1, int LOR)
{
	CV_Assert(vm0.depth() == CV_32F);
	CV_Assert(vm1.depth() == CV_32F);
	CV_Assert(combinedVm.depth() == CV_32F);
	Mat dispMap(h_, w_, CV_16S);

	const int n = param_.numDisparities;

	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			short* arm = HVL.ptr<short>(v, u);
			int shortest = 1000000;
			for (int dir = 0; dir < 4; dir++)
			{
				if (arm[dir] < shortest)
					shortest = arm[dir];
			}
			float a_ = 1 - exp(-0.5 / shortest);
			float* vm0Ptr = vm0.ptr<float>(v, u);
			float* vm1Ptr = vm1.ptr<float>(v, u);
			float* combinedVmPtr = combinedVm.ptr<float>(v, u);
			for (int d = 0; d < n; d++)
				combinedVmPtr[d] = a_*(1 - exp(-vm0Ptr[d] / ARU0)) + (1 - a_) * (1 - exp(-vm1Ptr[d] / ARU1));
		}
	}
}

void StereoMatching::gen_vm_from2vm_add(cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, int LOR)
{
	CV_Assert(vm0.depth() == CV_32F);
	CV_Assert(vm1.depth() == CV_32F);
	CV_Assert(combinedVm.depth() == CV_32F);
	Mat dispMap(h_, w_, CV_16S);

	const int n = param_.numDisparities;

	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vm0Ptr = vm0.ptr<float>(v, u);
			float* vm1Ptr = vm1.ptr<float>(v, u);
			float* combinedVmPtr = combinedVm.ptr<float>(v, u);

			short* armP = HVL[LOR].ptr<short>(v, u);
			int shortest = 10000;
			for (int dir = 0; dir < 4; dir++)
			{
				if (armP[dir] < shortest)
					shortest = armP[dir];
			}
			float a = 1 - exp(-1.0f / shortest);

			for (int d = 0; d < n; d++)
			{
				//if (u - W_U < 0 || u + W_U >= w_ || u - d * rightCoefficient - W_U < 0 || u + d * leftCoefficient + W_U >= w_ ||
				//	v - W_V < 0 || v + W_V >= h_)
				//	combinedVmPtr[d] = DEFAULT_MC;
				//else
				//combinedVmPtr[d] = vm0Ptr[d] * weight0 + vm1Ptr[d] * weight1;
				//combinedVmPtr[d] = vm0Ptr[d] * (1 - a) + vm1Ptr[d] * a;
				combinedVmPtr[d] = vm0Ptr[d] * 0.5 + vm1Ptr[d] * 0.5;
			}
		}
	}
}


void StereoMatching::gen_vm_from2vm_fixWgt(Mat& vm0, float wgt0, Mat& vm1, float wgt1, Mat& dst)
{
	CV_Assert(vm0.depth() == CV_32F);
	CV_Assert(vm1.depth() == CV_32F);
	CV_Assert(dst.depth() == CV_32F);
	Mat dispMap(h_, w_, CV_16S);

	const int n = param_.numDisparities;


	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* vm0Ptr = vm0.ptr<float>(v, u);
			float* vm1Ptr = vm1.ptr<float>(v, u);
			float* dstPtr = dst.ptr<float>(v, u);

			for (int d = 0; d < n; d++)
			{
				//if (u - W_U < 0 || u + W_U >= w_ || u - d * rightCoefficient - W_U < 0 || u + d * leftCoefficient + W_U >= w_ ||
				//	v - W_V < 0 || v + W_V >= h_)
				//	combinedVmPtr[d] = DEFAULT_MC;
				//else
				//combinedVmPtr[d] = vm0Ptr[d] * weight0 + vm1Ptr[d] * weight1;
				dstPtr[d] = vm0Ptr[d] * wgt0 + vm1Ptr[d] * wgt1;
			}
		}
	}
}

void StereoMatching::gen_vm_from2vm_add_wgt(cv::Mat& HVL, cv::Mat& combinedVm, cv::Mat& vm0, cv::Mat& vm1, const float weight0, const float weight1, int LOR)
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
			//******
			float* vm0Ptr = vm0.ptr<float>(v, u);
			float* vm1Ptr = vm1.ptr<float>(v, u);
			float* combinedVmPtr = combinedVm.ptr<float>(v, u);
			short* armP = HVL.ptr<short>(v, u);
			int arm_smallest = 100000;
			for (int dir = 0; dir < 4; dir++)
			{
				if (armP[dir] < arm_smallest)
					arm_smallest = armP[dir];
			}
			float a = 1 - exp(-0.5 / arm_smallest);
			for (int d = 0; d < n; d++)
			{
				//if (u - W_U < 0 || u + W_U >= w_ || u - d * rightCoefficient - W_U < 0 || u + d * leftCoefficient + W_U >= w_ ||
				//	v - W_V < 0 || v + W_V >= h_)
				//	combinedVmPtr[d] = DEFAULT_MC;
				//else
				combinedVmPtr[d] = (1 - a) * vm0Ptr[d] * weight0 + a * vm1Ptr[d] * weight1;
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
	CV_Assert(vm.depth() == CV_32F);
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

void StereoMatching::calPKR(Mat& vm, Mat& mask)
{
	Mat vmC = vm.clone();
	mask.create(h_, w_, CV_8U);
	mask = 0;
	vector<float> cost(2);
	float ratio_PKR = 0.1;
	int num_sum = 0;
	int num = 0;
	for (int v = 0; v < mask.rows; v++)
	{
		uchar* mP = mask.ptr<uchar>(v);
		for (int u = 0; u < mask.cols; u++)
		{
			num_sum++;
			float* vmP = vmC.ptr<float>(v, u);
			for (int n = 0; n < 2; n++)
			{
				float min = numeric_limits<float>::max();
				int disp = -1;
				for (int d = 0; d < d_; d++)
				{
					if (vmP[d] < min)
					{
						min = vmP[d];
						disp = d;
					}
				}
				cost[n] = min;
				vmP[disp] = numeric_limits<float>::max();
			}
			if ((cost[1] - cost[0]) / cost[1] < ratio_PKR)
			{
				mP[u] = 1;
				num++;
			}
		}
	}
	cout << "pkr比例：" << ((float)num / (float)num_sum) << endl;
}

void StereoMatching::signDp_UsingPKR(Mat& disp, Mat& PKR_Err_Mask)
{
	for (int v = 0; v < h_; v++)
	{
		uchar* mP = PKR_Err_Mask.ptr<uchar>(v);
		short* dP = disp.ptr<short>(v);
		for (int u = 0; u < w_; u++)
		{
			if (mP[u] > 0 && dP[u] >= 0)
				dP[u] = param_.DISP_PKR;
		}
	}
}

void StereoMatching::combine2Vm(vector<Mat>& vm, vector<Mat>& vm2)
{
	int imgNum = Do_refine ? 2 : 1;
	//Mat mask(h_, w_, CV_8U, Scalar::all(0));
	Mat mask;
	for (int n = 0; n < imgNum; n++)
	{
		calPKR(vm[n], mask);
		int step = 0;
		for (int v = 0; v < h_; v++)
		{
			uchar* pkrP = mask.ptr<uchar>(v);
			for (int u = 0; u < w_; u++)
			{
				if (pkrP[u] == 1)
				{
					step++;
					float* vmP = vm[n].ptr<float>(v, u);
					float* vmP2 = vm2[n].ptr<float>(v, u);
					for (int d = 0; d < d_; d++)
						vmP[d] = vmP[d] * 0.3 + vmP2[d] * 0.7;
				}
			}
		}
		cout << "step: " << step << endl;
	}
}

void StereoMatching::combine2Vm_2(vector<Mat>& vm, vector<Mat>& vm2, vector<Mat>& HVL)
{
	int imgNum = Do_refine ? 2 : 1;
	int armLimit = 10;
	//Mat mask(h_, w_, CV_8U, Scalar::all(0));
	int step = 0;
	for (int n = 0; n < imgNum; n++)
	{
		if (n == 0)
		{
			arm_Mask.create(h_, w_, CV_8U);
			arm_Mask = 0;
		}

		Mat HVL_ = HVL[n];
		int step = 0;
		for (int v = 0; v < h_; v++)
		{
			uchar* armMP = arm_Mask.ptr<uchar>(v);
			for (int u = 0; u < w_; u++)
			{
				ushort* arm = HVL_.ptr<ushort>(v, u);

				bool armInRange = true;
				for (int dir = 0; dir < 4; dir++)
				{
					if (arm[dir] > armLimit)
					{
						armInRange = false;
						break;
					}
				}
				if (armInRange) // 最长臂长低于阈值
				{
					armMP[u] = 1;
					step++;
					float* vmP = vm[n].ptr<float>(v, u);
					float* vmP2 = vm2[n].ptr<float>(v, u);
					for (int d = 0; d < d_; d++)
						vmP[d] = vmP[d] * 0.3 + vmP2[d] * 0.7;
				}
			}
		}
		cout << "step: " << step << endl;
	}
}

void StereoMatching::combine2Vm_3(vector<Mat>& vm, vector<Mat>& vm2)
{
	int imgNum = Do_refine ? 2 : 1;
	//Mat mask(h_, w_, CV_8U, Scalar::all(0));
	int step = 0;
	float* vmP[2];
	float disThres = 0;
	for (int n = 0; n < imgNum; n++)
	{
		if (n == 0)
		{
			arm_Mask.create(h_, w_, CV_8U);
			arm_Mask = 0;
		}

		int step = 0;
		for (int v = 0; v < h_; v++)
		{
			uchar* armP = arm_Mask.ptr<uchar>(v);
			for (int u = 0; u < w_; u++)
			{
				vmP[0] = vm[n].ptr<float>(v, u);
				vmP[1] = vm2[n].ptr<float>(v, u);

				float c_min[2];
				for (int w = 0; w < 2; w++)
				{
					c_min[w] = vmP[w][0];
					for (int d = 1; d < d_; d++)
					{
						if (c_min[w] > vmP[w][d])
							c_min[w] = vmP[w][d];
					}
				}

				bool choseISSmall = true;
				if (c_min[1] < c_min[0])
				{
					float dis = c_min[0] - c_min[1];
					if (dis / c_min[0] > disThres)
						choseISSmall = false;
						//choseISSmall = false;
				}
				if (!choseISSmall)
				{
					step++;
					armP[u] = 1;
					for (int d = 0; d < d_; d++)
						vmP[0][d] = 0.3 * vmP[0][d] + 0.7 * vmP[1][d];
				}
			}
		}
		cout << "step: " << step << endl;
	}
}

void StereoMatching::combine2Vm_4(vector<Mat>& vm, vector<Mat>& vm2)
{
	int imgNum = Do_refine ? 2 : 1;
	//Mat mask(h_, w_, CV_8U, Scalar::all(0));
	int step = 0;
	float* vmP[2];
	float disThres = 0;

	arm_Lst.create(h_, w_, CV_32F);
	for (int v = 0; v < h_; v++)
	{
		float* armLP = arm_Lst.ptr<float>(v);
		for (int u = 0; u < w_; u++)
		{
			ushort* crossP = HVL[0].ptr<ushort>(v, u);
			ushort lst = crossP[0];
			for (int num = 1; num < 4; num++)
			{
				lst = max(lst, crossP[num]);
			}
			armLP[u] = lst;
		}
	}
	boxFilter(arm_Lst, arm_Lst, -1, Size(3, 3));
	const float armLthres = 5;

	for (int n = 0; n < imgNum; n++)
	{
		if (n == 0)
		{
			arm_Mask.create(h_, w_, CV_8U);
			arm_Mask = 0;
		}

		int step = 0;
		for (int v = 0; v < h_; v++)
		{
			uchar* armP = arm_Mask.ptr<uchar>(v);
			float* armLP = arm_Lst.ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				vmP[0] = vm[n].ptr<float>(v, u);
				vmP[1] = vm2[n].ptr<float>(v, u);
				if (armLP[u] < armLthres)
				{
					armP[u] = 1;
					for (int d = 0; d < d_; d++)
					{
						//vmP[0][d] = vmP[0][d] * 0.3 + vmP[1][d] * 0.7;
						vmP[0][d] = vmP[0][d] * 0 + vmP[1][d] * 1;
					}
				}
				

			}
		}
		cout << "step: " << step << endl;
	}
}

void StereoMatching::CBCA()
{
	if (true)
	{
		if (param_.cbca_double_win)
		{
			vector<Mat> vm2(2);
			vm2[0] = vm[0].clone();
			vm2[1] = vm[1].clone();
			initArm();
			calArms<uchar>(I_c, HVL, HVL_INTERSECTION, param_.cbca_crossL[1], param_.cbca_crossL_out[1], param_.cbca_cTresh[1], param_.cbca_cTresh_out[1]);
			cbca_core(HVL, HVL_INTERSECTION, vm2, 2);  // 大窗口

			calArms<uchar>(I_c, HVL, HVL_INTERSECTION, param_.cbca_crossL[0], param_.cbca_crossL_out[0], param_.cbca_cTresh[0], param_.cbca_cTresh_out[0]);
			cbca_core(HVL, HVL_INTERSECTION, vm, 2);  // 原始窗口
			//cbca_aggregate(2, vm2);
			//cbca_aggregate(0, vm);

			//combine2Vm(vm, vm2);
			//combine2Vm_2(vm, vm2, HVL);
			//combine2Vm_3(vm, vm2);
			combine2Vm_4(vm, vm2);
			//vm[0] = vm[0] * 0.7+ vm2[0]*0.3;
			//vm[1] = vm[1] * 0.7 + vm2[1] * 0.3;
		}
		else
			cbca_aggregate(0, vm);
	}
	else
	{
		//vector<Mat> vm_res(2);
		//vector<Mat> disp_res(2);
		//vm_res[0] = vm[0].clone();
		//vm_res[1] = vm[1].clone();
		//for (int i = 0; i < 2; i++)
		//{
		//	//boxFilter(vm_res[i], vm_res[i], -1, Size(5, 5));
		//	ximgproc::guidedFilter(I_c[i], vm_res[i], vm_res[i], param_.gf_r[0], param_.gf_eps[0]);  // gf_r gf_eps
		//	disp_res[i].create(h_, w_, CV_16S);
		//}
		//for (int i = 0; i < 2; i++)
		//{
		//	gen_dispFromVm(vm_res[i], disp_res[i]);
		//	if (i == 0)
		//		saveFromDisp<short>(disp_res[i], "box(5,5)", false, true);
		//}
		//Mat dp_copy = disp_res[0].clone();
		//LRConsistencyCheck(dp_copy, disp_res[1], LRC_Err_Mask);
		//LRConsistencyCheck(disp_res[0], disp_res[1], LRC_Err_Mask, 1);
		//dp_copy.copyTo(disp_res[0]);
		//vector<int> L = { 17, 34 };
		//vector<int> thresh = { 20, 5};
		vector<int> L = { 5, 10, 15, 20, 34};
		vector<int> thresh = { 20, 17, 15, 10, 5};
		//vector<int> L = { 5, 15, 25, 35 };
		//vector<int> thresh = { 30, 20, 5, 3 };

		initArm();
		//calArms<short>(disp_res, HVL, HVL_INTERSECTION, 17, 34, 5, 3);
		//calArms<short>(disp_res, HVL, HVL_INTERSECTION, 10, 1, 20, 3, 30, 5);
		//calArms<short>(disp_res, HVL, HVL_INTERSECTION, 10, 2 34, 2);
		//calArms<uchar>(HVL, HVL_INTERSECTION, I_c, 17, 20);
		//calArms<uchar>(I_c, HVL, HVL_INTERSECTION, 17, 20);
		//calArms<uchar>(I_c, HVL, HVL_INTERSECTION, 5, 15, 30, 30, 20, 5);
		//calArms<uchar>(I_c, HVL, HVL_INTERSECTION, 17, 34, 20, 5);
		calArms<uchar>(I_c, HVL, HVL_INTERSECTION, L, thresh);
		cbca_core(HVL, HVL_INTERSECTION, vm, 2);
	}

}

void StereoMatching::GF()
{
	if (param_.GF_double_win)
	{
		//vector<Mat> vm2(2);
		//vm2[0] = vm[0].clone();
		//vm2[1] = vm[1].clone();
		guideFilter(0, vm);
		//guideFilter(1, vm2);
		//combine2Vm(vm, vm2);
		//vm[0] = vm[0] * 0.7+ vm2[0]*0.3;
		//vm[1] = vm[1] * 0.7 + vm2[1] * 0.3;
	}
	else
		guideFilter(0, vm);
}

void StereoMatching::GFNL()
{
	Mat vm_res = vm[0].clone();
	guideFilter(0, vm);

	NLCCA nl;
	Mat imgL_f, imgR_f;
	I_c[0].copyTo(imgL_f);
	I_c[1].copyTo(imgR_f);
	nl.aggreCV(imgL_f, imgR_f, param_.numDisparities, vm_res);
	Mat wetNL(h_, w_, CV_32FC(d_));
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* wP = wetNL.ptr<float>(v, u);
			for (int d = 0; d < d_; d++)
				wP[d] = 1;
		}
	}
	nl.aggreCV(imgL_f, imgR_f, param_.numDisparities, wetNL);
	vm_res /= wetNL;

	Mat dispMap(h_, w_, CV_16S);
	gen_dispFromVm(vm_res, dispMap);
	if (!param_.Do_vmTop)
		saveFromDisp<short>(dispMap, "NL");

	// 求方差
	Mat I_f, I_f2, I_f2_mean, I_f_mean, I_f_mean2, I_var;
	I_g[0].convertTo(I_f, CV_32F);

	multiply(I_f, I_f, I_f2);
	boxFilter(I_f2, I_f2_mean, -1, Size(19, 19));
	boxFilter(I_f, I_f_mean, -1, Size(19, 19));
	multiply(I_f_mean, I_f_mean, I_f_mean2);
	subtract(I_f2_mean, I_f_mean2, I_var);

	int num = 0;
	arm_Mask.create(h_, w_, CV_8U);
	arm_Mask = 0;
	for (int v = 0; v < h_; v++)
	{
		float* IP = I_var.ptr<float>(v);
		uchar* mP = arm_Mask.ptr<uchar>(v);
		for (int u = 0; u < w_; u++)
		{
			float* vmP = vm[0].ptr<float>(v, u);
			float* vm_resP = vm_res.ptr<float>(v, u);
			if (IP[u] < 400)
			{
				mP[u] = 1;
				num++;
				for (int d = 0; d < d_; d++)
				{
					vmP[d] = vm_resP[d];
				}
			}
			else
			{
				for (int d = 0; d < d_; d++)
				{
					vmP[d] = 0.5 * vm_resP[d] + 0.5 * vmP[d];
				}
			}
		}
	}
	cout << "NL num:" << num << endl;
	//vm[0] = 0.5 * vm[0] + 0.5 * vm_res;
}

void StereoMatching::guideFilter(int paramNum, vector<Mat>& vm)
{
	//vmTrans(vm, guideVm);  // 从3维mat值赋给2维的mat数组
	const int n = param_.numDisparities;
	vector<Mat> I(2);
	// xxxx

	int num = Do_LRConsis && Do_refine ? 2 : 1;
	for (int i = 0; i < num; i++)
	{
		I[i] = param_.gf_channel_isColor ? I_c[i].clone() : I_g[i].clone();
		I[i].convertTo(I[i], CV_32F);
#ifdef MY_GUIDE
		split(vm[i], guideVm[i]);
		for (int d = 0; d < n; d++)
		{
			//guideFilterCore(guideVm[i][d], I_g[i], guideVm[i][d], 9, 0.0001); // radius: 8, 4 // epsilon: 500, 0.0001
			guideVm[i][d] = guideFilterCore_matlab(I[i], guideVm[i][d], 9, 0.0001);
		}
		merge(guideVm[i], vm[i]);
#else
		ximgproc::guidedFilter(I[i], vm[i], vm[i], param_.gf_r[paramNum], param_.gf_eps[paramNum]);  // gf_r gf_eps

#endif
	}
	//vmTrans(guideVm, vm); // 从2维mat数组赋为3维的mat
	Mat dispMap(h_, w_, CV_16S);
	Mat	dispMap2(h_, w_, CV_16S);
	Mat	result(h_, w_, CV_16S);
	gen_dispFromVm(vm[0], dispMap);
	if (!param_.Do_vmTop)
		saveFromDisp<short>(dispMap, "guide");
	else
	{
		int sizeVmTop[] = { h_, w_, param_.vmTop_Num + 1, 2 };
		Mat topDisp(4, sizeVmTop, CV_32F);
		Mat vm_copy = vm[0].clone();
		selectTopCostFromVolumn(vm_copy, topDisp, param_.vmTop_thres);
		signCorrectFromTopVm("correctFromTopVmGuide.png", topDisp, DT);
		//if (object == "teddy")
			//genExcelFromTopDisp(topDisp, DT);
		//genDispFromTopCostVm(topDisp, dispMap2);
		genDispFromTopCostVm2(topDisp, dispMap2);
		signDispChange_for2Disp(dispMap, dispMap2, DT, I_mask[1], result);
		saveDispMap<short>(result, DT, "candidate_Change");
		saveFromDisp<short>(dispMap2, "guideCand");
	}
}

void StereoMatching::FIF()
{
	int img_num = 1;
	if (Do_refine && Do_LRConsis)
		img_num = 2;

	const float eps = 0.08;
	for (int i = 0; i < img_num; i++)
	{
		Mat Img_tem;
		I_c[i].convertTo(Img_tem, CV_32F, 1 / 255.0);
		Mat wgt_hor(h_, w_, CV_32F);
		Mat wgt_ver(h_, w_, CV_32F);
		// 计算水平两两点间的权重
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_ - 1; u++)
			{
				float* wgt_hP = wgt_hor.ptr<float>(v);
				float* IP = Img_tem.ptr<float>(v, u);
				float* IP_nextU = Img_tem.ptr<float>(v, u + 1);

				wgt_hP[u] = pow(IP_nextU[0] - IP[0], 2) + pow(IP_nextU[1] - IP[1], 2) + pow(IP_nextU[2] - IP[2], 2);
				wgt_hP[u] = exp(-wgt_hP[u] / (eps * eps));  //0.08 “Full-Image Guided Filtering for Fast Stereo Matching”
			}
		}
		// 计算垂直两两点间的权重
		for (int v = 0; v < h_ - 1; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* wgt_vP = wgt_ver.ptr<float>(v);
				float* IP = Img_tem.ptr<float>(v, u);
				float* IP_nextV = Img_tem.ptr<float>(v + 1, u);

				wgt_vP[u] = pow(IP_nextV[0] - IP[0], 2) + pow(IP_nextV[1] - IP[1], 2) + pow(IP_nextV[2] - IP[2], 2);
				wgt_vP[u] = exp(-wgt_vP[u] / (eps * eps));  //0.08 “Full-Image Guided Filtering for Fast Stereo Matching”
			}
		}

		Mat* calcu1 = new Mat[d_];
		Mat* calcu2 = new Mat[d_];
		Mat* calcu3 = new Mat[d_];
		for (int d = 0; d < d_; d++)
		{
			calcu1[d] = Mat(h_, w_, CV_32F, Scalar::all(0));
			calcu2[d] = Mat(h_, w_, CV_32F, Scalar::all(0));
			calcu3[d] = Mat(h_, w_, CV_32F, Scalar::all(0));
		}
		// 水平代价积累（从左到右）
		for (int d = 0; d < d_; d++)
		{
			Mat calcu = calcu1[d];
			for (int v = 0; v < h_; v++)
			{
				float* wgt_h = wgt_hor.ptr<float>(v);
				float* cP = calcu.ptr<float>(v);
				cP[0] = vm[i].ptr<float>(v, 0)[d];
				for (int u = 1; u < w_; u++)
				{
					cP[u] = vm[i].ptr<float>(v, u)[d] + cP[u - 1] * wgt_h[u - 1];
				}
			}
		}
		// 水平代价积累（从右到左）
		for (int d = 0; d < d_; d++)
		{
			Mat calcu = calcu2[d];
			for (int v = 0; v < h_; v++)
			{
				float* wgt_h = wgt_hor.ptr<float>(v);
				float* cP = calcu.ptr<float>(v);
				cP[w_ - 1] = vm[i].ptr<float>(v, w_ - 1)[d];
				for (int u = w_ - 2; u >= 0; u--)
				{
					cP[u] = vm[i].ptr<float>(v, u)[d] + cP[u + 1] * wgt_h[u];
				}
			}
		}
		// 水平和
		cout << "水平和" << endl;
		for (int d = 0; d < d_; d++)
		{
			for (int v = 0; v < h_; v++)
			{
				float* c1 = calcu1[d].ptr<float>(v);
				float* c2 = calcu2[d].ptr<float>(v);
				for (int u = 0; u < w_; u++)
				{
					c1[u] = c1[u] + c2[u] - vm[i].ptr<float>(v, u)[d];
				}
			}
		}

		// 垂直代价积累（从上到下）
		for (int d = 0; d < d_; d++)
		{
			Mat calcu_ori = calcu1[d];
			Mat calcu_new = calcu2[d];
			for (int u = 0; u < w_; u++)
				calcu_new.ptr<float>(0)[u] = calcu_ori.ptr<float>(0)[u];
			for (int v = 1; v < h_; v++)
			{
				float* wgt_v = wgt_ver.ptr<float>(v - 1);
				float* cP_ori = calcu_ori.ptr<float>(v);
				float* cP_new_pre = calcu_new.ptr<float>(v - 1);
				float* cP_new = calcu_new.ptr<float>(v);
				for (int u = 0; u < w_; u++)
					cP_new[u] = cP_new_pre[u] * wgt_v[u] + cP_ori[u];
			}
		}
		// 垂直代价积累（从下到上）
		for (int d = 0; d < d_; d++)
		{
			Mat calcu_ori = calcu1[d];
			Mat calcu_new = calcu3[d];
			for (int u = 0; u < w_; u++)
				calcu_new.ptr<float>(h_ - 1)[u] = calcu_ori.ptr<float>(h_ - 1)[u];
			for (int v = h_ - 2; v >= 0; v--)
			{
				float* wgt_v = wgt_ver.ptr<float>(v);
				float* cP_ori = calcu_ori.ptr<float>(v);
				float* cP_new_pre = calcu_new.ptr<float>(v + 1);
				float* cP_new = calcu_new.ptr<float>(v);
				for (int u = 0; u < w_; u++)
					cP_new[u] = cP_new_pre[u] * wgt_v[u] + cP_ori[u];
			}
		}
		// 垂直加和
		for (int d = 0; d < d_; d++)
		{
			for (int v = 0; v < h_; v++)
			{
				float* c1 = calcu2[d].ptr<float>(v);
				float* c2 = calcu3[d].ptr<float>(v);
				float* origin = calcu1[d].ptr<float>(v);
				for (int u = 0; u < w_; u++)
				{
					c1[u] = c1[u] + c2[u] - origin[u];
				}
			}
		}
		cout << "垂直和" << endl;
		// 更新
		for (int d = 0; d < d_; d++)
		{
			Mat cNewP = calcu2[d];
			for (int v = 0; v < h_; v++)
			{
				float* cNP = cNewP.ptr<float>(v);
				for (int u = 0; u < w_; u++)
					vm[i].ptr<float>(v, u)[d] = cNP[u];
			}
		}
		cout << "跟新代价卷" << endl;

		delete[] calcu1;
		delete[] calcu2;
		delete[] calcu3;
		calcu1 = NULL;
		calcu2 = NULL;
		calcu3 = NULL;
	}
	saveFromVm(vm, "FIF");	
}

void StereoMatching::FIF_Improve()
{
	int img_num = 1;
	if (Do_refine && Do_LRConsis)
		img_num = 2;

	const float eps = 0.08;
	const float Pn = 2;
	for (int i = 0; i < img_num; i++)
	{
		Mat Img_tem;
		I_c[i].convertTo(Img_tem, CV_32F, 1 / 255.0);
		Mat wgt_hor(h_, w_, CV_32F);
		Mat wgt_ver(h_, w_, CV_32F);
		// 计算水平两两点间的权重
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_ - 1; u++)
			{
				float* wgt_hP = wgt_hor.ptr<float>(v);
				float* IP = Img_tem.ptr<float>(v, u);
				float* IP_nextU = Img_tem.ptr<float>(v, u + 1);

				wgt_hP[u] = pow(IP_nextU[0] - IP[0], 2) + pow(IP_nextU[1] - IP[1], 2) + pow(IP_nextU[2] - IP[2], 2);
				wgt_hP[u] = exp(-wgt_hP[u] / (eps * eps));  //0.08 “Full-Image Guided Filtering for Fast Stereo Matching”
			}
		}
		// 计算垂直两两点间的权重
		for (int v = 0; v < h_ - 1; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* wgt_vP = wgt_ver.ptr<float>(v);
				float* IP = Img_tem.ptr<float>(v, u);
				float* IP_nextV = Img_tem.ptr<float>(v + 1, u);

				wgt_vP[u] = pow(IP_nextV[0] - IP[0], 2) + pow(IP_nextV[1] - IP[1], 2) + pow(IP_nextV[2] - IP[2], 2);
				wgt_vP[u] = exp(-wgt_vP[u] / (eps * eps));  //0.08 “Full-Image Guided Filtering for Fast Stereo Matching”
			}
		}

		Mat calcu1(h_, w_, CV_32FC(d_), Scalar::all(0));
		Mat calcu2(h_, w_, CV_32FC(d_), Scalar::all(0));
		Mat calcu3(h_, w_, CV_32FC(d_), Scalar::all(0));

		// 水平代价积累（从左到右）

		for (int v = 0; v < h_; v++)
		{
			float* cP = calcu1.ptr<float>(v, 0);
			for (int d = 0; d < d_; d++)
				cP[d] = vm[i].ptr<float>(v, 0)[d];
			float* wgt_h = wgt_hor.ptr<float>(v);
			for (int u = 1; u < w_; u++)
			{
				cP = calcu1.ptr<float>(v ,u);
				float* cP_pre = calcu1.ptr<float>(v, u - 1);
				float* vmP = vm[i].ptr<float>(v, u);
				for (int d = 0; d < d_; d++)
				{
					float c_dMinus = d > 0 ? cP_pre[d - 1] +  Pn : numeric_limits<float>::max();
					float c_dPlus = d < d_ - 1 ? cP_pre[d + 1] + Pn : numeric_limits<float>::max();
					float c_d = cP_pre[d];
					cP[d] = vmP[d] + min(min(c_dMinus, c_dPlus), c_d) * wgt_h[u - 1];
				}
			}
		}
		// 水平代价积累（从右到左）
		for (int v = 0; v < h_; v++)
		{
			float* cP = calcu2.ptr<float>(v, w_ - 1);
			for (int d = 0; d < d_; d++)
				cP[d] = vm[i].ptr<float>(v, w_ - 1)[d];

			float* wgt_h = wgt_hor.ptr<float>(v);
			for (int u = w_ - 2; u >= 0; u--)
			{
				cP = calcu2.ptr<float>(v, u);
				float* cP_pre = calcu2.ptr<float>(v, u + 1);
				float* vmP = vm[i].ptr<float>(v, u);
				for (int d = 0; d < d_; d++)
				{
					float c_dMinus = d > 0 ? cP_pre[d - 1] + Pn : numeric_limits<float>::max();
					float c_dPlus = d < d_ - 1 ? cP_pre[d + 1] + Pn : numeric_limits<float>::max();
					float c_d = cP_pre[d];
					cP[d] = vmP[d] + min(min(c_dMinus, c_dPlus), c_d) * wgt_h[u];
				}
			}
		}
		// 水平和
		cout << "水平和" << endl;
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* c1 = calcu1.ptr<float>(v, u);
				float* c2 = calcu2.ptr<float>(v, u);
				float* vmP = vm[i].ptr<float>(v, u);
				for (int d = 0; d < d_; d++)
					c1[d] = c1[d] + c2[d] - vmP[d];
			}
		}

		// 垂直代价积累（从上到下）
		for (int u = 0; u < w_; u++)
		{
			float* cP_new = calcu2.ptr<float>(0, u);
			float* cP_old = calcu1.ptr<float>(0, u);
			for (int d = 0; d < d_; d++)
				cP_new[d] = cP_old[d];
		}
		for (int v = 1; v < h_; v++)
		{
			float* wgt_v = wgt_ver.ptr<float>(v - 1);
			for (int u = 0; u < w_; u++)
			{
				float* cP_new = calcu2.ptr<float>(v, u);
				float* cP_newPre = calcu2.ptr<float>(v - 1, u);
				float* cP_old = calcu1.ptr<float>(v, u);
				for (int d = 0; d < d_; d++)
				{
					float c_dMinus = d > 0 ? cP_newPre[d - 1] + Pn : numeric_limits<float>::max();
					float c_dPlus = d < d_ - 1 ? cP_newPre[d + 1] + Pn : numeric_limits<float>::max();
					float c_d = cP_newPre[d];
					cP_new[d] = cP_old[d] + min(min(c_dMinus, c_dPlus), c_d) * wgt_v[u];
				}
			}
		}

		// 垂直代价积累（从下到上）
		for (int u = 0; u < w_; u++)
		{
			float* cP_new = calcu3.ptr<float>(h_ - 1, u);
			float* cP_old = calcu1.ptr<float>(h_ - 1, u);
			for (int d = 0; d < d_; d++)
				cP_new[d] = cP_old[d];
		}
		for (int v = h_ - 2; v >= 0; v--)
		{
			float* wgt_v = wgt_ver.ptr<float>(v);
			for (int u = 0; u < w_; u++)
			{
				float* cP_new = calcu3.ptr<float>(v, u);
				float* cP_newPre = calcu3.ptr<float>(v + 1, u);
				float* cP_old = calcu1.ptr<float>(v, u);
				for (int d = 0; d < d_; d++)
				{
					float c_dMinus = d > 0 ? cP_newPre[d - 1] + Pn : numeric_limits<float>::max();
					float c_dPlus = d < d_ - 1 ? cP_newPre[d + 1] + Pn : numeric_limits<float>::max();
					float c_d = cP_newPre[d];
					cP_new[d] = cP_old[d] + min(min(c_dMinus, c_dPlus), c_d) * wgt_v[u];
				}
			}
		}

		// 垂直加和
		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* c1 = calcu2.ptr<float>(v, u);
				float* c2 = calcu3.ptr<float>(v, u);
				float* origin = calcu1.ptr<float>(v, u);
				for (int d = 0; d < d_; d++)
					c1[d] = c1[d] + c2[d] - origin[d];
			}
		}
		cout << "垂直和" << endl;
		// 更新

		for (int v = 0; v < h_; v++)
		{
			for (int u = 0; u < w_; u++)
			{
				float* cP = calcu2.ptr<float>(v, u);
				float* vmP = vm[i].ptr<float>(v, u);
				for (int d = 0; d < d_; d++)
					vmP[d] = cP[d];
			}
		}
		cout << "跟新代价卷" << endl;
	}
	saveFromVm(vm, "FIF_imp");
}

void StereoMatching::NL()
{
	NLCCA nl;
	Mat imgL_f, imgR_f;
	I_c[0].copyTo(imgL_f);
	I_c[1].copyTo(imgR_f);
	nl.aggreCV(imgL_f, imgR_f, param_.numDisparities, vm[0]);
	Mat wetNL(h_, w_, CV_32FC(d_));
	for (int v = 0; v < h_; v++)
	{
		for (int u = 0; u < w_; u++)
		{
			float* wP = wetNL.ptr<float>(v, u);
			for (int d = 0; d < d_; d++)
				wP[d] = 1;
		}
	}
	nl.aggreCV(imgL_f, imgR_f, param_.numDisparities, wetNL);
	vm[0] /= wetNL;

	Mat dispMap(h_, w_, CV_16S);
	gen_dispFromVm(vm[0], dispMap);
	if (!param_.Do_vmTop)
		saveFromDisp<short>(dispMap, "NL");
	dispMap.convertTo(guideDisp, CV_32F);
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

	Mat N = Mat::ones(h_, w_, CV_32FC1);
	N = BoxFilter(N, r);

	Mat tem;
	Size kerS(2 * r + 1, 2 * r + 1);
	Mat mean_I[3], mean_p, mean_Ip[3], cov_Ip[3];
	//boxFilter(p, mean_p, CV_32F, kerS);
	mean_p = BoxFilter(p, r) / N;
	for (int c = 0; c < 3; c++)
	{
		//boxFilter(I_ch[c], mean_I[c], CV_32F, kerS);
		mean_I[c] = BoxFilter(I_ch[c], r) / N;
		multiply(I_ch[c], p, tem);
		//boxFilter(tem, mean_Ip[c], CV_32F, kerS);
		mean_Ip[c] = BoxFilter(tem, r) / N;
		multiply(mean_I[c], mean_p, tem);
		subtract(mean_Ip[c], tem, cov_Ip[c]);
	}

	//  %           br, bg, br
	//	%   Sigma = gb, gg, gr
	//	%           rb, rg, rr

	Mat var_I[6];
	int var_index = 0;
	for (int c = 0; c < 3; c++)
	{
		for (int c_a = c; c_a < 3; c_a++)
		{
			multiply(I_ch[c], I_ch[c_a], tem);
			//boxFilter(tem, var_I[var_index], CV_32F, kerS);
			var_I[var_index] = BoxFilter(tem, r) / N;
			multiply(mean_I[c], mean_I[c_a], tem);
			var_I[var_index] -= tem;
			var_index++;
		}
	}

	Mat a[3];
	for (int c = 0; c < 3; c++)
		a[c] = Mat::zeros(h_, w_, CV_32F);
	Mat eps_eye = Mat::eye(3, 3, CV_32F);
	eps_eye *= eps;
	for (int v = 0; v < h_; v++)
	{
		float* vData[6], * cData[3], * aData[3];
		for (int n = 0; n < 6; n++)
			vData[n] = var_I[n].ptr<float>(v);
		for (int c = 0; c < 3; c++)
		{
			cData[c] = cov_Ip[c].ptr<float>(v);
			aData[c] = a[c].ptr<float>(v);
		}
		for (int u = 0; u < w_; u++)
		{
#ifdef FAST_INV
			Mat sigma = (Mat_<float>(3, 3) <<
				vData[0][u], vData[1][u], vData[2][u],
				vData[1][u], vData[3][u], vData[4][u],
				vData[2][u], vData[4][u], vData[5][u]);

			sigma += eps_eye;
			Mat cov_Ip_13 = (Mat_<float>(1, 3) << cData[0][u], cData[1][u], cData[2][u]);
			tem = cov_Ip_13 * sigma.inv();
			float* temP = tem.ptr<float>(0);
			for (int c = 0; c < 3; c++)
				aData[c][u] = temP[c];
#else
		double c0 = cData[0][u];
		double c1 = cData[1][u];
		double c2 = cData[2][u];
		double a11 = vData[0][u] + eps;
		double a12 = vData[1][u];
		double a13 = vData[2][u];
		double a21 = vData[1][u];
		double a22 = vData[3][u] + eps;
		double a23 = vData[4][u];
		double a31 = vData[2][u];
		double a32 = vData[4][u];
		double a33 = vData[5][u] + eps;
		double DET = a11 * (a33 * a22 - a32 * a23) -
			a21 * (a33 * a12 - a32 * a13) +
			a31 * (a23 * a12 - a22 * a13);
		DET = 1 / DET;
		aData[0][u] = DET * (
			c0 * (a33 * a22 - a32 * a23) +
			c1 * (a31 * a23 - a33 * a21) +
			c2 * (a32 * a21 - a31 * a22)
			);
		aData[1][u] = DET * (
			c0 * (a32 * a13 - a33 * a12) +
			c1 * (a33 * a11 - a31 * a13) +
			c2 * (a31 * a12 - a32 * a11)
			);
		aData[2][u] = DET * (
			c0 * (a23 * a12 - a22 * a13) +
			c1 * (a21 * a13 - a23 * a11) +
			c2 * (a22 * a11 - a21 * a12)
			);
#endif // FAST_INV
		}
	}

	Mat b = mean_p.clone();
	for (int c = 0; c < 3; c++)
	{
		multiply(a[c], mean_I[c], tem);
		b -= tem;
	}

	Mat mean_a[3], mean_b;
	for (int c = 0; c < 3; c++)
		//boxFilter(a[c], mean_a[c], CV_32F, kerS);
		mean_a[c] = BoxFilter(a[c], r) / N;
	//boxFilter(b, mean_b, CV_32F, kerS);
	mean_b = BoxFilter(b, r) / N;

	Mat q = mean_b;
	for (int c = 0; c < 3; c++)
	{
		multiply(mean_a[c], I_ch[c], tem);
		q += tem;
	}
	return q;
}

// cum sum like cumsum in matlab
Mat StereoMatching::CumSum(const Mat& src, const int d)
{
	int H = src.rows;
	int W = src.cols;
	Mat dest = Mat::zeros(H, W, src.type());

	if (d == 1) {
		// summation over column
		for (int y = 0; y < H; y++) {
			float* curData = (float*)dest.ptr<float>(y);
			float* preData = (float*)dest.ptr<float>(y);
			if (y) {
				// not first row
				preData = (float*)dest.ptr<float>(y - 1);
			}
			float* srcData = (float*)src.ptr<float>(y);
			for (int x = 0; x < W; x++) {
				curData[x] = preData[x] + srcData[x];
			}
		}
	}
	else {
		// summation over row
		for (int y = 0; y < H; y++) {
			float* curData = (float*)dest.ptr<float>(y);
			float* srcData = (float*)src.ptr<float>(y);
			for (int x = 0; x < W; x++) {
				if (x) {
					curData[x] = curData[x - 1] + srcData[x];
				}
				else {
					curData[x] = srcData[x];
				}
			}
		}
	}
	return dest;
}
//  %   BOXFILTER   O(1) time box filtering using cumulative sum
//	%
//	%   - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
//  %   - Running time independent of r; 
//  %   - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
//  %   - But much faster.
Mat StereoMatching::BoxFilter(const Mat& imSrc, const int r)
{
	int H = imSrc.rows;
	int W = imSrc.cols;
	// image size must large than filter size
	CV_Assert(W >= r && H >= r);
	Mat imDst = Mat::zeros(H, W, imSrc.type());
	// cumulative sum over Y axis
	Mat imCum = CumSum(imSrc, 1);
	// difference along Y ( [ 0, r ], [r + 1, H - r - 1], [ H - r, H ] )
	for (int y = 0; y < r + 1; y++) {
		float* dstData = (float*)imDst.ptr<float>(y);
		float* plusData = (float*)imCum.ptr<float>(y + r);
		for (int x = 0; x < W; x++) {
			dstData[x] = plusData[x];
		}

	}
	for (int y = r + 1; y < H - r; y++) {
		float* dstData = (float*)imDst.ptr<float>(y);
		float* minusData = (float*)imCum.ptr<float>(y - r - 1);
		float* plusData = (float*)imCum.ptr<float>(y + r);
		for (int x = 0; x < W; x++) {
			dstData[x] = plusData[x] - minusData[x];
		}
	}
	for (int y = H - r; y < H; y++) {
		float* dstData = (float*)imDst.ptr<float>(y);
		float* minusData = (float*)imCum.ptr<float>(y - r - 1);
		float* plusData = (float*)imCum.ptr<float>(H - 1);
		for (int x = 0; x < W; x++) {
			dstData[x] = plusData[x] - minusData[x];
		}
	}

	// cumulative sum over X axis
	imCum = CumSum(imDst, 2);
	for (int y = 0; y < H; y++) {
		float* dstData = (float*)imDst.ptr<float>(y);
		float* cumData = (float*)imCum.ptr<float>(y);
		for (int x = 0; x < r + 1; x++) {
			dstData[x] = cumData[x + r];
		}
		for (int x = r + 1; x < W - r; x++) {
			dstData[x] = cumData[x + r] - cumData[x - r - 1];
		}
		for (int x = W - r; x < W; x++) {
			dstData[x] = cumData[W - 1] - cumData[x - r - 1];
		}
	}
	return imDst;
}
//  %   GUIDEDFILTER   O(1) time implementation of guided filter.
//	%
//	%   - guidance image: I (should be a gray-scale/single channel image)
//	%   - filtering input image: p (should be a gray-scale/single channel image)
//	%   - local window radius: r
//	%   - regularization parameter: eps

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
	{
		if (param_.adCensus_useAdpWgt)
		{
			if (!param_.has_initArm)
				initArm();
			if (!param_.has_calArms)
				calArms<uchar>(I_c, HVL, HVL_INTERSECTION, param_.cbca_crossL[0], param_.cbca_crossL_out[0], param_.cbca_cTresh[0], param_.cbca_cTresh_out[0]);
			gen_vm_from2vm_expadpWgt(HVL[i], vm[i], adVm[i], cenVm[i], 10, 30, i);
		}
		else
		{
			//addWeighted(adVm[i], 0.45, cenVm[i], 0.55, 0, vm[i]);
			//xxxxx
			//gen_vm_from2vm_add(vm[i], adVm[i], cenVm[i], i);
			gen_vm_from2vm_exp(vm[i], adVm[i], cenVm[i], 10, 30, i); // lamAD 10 5 3.5, lamC 30 10 0.3
		}
	}
#ifdef DEBUG
	saveFromVm(vm, "adCensus");
#endif // DEBUG
	cout << "adCensus vm generated" << endl;
}

void StereoMatching::adCensuGradCombine(vector<Mat>& adVm, vector<Mat>& cenVm, vector<Mat>& gradVm)
{
	const int n = param_.numDisparities;
	Mat dispMap(h_, w_, CV_16S);
	int imgNum = Do_LRConsis ? 2 : 1;
	for (int i = 0; i < imgNum; i++)
	{
		//gen_vm_from3vm_exp(vm[i], adVm[i], cenVm[i], gradVm[i], 5, 10, 2, i);
		gen_vm_from3vm_exp(vm[i], adVm[i], cenVm[i], gradVm[i], 30, 45, 15, i);
		//gen_vm_from3vm_add(vm[i], adVm[i], cenVm[i], gradVm[i], i);
	}
#ifdef DEBUG
	saveFromVm(vm, "adCensusGrad");
#endif // DEBUG
	cout << "adCensusGrad vm generated" << endl;
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
template <typename T>
void StereoMatching::calArms(vector<Mat>& I, vector<Mat>& cross, vector<Mat>& cross_intersec, int L, int L_out, int cTresh, int cTresh_out)
{
	const uchar minL = param_.cbca_minArmL;  // 1

	for (int num = 0; num < 2; num++)
	{
		cout << "start cal horVerArm for img " << num << endl;
		//if (param_.cbca_use_adpArm)
			//calHorVerDis2(num, channels, L / param_.disSc, L_out / param_.disSc, cTresh, cTresh_out, minL);
			//calHorVerDis2(num, channels, L / param_.disSc, L_out / param_.disSc, cTresh, cTresh_out, minL);
		//else
			//calHorVerDis(num, channels, L / param_.disSc, L_out / param_.disSc, cTresh, cTresh_out, minL);
		{
			int scale = 1;
			if (param_.disSc > 1)
				scale = param_.disSc;
			calHorVerDis<T>(I[num], cross[num], L / scale, L_out / scale , cTresh, cTresh_out, minL);
			//calHorVerDis<T>(I[num], cross[num], L / scale, cTresh, minL);
		}

			
		//if (num == 0 && object == "teddy")
		//{
		//	int v[] = {28, 58, 68, 371, 361, 370, 27, 65, 98, 150, 204, 299};
		//	int u[] = {381, 310, 308, 76, 162, 225, 177, 316, 363, 345, 312, 198};
		//	drawArmForPoint(HVL[0], v, u, 12);
		//}

	
		cout << "finish cal horVerArm for img " << num << endl;
	}
	// 生成左右图每个点在每个视差下的水平、竖直轴（通过交运算），存成一个h*w*n*4的mat
	if (param_.cbca_intersect)
		genTrueHorVerArms(cross, cross_intersec);

	param_.has_calArms = 1;
	cout << "HVL generated" << endl;
}

template <typename T>
void StereoMatching::calArms(vector<Mat>& I, vector<Mat>& cross, vector<Mat>& cross_intersec, int L, int cTresh)
{
	const uchar minL = param_.cbca_minArmL;  // 1

	for (int num = 0; num < HVL_num; num++)
	{
		cout << "start cal horVerArm for img " << num << endl;
			//calHorVerDis(num, channels, L / param_.disSc, L_out / param_.disSc, cTresh, cTresh_out, minL);
		int scale = 1;
		if (param_.disSc > 1)
			scale = param_.disSc;
		calHorVerDis<T>(I[num], cross[num], L / scale, cTresh / scale, minL);


		//if (num == 0 && object == "teddy")
		//{
			//int v[] = {28, 58, 68, 371, 361, 370, 27, 65, 98, 150, 204, 299, 237, 145, 37, 44};
			//int u[] = {381, 310, 308, 76, 162, 225, 177, 316, 363, 345, 312, 198, 246, 310, 361, 386};
			//drawArmForPoint(HVL[0], v, u, 16);
		//}


		cout << "finish cal horVerArm for img " << num << endl;
	}
	// 生成左右图每个点在每个视差下的水平、竖直轴（通过交运算），存成一个h*w*n*4的mat
	if (param_.cbca_intersect)
		genTrueHorVerArms(cross, cross_intersec);

	param_.has_calArms = 1;
	cout << "HVL generated" << endl;
}

template <typename T>
void StereoMatching::calArms(vector<Mat>& I, vector<Mat>& cross, vector<Mat>& cross_intersec, int L0, int L1, int L2, int tresh0, int tresh1, int tresh2)
{
	const uchar minL = param_.cbca_minArmL;  // 1

	for (int num = 0; num < 2; num++)
	{
		cout << "start cal horVerArm for img " << num << endl;

		{
			int scale = 1;
			if (param_.disSc > 1)
				scale = param_.disSc;
			calHorVerDis<T>(I[num], cross[num], L0 / scale, L1 / scale, L2 / scale, tresh0, tresh1, tresh2, minL);
		}


		cout << "finish cal horVerArm for img " << num << endl;
	}
	// 生成左右图每个点在每个视差下的水平、竖直轴（通过交运算），存成一个h*w*n*4的mat
	if (param_.cbca_intersect)
		genTrueHorVerArms(cross, cross_intersec);

	param_.has_calArms = 1;
	cout << "HVL generated" << endl;
}

template <typename T>
void StereoMatching::calArms(vector<Mat>& I, vector<Mat>& cross, vector<Mat>& cross_intersec, vector<int> L, vector<int> thres)
{
	const uchar minL = param_.cbca_minArmL;  // 1
	for (int n = 0; n < L.size(); n++)
	{
		L[n] /=param_.disSc;
	}

	for (int num = 0; num < 2; num++)
		calHorVerDis<T>(I[num], cross[num], L, thres, minL);

	// 生成左右图每个点在每个视差下的水平、竖直轴（通过交运算），存成一个h*w*n*4的mat
	if (param_.cbca_intersect)
		genTrueHorVerArms(cross, cross_intersec);

	param_.has_calArms = 1;
	cout << "HVL generated" << endl;
}


void StereoMatching::showArms(int v, int u) 
{
	Mat out = I_c[0].clone();
	ushort* armP = HVL[0].ptr<ushort>(v, u);
	int left = armP[0];
	int right = armP[1];
	int up = armP[2];
	int down = armP[3];

	for (int du = -left; du <= right; du++)
	{
		uchar* hp = out.ptr<uchar>(v, u + du);
		hp[0] = 15;
		hp[1] = 55;
		hp[2] = 105;
	}
	for (int dv = -up; dv <= down; dv++)
	{
		uchar* vp = out.ptr<uchar>(v + dv, u);
		vp[0] = 15;
		vp[1] = 55;
		vp[2] = 105;
	}
	string name = "arm_y";
	imwrite(param_.savePath + name + ".png", out);

	string addr = param_.savePath + name + ".txt";
	ofstream fout;
	fout.open(addr, ios::app | ios::out);
	if (fout.is_open())
	{
		fout << "v: " << v << ", u:" << u << "\n";

		fout << "l:" << left << "\t";
		fout << "r:" << right << "\t";
		fout << "u:" << up << "\t";
		fout << "d:" << down << "\t";
		fout << "\n";
		fout.close();
	}
	else
	{
		cout << "can't write HVL to txt" << endl;
		exit(0);
	}
}

void StereoMatching::calTileArms()
{
	int channels = cbca_genArm_isColor ? 3 : 1;
	const uchar minL = param_.cbca_minArmL;  // 1

	for (int num = 0; num < TileL_num; num++)
	{
		cout << "start cal horVerArm for img " << num << endl;
		calTileDis(num, channels, param_.cbca_crossL[0], param_.cbca_crossL_out[0], param_.cbca_cTresh[0], param_.cbca_cTresh_out[0], minL); // yyyyyy
		//if (num == 0 && object == "teddy")
		//{
		//	int v[] = { 28, 58, 68, 371, 361, 370, 27, 65, 98, 150, 204, 299 };
		//	int u[] = { 381, 310, 308, 76, 162, 225, 177, 316, 363, 345, 312, 198 };
		//	drawArmForPoint(tileCrossL[0], v, u, 12);
		//}

		// 生成左右图每个点在每个视差下的水平、竖直轴（通过交运算），存成一个h*w*n*4的mat
		//if (param_.cbca_intersect)
		//	genTrueHorVerArms();

		cout << "finish cal TileArm for img " << num << endl;
	}
	param_.has_calTileArms = 1;
	cout << "TileHVL generated" << endl;
}

void StereoMatching::initArm()
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
	param_.has_initArm = 1;
}

void StereoMatching::initTileArm()
{
	TileL_num = 2; // 分别表示左图和右图
	tileCrossL.resize(TileL_num);
	//if (param_.cbca_intersect)  // 这个后面再做处理（因为涉及到后面一些代码）
	tile_INTERSECTION.resize(TileL_num);

	int Tile_size[] = { h_, w_, 5 };
	int Tile_IS_size[] = { h_, w_, param_.numDisparities, 5 };
	for (int i = 0; i < HVL_num; i++)
	{
		tileCrossL[i].create(3, Tile_size, CV_16U);
		//if (param_.cbca_intersect)
		tile_INTERSECTION[i].create(4, Tile_IS_size, CV_16U);
	}
	param_.has_initTileArm = 1;
	/////
}

void StereoMatching::cbca_core( vector<Mat>& HVL, vector<Mat>& HVL_INTERSECTION, vector<Mat>& vm, int ITNUM)
{
	cout << "\n" << endl;
	cout << "start CBCA aggregation" << endl;
	clock_t start = clock();

	int du[2] = { -1, 0 }, dv[2] = { 0, -1 };  // 表示前面一点的位置
	int imgNum = Do_refine && Do_LRConsis ? 2 : 1;

	Mat dispMap(h_, w_, CV_16S);
	Mat dispMap2(h_, w_, CV_16S);
	Mat result(h_, w_, CV_16S);

	Mat area, areaIS, vm_copy;
	for (int LOR = 0; LOR < imgNum; LOR++)
	{
		for (int agItNum = 0; agItNum < ITNUM; agItNum++)
		{
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

			clock_t end_inner = clock();
			clock_t time_inner = end_inner - start_inner;
			cout << "time: " << time_inner << endl;
#ifdef DEBUG
			if (LOR == 0)
				saveTime(time_inner, "cbciInner0");
#endif // DEBUG
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
					selectTopCostFromVolumn(vm_copy, topDisp, param_.vmTop_thres);
					signCorrectFromTopVm("correctFromTopVmCBCA" + to_string(agItNum) + ".png", topDisp, DT);
					//if (object == "teddy")
						//genExcelFromTopDisp(topDisp, DT);
					//genDispFromTopCostVm(topDisp, dispMap2);
					genDispFromTopCostVm2(topDisp, dispMap2);
					signDispChange_for2Disp(dispMap, dispMap2, DT, I_mask[1], result);
					saveDispMap<short>(result, DT, "candidate_Change" + to_string(agItNum));
					saveFromDisp<short>(dispMap2, "cbcatopVm" + to_string(agItNum));
				}
			}
#endif // DEBUG
		}
	}
	clock_t end = clock();
	clock_t time = end - start;
	cout << "1cbcaTime: " << time << endl;
	cout << time << endl;
	cout << "CBCATime: " << time << endl;
	cout << "finish CBCA aggregation" << endl;
#ifdef DEBUG
	saveTime(time, "CBCA aggre");
#endif // DEBUG
}

void StereoMatching::cbca_aggregate(int param_Num, vector<Mat>& vm)
{
	cout << "start CBCA aggregation" << endl;
	clock_t start = clock();

	// 计算臂长
	clock_t armStart = clock();
	if (!param_.has_initArm)
		initArm();
	if (!param_.has_calArms)
		calArms<uchar>(I_c, HVL, HVL_INTERSECTION, param_.cbca_crossL[param_Num], param_.cbca_crossL_out[param_Num], param_.cbca_cTresh[param_Num], param_.cbca_cTresh_out[param_Num]);
		//calArms<uchar>(I_c, HVL, HVL_INTERSECTION, 20, 35);
	
	cbca_core(HVL, HVL_INTERSECTION, vm, param_.cbca_iterationNum);

	clock_t end = clock();
	clock_t time = end - start;
	cout << "cbcaTime: " << time << endl;
	cout << "finish CBCA aggregation" << endl;
#ifdef DEBUG
	saveTime(time, "CBCA");
#endif // DEBUG
}

void StereoMatching::AWS()
{
#ifdef JBF_STANDARD
	int num = Do_refine && Do_LRConsis ? 2 : 1;
	Mat I_tem;
	for (int i = 0; i < 2; i++)
	{
		I_c[i].convertTo(I_tem, CV_32F);
		vector<Mat> channels;
		split(vm[i], channels);
		for (int d = 0; d < d_; d++)
		{
			cv::ximgproc::jointBilateralFilter(I_tem, channels[d], channels[d], 35, 5, 17.5);
			cout << "JBF--";
		}
		cout << endl;
		merge(channels, vm[i]);
	}

#else
	const auto t1 = std::chrono::system_clock::now();
	std::cout << "AWS start" << endl;
	// 计算内部区域代价
	const int W_U_AWS = 17, W_V_AWS = 17;
	const int W_S = (W_U_AWS * 2 + 1) * (W_V_AWS * 2 + 1);
	//const int HI_L = W_U_AWS;  // 左右边缘往外拓展的长度
	int size_wt[] = { h_, w_, W_S };
	Mat wt[2], Lab[2], vm_IB[2];
	int size_vmIB[] = { h_ + W_V_AWS * 2, w_ + W_U_AWS * 2, param_.numDisparities };
	int loopNum = Do_LRConsis ? 2 : 1;
	//for (int i = 0; i < loopNum; i++)
	//{
	//	vmTem[i].create(3, size_vm, CV_32F);
	//	initializeMat<float>(vmTem[i], W_S * 3);
	//	vm_IB[i].create(3, size_vmIB, CV_32F);
	//}
	for (int i = 0; i < 2; i++)
	{
		wt[i].create(3, size_wt, CV_32F);
		//copyMakeBorder(I_c[i], I_IpolBorder[i], W_V_AWS, W_V_AWS, 0, 0, BORDER_REFLECT_101);
		copyMakeBorder(I_c[i], Lab[i], W_V_AWS, W_V_AWS, W_U_AWS, W_U_AWS, BORDER_REFLECT_101);
		cv::cvtColor(Lab[i], Lab[i], COLOR_BGR2Lab);
	}

	genWeight_AWS(W_V_AWS, W_U_AWS, wt[0], Lab[0]);
	genWeight_AWS(W_V_AWS, W_U_AWS, wt[1], Lab[1]);
	cout << "finish adapt weight cal" << endl;
	//genTadVm<0>(I_IpolBorder, vm_IB[0]);
	//if (Do_LRConsis)
	//	genTadVm<1>(I_IpolBorder, vm_IB[1]);
	copyMakeBorder(vm[0], vm_IB[0], W_V_AWS, W_V_AWS, W_U_AWS, W_U_AWS, BORDER_REFLECT_101);
	copyMakeBorder(vm[1], vm_IB[1], W_V_AWS, W_V_AWS, W_U_AWS, W_U_AWS, BORDER_REFLECT_101);
	calvm_AWS<0>(W_V_AWS, W_U_AWS, vm[0], wt, vm_IB[0]);
	if (Do_LRConsis)
		calvm_AWS<1>(W_V_AWS, W_U_AWS, vm[1], wt, vm_IB[1]);

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
#endif // JBF_STANDARD

	
	

#ifdef DEBUG
	saveFromVm(vm, "asw");
	saveTime(duration, "asw");
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
	signDispChange_for2Disp(dispMat, dispMat2, DT, I_mask[1], result);
	saveDispMap<short>(result, DT, "BF-CBBI_Change" + to_string(imgNum));
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

	//const int rv[MAX_DIRECTIONS] = { 0, 0  };
	//const int ru[MAX_DIRECTIONS] = { +1, -1 };

	//int numOfDirec = param_.sgm_scanNum;
	int numOfDirec = 4;
	L.resize(numOfDirec);

	for (int i = 0; i < numOfDirec; i++)
	{
		L[i].create(3, size_vm);
		costScan(L[i], vm, rv[i], ru[i], leftFirst);
		cout << "scanLline " << i << "finished" << endl;
	}
	gen_sgm_vm(vm, L, numOfDirec);
}


void StereoMatching::sgm_hori(cv::Mat& vm, bool leftFirst)  // vm的数据类型由代价计算方法而定
{
	const int rv[2] = { 0, 0};  // 来向（前一个点的方向）
	const int ru[2] = {+1, -1};

	//const int rv[MAX_DIRECTIONS] = { 0, 0  };
	//const int ru[MAX_DIRECTIONS] = { +1, -1 };

	int numOfDirec = 2;
	L.resize(numOfDirec);

	Mat vm_res = vm.clone();
	for (int i = 0; i < numOfDirec; i++)
	{
		L[i].create(3, size_vm);
		costScan(L[i], vm, rv[i], ru[i], leftFirst);
		cout << "scanLline " << i << "finished" << endl;
	}
	gen_sgm_vm(vm, L, numOfDirec);
	vm = vm - vm_res;
}

void StereoMatching::sgm_verti(cv::Mat& vm, bool leftFirst)  // vm的数据类型由代价计算方法而定
{
	const int rv[2] = { +1, -1 };
	const int ru[2] = { 0, 0 };

	//const int rv[MAX_DIRECTIONS] = { 0, 0  };
	//const int ru[MAX_DIRECTIONS] = { +1, -1 };

	int numOfDirec = 2;
	L.resize(numOfDirec);
	Mat vm_res = vm.clone();

	for (int i = 0; i < numOfDirec; i++)
	{
		L[i].create(3, size_vm);
		costScan(L[i], vm, rv[i], ru[i], leftFirst);
		cout << "scanLline " << i << "finished" << endl;
	}
	gen_sgm_vm(vm, L, numOfDirec);
	vm = vm - vm_res;
}


void StereoMatching::so(Mat& vm, Mat& DP, vector<Mat>& I)
{
	Mat trace(h_, w_, CV_32SC(d_));
	vector<float> costCandi(d_, 0);

	// 前向传递
	for (int v = 0; v < h_; v++)
	{
		for (int u = 1; u < w_; u++)
		{
			
			uchar* IP = I[0].ptr<uchar>(v, u);
			uchar* IP_pre = I[0].ptr<uchar>(v, u - 1);
			float sum = 0;
			bool L_isDisc = false;
			for (int c = 0; c < 3; c++)
			{
				sum += abs(IP[c] - IP_pre[c]);
			}
			sum /= 3;
			if (sum > 15)
			{
				L_isDisc = true;
				//Pn2 /= 2;
				//Pn3 /= 2;
			}
			
			float* vmP = vm.ptr<float>(v, u);
			float* vmPre = vm.ptr<float>(v, u - 1);
			int* traP = trace.ptr<int>(v, u);
			for (int d = 0; d < d_; d++)
			{
				float Pn = 0.2;
				float Pn2 = 1.2;  // 0.3
				float Pn3 = 3.6;   // 0.9
				//bool R_isDisc = false;
				//if (u - d - 1 >= 0)
				//{
				//	uchar* IP_r = I[1].ptr<uchar>(v, u - d);
				//	uchar* IP_r_pre = I[1].ptr<uchar>(v, u - d - 1);
				//	float sum = 0;
				//	for (int c = 0; c < 3; c++)
				//	{
				//		sum += abs(IP_r[c] - IP_r_pre[c]);
				//	}
				//	sum /= 3;
				//	if (sum > 15)
				//	{
				//		R_isDisc = true;
				//		//Pn2 /= 2;
				//		//Pn3 /= 2;
				//	}
				//}
				if (L_isDisc)
				{
					Pn2 /= 2;
					Pn3 /= 2;
				}
				//if (L_isDisc && R_isDisc)
				//{
				//	Pn2 /= 10;
				//	Pn3 /= 10;
				//}
				//else if (L_isDisc || R_isDisc)
				//{
				//	Pn2 /= 4;
				//	Pn3 /= 4;
				//}
				//for (int d_pre = 0; d_pre < d_; d_pre++)
				//{
				//	costCandi[d_pre] = vmPre[d_pre] + Pn * abs(d - d_pre);
				//}
				//float cost_min = numeric_limits<float>::max();
				//float d_min = -1;
				//for (int d = 0; d < d_; d++)
				//{
				//	if (costCandi[d] < cost_min)
				//	{
				//		cost_min = costCandi[d];
				//		d_min = d;
				//	}
				//}
				float c_min = vmPre[0];
				float d_cMin = 0;
				for (int d_l = 1; d_l < d_; d_l++)
				{
					if (vmPre[d_l] < c_min)
					{
						c_min = vmPre[d_l];
						d_cMin = d_l;
					}
				}
				float c_minus = d > 0 ? vmPre[d - 1] + Pn2 : numeric_limits<float>::max();
				float c_plus = d < d_ - 1 ? vmPre[d + 1] + Pn2 : numeric_limits<float>::max();
				c_min += Pn3;
				int d_min = d;
				float cost_min = vmPre[d];
				if (c_minus < cost_min)
				{
					d_min = d - 1;
					cost_min = c_minus;
				}
				if (c_plus < cost_min)
				{
					d_min = d + 1;
					cost_min = c_plus;
				}
				if (c_min < cost_min)
				{
					d_min = d_cMin;
					cost_min = c_min;
				}

				//vmp[d] += (cost_min - c_min);
				vmP[d] += cost_min;
				traP[d] = d_min;
			}
		}
	}

	// 反向追踪
	for (int v = 0; v < h_; v++)
	{
		short* disp = DP.ptr<short>(v);
		float c_min = vm.ptr<float>(v, w_ - 1)[0];
		int d_min = 0;
		float* cP = vm.ptr<float>(v, w_ - 1);
		for (int d = 1; d < d_; d++)
		{
			if (cP[d] < c_min)
			{
				c_min = cP[d];
				d_min = d;
			}
		}
		disp[w_ - 1] = d_min;
		for (int u = w_ - 1; u > 0; u--)
		{
			int* traP = trace.ptr<int>(v, u);
			int d_pre = traP[d_min];
			disp[u - 1] = d_pre;
			d_min = d_pre;
		}
	}
}


void StereoMatching::so_change(Mat& vm, Mat& DP, vector<Mat>& I)
{
	Mat trace(h_, w_, CV_32SC(d_));
	vector<float> costCandi(d_, 0);

	// 前向传递
	for (int v = 0; v < h_; v++)
	{
		short* dP = DP.ptr<short>(v);
		for (int u = 1; u < w_; u++)
		{
			short dP_ref = dP[u - 1];
			uchar* IP = I[0].ptr<uchar>(v, u);
			uchar* IP_pre = I[0].ptr<uchar>(v, u - 1);
			float sum = 0;
			bool L_isDisc = false;
			for (int c = 0; c < 3; c++)
			{
				sum += abs(IP[c] - IP_pre[c]);
			}
			sum /= 3;
			if (sum > 15)
			{
				L_isDisc = true;
				//Pn2 /= 2;
				//Pn3 /= 2;
			}

			float* vmP = vm.ptr<float>(v, u);
			float* vmPre = vm.ptr<float>(v, u - 1);
			int* traP = trace.ptr<int>(v, u);
			for (int d = 0; d < d_; d++)
			{
				float Pn = 0.2;
				float Pn2 = 1.2;  // 0.3
				float Pn3 = 3.6;   // 0.9
				float Pn2_ = 2;
				//bool R_isDisc = false;
				//if (u - d - 1 >= 0)
				//{
				//	uchar* IP_r = I[1].ptr<uchar>(v, u - d);
				//	uchar* IP_r_pre = I[1].ptr<uchar>(v, u - d - 1);
				//	float sum = 0;
				//	for (int c = 0; c < 3; c++)
				//	{
				//		sum += abs(IP_r[c] - IP_r_pre[c]);
				//	}
				//	sum /= 3;
				//	if (sum > 15)
				//	{
				//		R_isDisc = true;
				//		//Pn2 /= 2;
				//		//Pn3 /= 2;
				//	}
				//}
				if (L_isDisc)
				{
					Pn2 /= 2;
					Pn3 /= 2;
					Pn2_ / 2;
				}
				//if (L_isDisc && R_isDisc)
				//{
				//	Pn2 /= 10;
				//	Pn3 /= 10;
				//}
				//else if (L_isDisc || R_isDisc)
				//{
				//	Pn2 /= 4;
				//	Pn3 /= 4;
				//}
				//for (int d_pre = 0; d_pre < d_; d_pre++)
				//{
				//	costCandi[d_pre] = vmPre[d_pre] + Pn * abs(d - d_pre);
				//}
				//float cost_min = numeric_limits<float>::max();
				//float d_min = -1;
				//for (int d = 0; d < d_; d++)
				//{
				//	if (costCandi[d] < cost_min)
				//	{
				//		cost_min = costCandi[d];
				//		d_min = d;
				//	}
				//}
				//float c_min = vmPre[0];
				//float d_cMin = 0;
				//for (int d_l = 1; d_l < d_; d_l++)
				//{
				//	if (vmPre[d_l] < c_min)
				//	{
				//		c_min = vmPre[d_l];
				//		d_cMin = d_l;
				//	}
				//}
				float c_min = vmPre[dP_ref] + Pn3;
				float c_minus = d > 0 ? vmPre[d - 1] + Pn2 : numeric_limits<float>::max();
				float c_plus = d < d_ - 1 ? vmPre[d + 1] + Pn2 : numeric_limits<float>::max();
				float c_minus_ = d - 2 >= 0 ? vmPre[d - 2] + Pn2_ : numeric_limits<float>::max();
				float c_plus_ = d + 2 < d_ ? vmPre[d + 2] + Pn2_ : numeric_limits<float>::max();
				//c_min += Pn3;
				int d_min = d;
				float cost_min = vmPre[d];
				if (c_minus < cost_min)
				{
					d_min = d - 1;
					cost_min = c_minus;
				}
				if (c_plus < cost_min)
				{
					d_min = d + 1;
					cost_min = c_plus;
				}
				if (c_min < cost_min)
				{
					d_min = dP_ref;
					cost_min = c_min;
				}
				if (c_minus_ < cost_min)
				{
					d_min = d - 2;
					cost_min = c_minus_;
				}
				if (c_plus_ < cost_min)
				{
					d_min = d + 2;
					cost_min = c_plus_;
				}
				//vmP[d] += (cost_min - c_min);
				vmP[d] += cost_min;
				traP[d] = d_min;
			}
		}
	}

	// 反向追踪
	for (int v = 0; v < h_; v++)
	{
		short* disp = DP.ptr<short>(v);
		float c_min = vm.ptr<float>(v, w_ - 1)[0];
		int d_min = 0;
		float* cP = vm.ptr<float>(v, w_ - 1);
		for (int d = 1; d < d_; d++)
		{
			if (cP[d] < c_min)
			{
				c_min = cP[d];
				d_min = d;
			}
		}
		disp[w_ - 1] = d_min;
		for (int u = w_ - 1; u > 0; u--)
		{
			int* traP = trace.ptr<int>(v, u);
			int d_pre = traP[d_min];
			disp[u - 1] = d_pre;
			d_min = d_pre;
		}
	}
}

void StereoMatching::so_T2D(Mat& vm, Mat& DP, vector<Mat>& I)
{
	Mat trace(h_, w_, CV_32SC(d_));
	vector<float> costCandi(d_, 0);

	// 前向传递
	for (int u = 0; u < w_; u++)
	{
		for (int v = 1; v < h_; v++)
		{
			uchar* IP = I[0].ptr<uchar>(v, u);
			uchar* IP_pre = I[0].ptr<uchar>(v - 1, u);
			float sum = 0;
			bool L_isDisc = false;
			for (int c = 0; c < 3; c++)
			{
				sum += abs(IP[c] - IP_pre[c]);
			}
			sum /= 3;
			if (sum > 15)
				L_isDisc = true;
			float* vmP = vm.ptr<float>(v, u);
			float* vmPre = vm.ptr<float>(v - 1, u);
			int* traP = trace.ptr<int>(v, u);
			for (int d = 0; d < d_; d++)
			{
				float Pn = 0.2;
				float Pn2 = 0.3;  // 0.3
				float Pn3 = 0.9;   // 0.9
				if (L_isDisc)
				{
					Pn2 /= 2;
					Pn3 /= 2;
				}
				float c_min = vmPre[0];
				float d_cMin = 0;
				for (int d_l = 1; d_l < d_; d_l++)
				{
					if (vmPre[d_l] < c_min)
					{
						c_min = vmPre[d_l];
						d_cMin = d_l;
					}
				}
				float c_minus = d > 0 ? vmPre[d - 1] + Pn2 : numeric_limits<float>::max();
				float c_plus = d < d_ - 1 ? vmPre[d + 1] + Pn2 : numeric_limits<float>::max();
				c_min += Pn3;
				int d_min = d;
				float cost_min = vmPre[d];
				if (c_minus < cost_min)
				{
					d_min = d - 1;
					cost_min = c_minus;
				}
				if (c_plus < cost_min)
				{
					d_min = d + 1;
					cost_min = c_plus;
				}
				if (c_min < cost_min)
				{
					d_min = d_cMin;
					cost_min = c_min;
				}
				vmP[d] += (cost_min - c_min);
				//vmP[d] += cost_min;
				traP[d] = d_min;
			}
		}

		// 反向追踪
		short* disp = DP.ptr<short>(h_ - 1);
		for (int u = 0; u < w_; u++)
		{
			float* vP = vm.ptr<float>(h_ - 1, u);
			float c_min = vP[0];
			int d_min = 0;
			for (int d = 1; d < d_; d++)
			{
				if (c_min > vP[d])
				{
					c_min = vP[d];
					d_min = d;
				}
			}
			disp[u] = d_min;
		}
		for (int v = h_ - 1; v >= 1; v--)
		{
			short* disp = DP.ptr<short>(v);
			short* disp_pre = DP.ptr<short>(v - 1);

			for (int u = 0; u < w_; u++)
			{
				short d = disp[u];
				int* tP = trace.ptr<int>(v, u);
				int d_pre = tP[d];
				disp_pre[u] = d_pre;
			}
		}
	}
}

void StereoMatching::so_R2L(Mat& vm, Mat& DP, vector<Mat>& I)
{
	Mat trace(h_, w_, CV_32SC(d_));
	vector<float> costCandi(d_, 0);

	// 前向传递
	for (int v = 0; v < h_; v++)
	{
		for (int u = w_ - 2; u >= 0; u--)
		{

			uchar* IP = I[0].ptr<uchar>(v, u);
			uchar* IP_pre = I[0].ptr<uchar>(v, u + 1);
			float sum = 0;
			bool L_isDisc = false;
			for (int c = 0; c < 3; c++)
			{
				sum += abs(IP[c] - IP_pre[c]);
			}
			sum /= 3;
			if (sum > 15)
			{
				L_isDisc = true;
				//Pn2 /= 2;
				//Pn3 /= 2;
			}

			float* vmP = vm.ptr<float>(v, u);
			float* vmPre = vm.ptr<float>(v, u + 1);
			int* traP = trace.ptr<int>(v, u);
			for (int d = 0; d < d_; d++)
			{
				float Pn = 0.2;
				float Pn2 = 0.3;  // 0.3
				float Pn3 = 0.9;   // 0.9
				//bool R_isDisc = false;
				//if (u - d - 1 >= 0)
				//{
				//	uchar* IP_r = I[1].ptr<uchar>(v, u - d);
				//	uchar* IP_r_pre = I[1].ptr<uchar>(v, u - d - 1);
				//	float sum = 0;
				//	for (int c = 0; c < 3; c++)
				//	{
				//		sum += abs(IP_r[c] - IP_r_pre[c]);
				//	}
				//	sum /= 3;
				//	if (sum > 15)
				//	{
				//		R_isDisc = true;
				//		//Pn2 /= 2;
				//		//Pn3 /= 2;
				//	}
				//}
				if (L_isDisc)
				{
					Pn2 /= 2;
					Pn3 /= 2;
				}
				//if (L_isDisc && R_isDisc)
				//{
				//	Pn2 /= 10;
				//	Pn3 /= 10;
				//}
				//else if (L_isDisc || R_isDisc)
				//{
				//	Pn2 /= 4;
				//	Pn3 /= 4;
				//}
				//for (int d_pre = 0; d_pre < d_; d_pre++)
				//{
				//	costCandi[d_pre] = vmPre[d_pre] + Pn * abs(d - d_pre);
				//}
				//float cost_min = numeric_limits<float>::max();
				//float d_min = -1;
				//for (int d = 0; d < d_; d++)
				//{
				//	if (costCandi[d] < cost_min)
				//	{
				//		cost_min = costCandi[d];
				//		d_min = d;
				//	}
				//}
				float c_min = vmPre[0];
				float d_cMin = 0;
				for (int d_l = 1; d_l < d_; d_l++)
				{
					if (vmPre[d_l] < c_min)
					{
						c_min = vmPre[d_l];
						d_cMin = d_l;
					}
				}
				float c_minus = d > 0 ? vmPre[d - 1] + Pn2 : numeric_limits<float>::max();
				float c_plus = d < d_ - 1 ? vmPre[d + 1] + Pn2 : numeric_limits<float>::max();
				c_min += Pn3;
				int d_min = d;
				float cost_min = vmPre[d];
				if (c_minus < cost_min)
				{
					d_min = d - 1;
					cost_min = c_minus;
				}
				if (c_plus < cost_min)
				{
					d_min = d + 1;
					cost_min = c_plus;
				}
				if (c_min < cost_min)
				{
					d_min = d_cMin;
					cost_min = c_min;
				}
				//vmP[d] += (cost_min - c_min);
				vmP[d] += cost_min;
				traP[d] = d_min;
			}
		}
	}

	// 反向追踪
	for (int v = 0; v < h_; v++)
	{
		short* disp = DP.ptr<short>(v);
		float c_min = vm.ptr<float>(v, 0)[0];
		int d_min = 0;
		float* cP = vm.ptr<float>(v, 0);
		for (int d = 1; d < d_; d++)
		{
			if (cP[d] < c_min)
			{
				c_min = cP[d];
				d_min = d;
			}
		}
		disp[0] = d_min;
		for (int u = 0; u < w_ - 1; u++)
		{
			int* traP = trace.ptr<int>(v, u);
			int d_pre = traP[d_min];
			disp[u + 1] = d_pre;
			d_min = d_pre;
		}
	}
}

//void StereoMatching::so_two

int StereoMatching::cal_histogram_for_HV(Mat& dispImg, int v_ancher, int u_ancher, int numThres, float ratioThre, int LOR = 0)
{
	int n = param_.numDisparities;
	vector<int> hist(n, 0);
	int validNum = 0;
	int v_begin = v_ancher - HVL[LOR].ptr<ushort>(v_ancher, u_ancher)[2];
	int v_end = v_ancher + HVL[LOR].ptr<ushort>(v_ancher, u_ancher)[3];
	for (int v = v_begin; v <= v_end; v++)
	{
		int u_begin = u_ancher - HVL[LOR].ptr<ushort>(v, u_ancher)[0];
		int u_end = u_ancher + HVL[LOR].ptr<ushort>(v, u_ancher)[1];
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
	//if (param_.cbca_armHV && param_.cbca_armTile)
	//{
	//	if (param_.regVote_type == 2)
	//	{
	//		int result = compareArmL(v, u);
	//		dp_ = result == 0 ? cal_histogram_for_HV(Dp, v, u, SThres, hratioThres) : cal_histogram_for_Tile(Dp, v, u, SThres, hratioThres);
	//	}
	//	else if (param_.regVote_type == 1)
	//		dp_ = cal_histogram_for_Tile(Dp, v, u, SThres, hratioThres);
	//	else
	//		dp_ = cal_histogram_for_HV(Dp, v, u, SThres, hratioThres);

	//}
	//if (param_.cbca_armHV)
		dp_ = cal_histogram_for_HV(Dp, v, u, SThres, hratioThres);
	//else
	//	dp_ = cal_histogram_for_Tile(Dp, v, u, SThres, hratioThres);
	return dp_;
}

//为视差图中每个点执行区域投票并选择票数最多的点替换目标点
void StereoMatching::regionVoteForWholeDispImg(Mat& Dp)
{
	for (int v = 0; v < h_; v++)
	{
		short* dP = Dp.ptr<short>(v);
		for (int u = 0; u < w_; u++)
		{
			dP[u] = cal_histogram_for_HV(Dp, v, u, 0, 0);
		}
	}
}
/*
 *@searchDepth 表示往左搜寻的深度
 *@numThres 表示找到几个相同的值就不再寻找
 */
void StereoMatching::backgroundInterpolateCore(Mat& Dp, int v, int u, int* result)
{
	vector<int> vec(2, -1);
	const int searchDepth = param_.bgIplDepth;
	//const int searchDepth = 15;

	int coeffi = 1;
	for (int dir = 0; dir < 2; dir++)
	{ 
		short* dipP = Dp.ptr<short>(v);
		for (int d = 1; d <= searchDepth; d++)
		{
			int u_nei = u + d * coeffi;
			if (u_nei >= 0 && u_nei < w_)
			{
				if (dipP[u_nei] >= 0)
				{
					vec[dir] = dipP[u_nei];
					break;
				}
			}
			else
				break;
		}
		coeffi *= -1;
	}

	int candi_num = 0;
	int disp = 10000;
	for (int i = 0; i < 2; i++)
	{
		if (vec[i] >= 0)
			candi_num++;
		if (vec[i] >= 0 && vec[i] < disp)
			disp = vec[i];
	}
	*result = candi_num;
	result++;
	*result = disp == 10000 ? -1 : disp;

	//int res_num = vec[0] <= vec[1] && vec[0] >= 0 ? 0 : 1;
	//if (vec[res_num] < 0)
	//	res_num = res_num == 0 ? 1 : 0;
	//return vec[res_num];
}

// 传统的方法
int StereoMatching::backgroundInterpolateCore(Mat& Dp, int v, int u)
{
	vector<int> vec(2, -1);
	const int searchDepth = param_.bgIplDepth;
	//const int searchDepth = 15;

	int coeffi = 1;
	for (int dir = 0; dir < 2; dir++)
	{
		short* dipP = Dp.ptr<short>(v);
		for (int d = 1; d <= searchDepth; d++)
		{
			int u_nei = u + d * coeffi;
			if (u_nei >= 0 && u_nei < w_)
			{
				if (dipP[u_nei] >= 0)
				{
					vec[dir] = dipP[u_nei];
					break;
				}
			}
			else
				break;
		}
		coeffi *= -1;
	}
	if (vec[0] != -1 && vec[1] == -1)
		return vec[0];
	else if (vec[0] == -1 && vec[1] != -1)
		return vec[1];
	else
		return vec[0] < vec[1] ? vec[0] : vec[1];
}

int StereoMatching::backgroundInterpolateCore_(Mat& Dp, int v, int u)
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
					{ 
						dp_ = regionVoteCore(Dp, v, u, SThres, hratioThres);
						//dp_ = cal_histogram_for_HV(Dp, v, u, SThres, hratioThres);
					}
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
							int bgIpol[2] = { 0, -1 };
							backgroundInterpolateCore(Dp, v, u, bgIpol);
							dp_rv = regionVoteCore(Dp, v, u, SThres, hratioThres);
							//if (bgIpol[0] == 2)
							//{
								int dp_bg = bgIpol[1];
								if (dp_bg >= 0 && dp_rv < 0)
									dp_ = dp_bg;
								else if (dp_bg < 0 && dp_rv >= 0)
									dp_ = dp_rv;
								else if (dp_bg >= 0 && dp_rv >= 0)
									dp_ = dp_rv <= dp_bg ? dp_rv : dp_bg;  // dp_rv为bg_rv1, dp_bg为bg_rv2
								else
									dp_ = -1;
							//}
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


void StereoMatching::regionVote_my(cv::Mat& Dp, float rv_ratio, int rv_s)
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
				int n = param_.numDisparities;
				vector<int> hist(n, 0);
				int validNum = 0;
				int v_begin = v - HVL[0].ptr<ushort>(v, u)[2];
				int v_end = v + HVL[0].ptr<ushort>(v, u)[3];
				for (int v_n = v_begin; v_n <= v_end; v_n++)
				{
					int u_begin = u - HVL[0].ptr<ushort>(v_n, u)[0];
					int u_end = u + HVL[0].ptr<ushort>(v_n, u)[1];
					short* dispP = Dp.ptr<short>(v_n);
					for (int u_n = u_begin; u_n <= u_end; u_n++)
					{
						if (dispP[u_n] >= 0)
						{
							validNum++;
							hist[dispP[u_n]]++;
						}
					}
				}
				if (validNum <= rv_s)
					continue;

				int dispMost = 0;
				for (int d = 1; d < n; d++)
				{
					if (hist[d] > hist[dispMost])
						dispMost = d;
				}
				if (hist[dispMost] / validNum >= rv_ratio)
					dp_resP[u] = dispMost;
			}
		}
	}
	dp_res.copyTo(Dp);
	cout << "finish RV_combine_BG" << endl;
}


void StereoMatching::regionVote(cv::Mat& Dp, cv::Mat& cross)
{
	CV_Assert(Dp.type() == CV_16S);
	const int n = param_.numDisparities;

	//OMP_PARALLEL_FOR  加上后误差增大，原因未知
	Mat dp_res = Dp.clone();

	for (int v = 0; v < h_; v++)
	{
		short* dp_resP = dp_res.ptr<short>(v);
		for (int u = 0; u < w_; u++)
		{
			int n = param_.numDisparities;
			vector<int> hist(n, 0);
			int v_begin = v - cross.ptr<ushort>(v, u)[2];
			int v_end = v + cross.ptr<ushort>(v, u)[3];
			for (int v_n = v_begin; v_n <= v_end; ++v_n)
			{
				int u_begin = u - cross.ptr<ushort>(v_n, u)[0];
				int u_end = u + cross.ptr<ushort>(v_n, u)[1];
				short* dispP = Dp.ptr<short>(v_n);
				for (int u_n = u_begin; u_n <= u_end; ++u_n)
				{
					short cache = dispP[u_n];
					if (cache >= 0)
					{
						hist[cache]++;
					}
				}
			}
			int dispMost = 0;
			for (int d = 1; d < n; d++)
			{
				if (hist[d] > hist[dispMost])
					dispMost = d;
			}
			dp_resP[u] = dispMost;
		}
	}
	dp_res.copyTo(Dp);
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

void StereoMatching::WM(Mat& disp, Mat& mask, Mat& img)
{
	const int MED_SZ = 19;
	const int wnd_R = MED_SZ / 2;
	const float SIG_DIS = 9;
	const float SIG_CLR = 25;
	Mat disp_Bor, img_Bor;
	vector<float> dispHist(d_, 0);
	copyMakeBorder(disp, disp_Bor, wnd_R, wnd_R, wnd_R, wnd_R, BORDER_REFLECT_101);
	copyMakeBorder(img, img_Bor, wnd_R, wnd_R, wnd_R, wnd_R, BORDER_REFLECT_101);
	int num_err = 0;
	int num = 0;
	for (int v = 0; v < h_; v++)
	{
		uchar* mP = mask.ptr<uchar>(v);
		for (int u = 0; u < w_; u++)
		{
			if (mP[u] > 0)
			{
				num_err++;
				float wgtSum = 0;
				uchar* ipP = img.ptr<uchar>(v, u);
				for (int dv = -wnd_R; dv <= wnd_R; dv++)
				{
					for (int du = -wnd_R; du <= wnd_R; du++)
					{
						short q = disp_Bor.ptr<short>(v + wnd_R + dv)[u + wnd_R + du];
						uchar* iqP = img_Bor.ptr<uchar>(v + wnd_R + dv, u + wnd_R + du);
						float colDis = pow(ipP[0] - iqP[0], 2) + pow(ipP[1] - iqP[1], 2) + pow(ipP[2] - iqP[2], 2);
						float spaDis = pow(dv, 2) + pow(du, 2);
						float wgt = exp(-colDis / (SIG_CLR * SIG_CLR) - spaDis / (SIG_DIS * SIG_DIS));
						dispHist[q] += wgt;
						wgtSum += wgt;
					}
				}
				float wgt_half = wgtSum / 2;
				float wgt_cum = 0;
				for (int d = 0; d < d_; d++)
				{
					wgt_cum += dispHist[d];
					if (wgt_cum >= wgt_half)
					{
						disp.ptr<short>(v)[u] = d;
						num++;
						break;
					}
				}
				dispHist.assign(d_, 0);
			}
		}
	}
	cout << "err num: " << num_err << endl;
	cout << "WM num: " << num << endl;
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
		short* dP = Dp.ptr<short>(v);
		short* dcP = DpCopy.ptr<short>(v);
		for (int u = 0; u < w; u++)
		{
			if (dP[u] >= 0)
				dcP[u] = dP[u];
			if (dP[u] < 0)
			{
				vector<int> directionDisp(16, -1);
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
				if (dP[u] == param_.DISP_OCC)  // param.DISP_OCC = -2 * 16
				{
					int minDisp = numeric_limits<int>::max();
					int init = minDisp;
					for (int direction = 0; direction < 16; direction++)
					{
						if (directionDisp[direction] >= 0 && minDisp > directionDisp[direction])
							minDisp = directionDisp[direction];
					}
					dcP[u] = init != minDisp ? minDisp : dP[u];  // 防止没有找到的情况
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
					dcP[u] = disp >= 0 ? disp : dP[u];
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
			saveDispMap<short>(Dp, DT, "06disp_cbbi0");

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

void StereoMatching::genExcelFromTopDisp(Mat& topDisp, Mat& DT)
{
	int num = topDisp.size[2];
	ofstream opt;
	opt.open(param_.savePath + "候选视差.csv", ios::out | ios::trunc);
	opt << ",";
	for (int i = 0; i < param_.vmTop_Num; i++)
	{
		opt << to_string(i + 1) << "," << ",";
	}
	opt << "真实视差" << "," << "视差" << "," << "代价" << "," << "第几顺位(最小是0)" << "," << "和最小代价的差" << "," << "差占最小代价的百分比" << endl;
	int disp = -1;
	float cost = numeric_limits<float>::max();
	int pos = -1;
	float dif = -1;
	float ratio = -1;
	for (int v = param_.sign_v_range[0]; v < param_.sign_v_range[1]; v++)
	{
		opt << "行：" << v << endl;
		float* dtP = DT.ptr<float>(v);
		for (int u = param_.sign_u_range[0]; u < param_.sign_u_range[1]; u++)
		{
			if (dtP[u] > 0)
			{
				opt << "列：" << u << ",";
				float dtV = dtP[u];
				bool has_find = false;
				//int num_ = topDisp.ptr<float>(v, u, num - 1)[0];
				int num_ = param_.vmTop_Num;
				disp = -1;
				cost = -1;
				for (int n = 0; n < num_; n++)
				{
					float* p = topDisp.ptr<float>(v, u, n);
					opt << p[0] << "," << p[1] << ",";
					if (has_find == false)
					{
						if (abs(p[0] - dtV) <= param_.errorThreshold)
						{
							has_find = true;
							pos = n;
							disp = p[0];
							cost = p[1];
							dif = cost - topDisp.ptr<float>(v, u, 0)[1];
							ratio = dif / topDisp.ptr<float>(v, u, 0)[1];
						}
					}
				}
				opt << dtV << ",";
				opt << disp << "," << cost << ",";
				if (has_find)
					opt << pos << "," << dif << "," << ratio << endl;
				else
					opt << endl;
			}
		}
	}
	opt.close();
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