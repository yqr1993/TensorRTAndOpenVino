#pragma once
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>

#include <cstdlib>
#include <zbar.h>
#include <malloc.h>

#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace zbar;


//the struct
extern "C" struct Symbol_target
{
    string					 data;
    std::vector<cv::Point2f> corner;
    cv::Point center;
};

extern "C" struct ParamQR
{
    string info;
    double angle;
    double d;
    double s;
    double v;
    double cx;
    double cy;
};

extern "C" struct XYZ
{
	double x;
	double y;
	double z;

	XYZ(double x, double y, double z) :
		x(x), y(y), z(z)
	{}
};

extern "C" struct ToPythonRect
{
	int x;
	int y;
	int w;
	int h;

    ToPythonRect(int x, int y, int w, int h) :
		x(x), y(y), w(w), h(h)
	{}
};
