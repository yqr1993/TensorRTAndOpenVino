#pragma once
#include "configMade.h"

extern "C" void initZbarConfig();
extern "C" Symbol_target detectQR(cv::Mat opencv_frame, int width, int height, string target);
