#pragma once
#include "configMade.h"
#include "camera.h"


int loadTarget(string fn, int scale, int Volume);
vector<cv::Rect> detect(cv::Mat frame, string fn, float OV);
void delTarget();
