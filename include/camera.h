#pragma once
#include <pthread.h>
#include "configMade.h"


extern int frame_height;
extern int frame_width;


int initCam(int camera_id, string camera_ip, int camera_ifset);
void releaseCam();

void* imageRead(void *arg);
cv::Mat imageCapture();  // get the current frame

void saveRGB(cv::Mat mat, string fileName);  // save the RGB img

void gamma_correction(cv::Mat& src, cv::Mat& dst, float K);
