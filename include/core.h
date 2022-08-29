#pragma once
#include "qrcode.h"
#include "detection.h"
#include "camera.h"
#include "measure.h"
#include "ocr.h"

extern "C"{
	int  initCamera(int camera_id,  char* camera_ip, int camera_ifset);
	int initOCR();

	int  uploadIcon(char* keyFn, int scale, int Volume);

	ToPythonRect* findTarget(char* fileName, float OV, int saveShow);

	RetBox* useOCR();

	XYZ measureTarget(int x, int y, float fx, float cx, float fy, float cy, float Zc);

	void clearIconsInPro();

	int capture();
}
