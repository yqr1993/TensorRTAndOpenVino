#include "core.h"

#define PROC_SUCCESS  	 1
#define PROC_FAIL 	 	-1
#define PROC_RUNNING 	-2
#define MEMORY_FULL  	-3
#define ALREADY_IN   	-4
#define PARAM_ERROR  	-5
#define FILE_NOT_EXIST  -6


int frame_height = 720;
int frame_width = 1280;

int INIT_CAM = 0;
int INIT_OCR = 0;


int initCamera(int camera_id,  char* camera_ip, int camera_ifset)
{
	if(INIT_CAM == 0)
	{
		int ret2 = initCam(camera_id, camera_ip, camera_ifset);
		
		if(ret2 == 1)
		{
			INIT_CAM = 1;
			return PROC_SUCCESS;
		}
		else
		{
			return PROC_FAIL;
		}

	}
	else
	{
		return PROC_RUNNING;
	}
}


int initOCR()
{
	if(INIT_OCR == 0)
	{
		init();
		INIT_OCR = 1;
		return PROC_SUCCESS;
	}
	else
	{
		return PROC_RUNNING;
	}

}


int uploadIcon(char* keyFn, int scale, int Volume)
{
	int ret = loadTarget(keyFn, scale, Volume);

	if(ret == -2)
	{
		return MEMORY_FULL;
	}
	else if (ret == -4)
	{
		return ALREADY_IN;
	}

	return ret;
}


ToPythonRect* findTarget(char* fileName, float OV, int saveShow)
{
	cv::Mat imageDst = imageCapture();
	
	vector<cv::Rect> pos = detect(imageDst, fileName, OV);

	if(saveShow)
	{
		for(auto pt : pos)
        {
            rectangle(imageDst, cv::Rect(pt.x, pt.y, pt.width, pt.height), cv::Scalar(0, 255, 0));
        }
		saveRGB(imageDst, "render_frames/temp.jpg");
	}

	int posSIZE = pos.size();

	ToPythonRect* pythonRectS = (ToPythonRect*)malloc(sizeof(ToPythonRect) * (posSIZE + 1));

	for(int i = 0; i < posSIZE; i++)
	{
		ToPythonRect pythonRect(pos[i].x, pos[i].y, pos[i].width, pos[i].height);

		memcpy(pythonRectS + i, &pythonRect, sizeof(ToPythonRect));
	}

	ToPythonRect tail(-1000, -1000, 0, 0);

	memcpy(pythonRectS + posSIZE, &tail, sizeof(ToPythonRect));

	return pythonRectS;
}


RetBox* useOCR()
{
	cv::Mat imageDst = imageCapture();
    vector<RetBox> rets = use(imageDst);

	int lineSIZE = rets.size();

	RetBox *retBoxS = (RetBox*)malloc(sizeof(RetBox) * (lineSIZE + 1));

	memcpy(retBoxS, rets.data(), sizeof(RetBox) * lineSIZE); 
	RetBox tail = {(wchar_t*)L"none", -1, 0, 0, 0.0};
	memcpy(retBoxS + lineSIZE, &tail, sizeof(RetBox));

	return retBoxS;
}


XYZ measureTarget(int x, int y, float fx, float cx, float fy, float cy, float Zc)
{

	XYZ xyz = mPos(cv::Rect(x, y, 0, 0), fx, cx, fy, cy, Zc);

	return xyz;

}


int capture()
{
	cv::Mat mat = imageCapture();
	saveRGB(mat, "frames/temp.jpg");

	return PROC_SUCCESS;
}


void clearIconsInPro()
{
	delTarget();
}
