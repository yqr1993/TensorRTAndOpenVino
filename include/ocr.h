#pragma once
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <sstream>
#include <codecvt>

#include "configMade.h"


typedef struct RetBox
{
    wchar_t *text;
    int charNum;
    int cx;
    int cy;
    float score;
}RetBox;


void init();
vector<RetBox> use(cv::Mat frame);
