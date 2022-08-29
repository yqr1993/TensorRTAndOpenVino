#include "ocr.h"


#define TIME_COUNT 0
#define DEBUG 0
#define SHOW 0


#if TIME_COUNT
#include<time.h>
vector<string> charsetTest;
#endif


typedef struct FrameWithRatio
{
    cv::Mat frame;
    float ratioH;
    float ratioW;
    int h;
    int w;
}FrameWithRatio;

typedef struct RecBox
{
    string text;
    vector<cv::Point> pts;
    float score;
    float h;
    float w;
    float cx;
    float cy;
}RecBox;

const float MEAN[] = {0.485, 0.456, 0.406};
const float STD[]     = {0.229, 0.224, 0.225};
const int C = 3;

const float thresh = 0.3;
const float box_thresh = 0.3;
const int min_size = 3;

const float recHeight = 32;

vector<wchar_t> charset;


using samplesCommon::SampleUniquePtr;


FrameWithRatio resizeToRight(cv::Mat im, int maxSide=1920)
{
    int h = im.rows;
    int w = im.cols;

    int resizeH = h;
    int resizeW = w;

     float ratio = 1.0;

    if (max(h, w) > maxSide)
    {
        if(resizeH > resizeW)
        {
             ratio = (float)maxSide / resizeH;
        }
        else
        {
            ratio = (float)maxSide / resizeW;
        }
    }

    resizeH = (int)(resizeH * ratio);
    resizeW = (int)(resizeW * ratio);

    if(resizeH % 32 == 0){}
    else if((float)resizeH / 32.0 < 2.0)
    {
        resizeH = 32;
    }
    else
    {
        resizeH = (int)round((float)resizeH / 32.0 ) * 32;
    }

    if(resizeW % 32 == 0){}
    else if((float)resizeW / 32.0 < 2.0)
    {
        resizeW = 32;
    }
    else
    {
        resizeW = (int)round((float)resizeW / 32.0 ) * 32;
    }

    cv::Mat resultFrame;
    cv::resize(im, resultFrame, cv::Size(resizeW, resizeH));

    float ratioH = (float)resizeH / (float)h;
    float ratioW = (float)resizeW / (float)w;

    FrameWithRatio ret;
    ret.frame = resultFrame;
    ret.ratioH = ratioH;
    ret.ratioW = ratioW;
    ret.h = resizeH;
    ret.w = resizeW;

    return ret;
}


void  unclip(vector<cv::Point> &ptsTemp, cv::RotatedRect  rectTemp, float unclip_ratio=1.6)
{
    float offset = unclip_ratio * min(rectTemp.size.height, rectTemp.size.width);

    float cx = rectTemp.center.x;
    float cy = rectTemp.center.y;

    float x1 = (ptsTemp[0].x + ptsTemp[1].x) / 2.0;
    float y1 = (ptsTemp[0].y + ptsTemp[1].y) / 2.0;

    float vx1 = x1 - cx;
    float vy1 = y1 - cy;

    float md1 = sqrt(vx1 * vx1 + vy1 * vy1);
    float mdn1 = md1 + offset;
    float sratio1 = mdn1 / md1;

    float vxn1 = vx1 * sratio1;
    float vyn1 = vy1 * sratio1;

    float x2 = (ptsTemp[1].x + ptsTemp[2].x) / 2.0;
    float y2 = (ptsTemp[1].y + ptsTemp[2].y) / 2.0;

    float vx2 = x2 - cx;
    float vy2 = y2 - cy;

    float md2 = sqrt(vx2 * vx2 + vy2 * vy2);
    float mdn2 = md2 + offset;
    float sratio2 = mdn2 / md2;

    float vxn2 = vx2 * sratio2;
    float vyn2 = vy2 * sratio2;

    float x3 = (ptsTemp[2].x + ptsTemp[3].x) / 2.0;
    float y3 = (ptsTemp[2].y + ptsTemp[3].y) / 2.0;

    float vx3 = x3 - cx;
    float vy3 = y3 - cy;

    float md3 = sqrt(vx3 * vx3 + vy3 * vy3);
    float mdn3 = md3 + offset;
    float sratio3 = mdn3 / md3;

    float vxn3 = vx3 * sratio3;
    float vyn3 = vy3 * sratio3;

    float x4 = (ptsTemp[3].x + ptsTemp[0].x) / 2.0;
    float y4 = (ptsTemp[3].y + ptsTemp[0].y) / 2.0;

    float vx4 = x4 - cx;
    float vy4 = y4 - cy;

    float md4 = sqrt(vx4 * vx4 + vy4 * vy4);
    float mdn4 = md4 + offset;
    float sratio4 = mdn4 / md4;  

    float vxn4 = vx4 * sratio4;
    float vyn4 = vy4 * sratio4;

    // new vectors
    float offsetVX1 = vxn4 + vxn1;
    float offsetVY1 = vyn4 + vyn1;

    float offsetVX2 = vxn1 + vxn2;
    float offsetVY2 = vyn1 + vyn2;

    float offsetVX3 = vxn2 + vxn3;
    float offsetVY3 = vyn2 + vyn3;

    float offsetVX4 = vxn3 + vxn4;
    float offsetVY4 = vyn3 + vyn4;

    // new pts
    ptsTemp[0].x = round(offsetVX1 + cx);
    ptsTemp[0].y = round(offsetVY1 + cy);

    ptsTemp[1].x = round(offsetVX2 + cx);
    ptsTemp[1].y = round(offsetVY2 + cy);

    ptsTemp[2].x = round(offsetVX3 + cx);
    ptsTemp[2].y = round(offsetVY3 + cy);

    ptsTemp[3].x = round(offsetVX4 + cx);
    ptsTemp[3].y = round(offsetVY4 + cy);
}


vector<RecBox> detectPost(cv::Mat bitMap)
{
    vector<vector<cv::Point>> contours;
    
    # if DEBUG
    cv::imshow("show", bitMap);
    cv::waitKey(0);
    # endif

    cv::findContours(bitMap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    vector<RecBox> recBoxes;

    for(vector<cv::Point> contour : contours)
    {
        cv::RotatedRect minRect = cv::minAreaRect(cv::Mat(contour));

        if(min(minRect.size.height, minRect.size.width) < min_size)
        {
            continue;
        }

        cv::Mat minRectPt;
        cv::boxPoints(minRect, minRectPt);

        int numRect = minRectPt.rows;
        vector<cv::Point> ptsTemp;
        for(int i = 0; i < numRect; i++)
        {
            float *pts = minRectPt.ptr<float>(i);
            ptsTemp.push_back(cv::Point(pts[0], pts[1]));
        }

        unclip(ptsTemp, minRect);

        float side1 = sqrt((ptsTemp[0].x - ptsTemp[1].x) * (ptsTemp[0].x - ptsTemp[1].x) + (ptsTemp[0].y - ptsTemp[1].y) * (ptsTemp[0].y - ptsTemp[1].y));
        float side2 = sqrt((ptsTemp[1].x - ptsTemp[2].x) * (ptsTemp[1].x - ptsTemp[2].x) + (ptsTemp[1].y - ptsTemp[2].y) * (ptsTemp[1].y - ptsTemp[2].y));
        RecBox recBox = {"", ptsTemp, 1.0, min(side1, side2), max(side1, side2), minRect.center.x, minRect.center.y};

        recBoxes.push_back(recBox);
    }

    return recBoxes;
}


cv::Mat getRotateCropImage(cv::Mat imageOrin, vector<cv::Point> box, float ratio_w, float ratio_h, float w_, float h_, float cx, float cy)
{
    float w = w_ / ratio_w;
    float h  = h_  / ratio_h;

    vector<cv::Point2f> temp;
    float x1 = box[0].x * ratio_w;
    float x2 = box[1].x * ratio_w;
    float x3 = box[2].x * ratio_w;
    float x4 = box[3].x * ratio_w;

    float y1 = box[0].y * ratio_h;
    float y2 = box[1].y * ratio_h;
    float y3 = box[2].y * ratio_h;
    float y4 = box[3].y * ratio_h;

    temp.push_back(cv::Point2f(x1, y1));
    temp.push_back(cv::Point2f(x2, y2));
    temp.push_back(cv::Point2f(x3, y3));
    temp.push_back(cv::Point2f(x4, y4));

    vector<cv::Point2f> left;
    vector<cv::Point2f> right;
    vector<cv::Point2f> src;
    vector<cv::Point2f> dst;

    for (auto pt : temp)
    {
        if(pt.x < cx)
        {
            left.push_back(pt);
        }
        else
        {
            right.push_back(pt);
        }
    }

    if (left[0].y < left[1].y)
    {
        src.push_back(left[0]);
        dst.push_back(cv::Point(0, 0));
        src.push_back(left[1]);
        dst.push_back(cv::Point(0, h));
    }
    else
    {
        src.push_back(left[1]);
        dst.push_back(cv::Point(0, 0));
        src.push_back(left[0]);
        dst.push_back(cv::Point(0, h));
    }

    if (right[0].y < right[1].y)
    {
        src.push_back(right[0]);
        dst.push_back(cv::Point(w, 0));
        src.push_back(right[1]);
        dst.push_back(cv::Point(w, h));
    }
    else
    {
        src.push_back(right[1]);
        dst.push_back(cv::Point(w, 0));
        src.push_back(right[0]);
        dst.push_back(cv::Point(w, h));
    }


    cv::Mat transfrom = cv::getPerspectiveTransform(src, dst);
    cv::Mat recMat;
    cv::warpPerspective(imageOrin, recMat, transfrom, cv::Size((int)round(w), (int)round(h)), cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    #if DEBUG
    cv::imshow("show", recMat);
    cv::waitKey(0);
    #endif

    return recMat;
}


cv::Mat resizeNormImageRec(cv::Mat textImage)
{
    int h = textImage.rows;
    int w = textImage.cols;

    float resizeScale = (float)h / 32.0;
    int resizeW = (int)round(w / resizeScale);

    cv::Mat resizeScaleImage;
    cv::resize(textImage, resizeScaleImage, cv::Size(resizeW, 32));

    # if DEBUG
    std::cout << resizeScaleImage.cols << " " << resizeScaleImage.rows << endl;
    cv::imshow("show", resizeScaleImage);
    cv::waitKey(0);
    # endif

    return resizeScaleImage;
}


void decode(cv::Mat textMat, RetBox &retBox)
{
    int w = textMat.rows;

    #if DEBUG
    std::cout << textMat.cols << " " << textMat.rows << endl;
    #endif

    vector<wchar_t> textCode;

    int last_index = 0;
    for (int i = 0; i < w; i++)
    {   
        float *charSoftMax = textMat.ptr<float>(i);

        int index = distance(charSoftMax, max_element(charSoftMax,  charSoftMax + 6625));

        // int index = 0;
        // float maxOne = 0.0;
        // int maxIndex = 0;
        // for (int j = 0; j < 6625; j++)
        // {
        //     if(charSoftMax[j] > maxOne)
        //     {
        //         maxOne = charSoftMax[j] ;
        //         maxIndex = j;
        //     }
        //     index ++;
        // }

        #if DEBUG
        std::cout << index << " ";
        #endif

        if(index != last_index && index != 0 && index != 6624)
        {
            textCode.push_back(charset[index-1]);
            #if TIME_COUNT
            std::cout << charsetTest[index - 1] ;
            #endif
        }

        last_index = index;
    }

    #if DEBUG || TIME_COUNT
    std::cout << endl;
    #endif

    retBox.text = new wchar_t[textCode.size()];
    memcpy(retBox.text, textCode.data(), textCode.size() * sizeof(wchar_t));
    retBox.charNum = textCode.size();
}


class OnnxOCR
{
public:
    OnnxOCR(string onnxModelPath_, string inputName_, string outputName_)
        : onnxModelPath(onnxModelPath_),
          inputName(inputName_),
          outputName(outputName_),
          mEngine(nullptr)
    {
        ifstream modelFile(onnxModelPath);
        modelFile.seekg(0, ios::end);
        size_t size = modelFile.tellg(); 
        modelFile.seekg(0, ios::beg);

        char * buff = new char [size];
        modelFile.read(buff, size);
        modelFile.close();
        
        IRuntime* runtime = createInferRuntime(sample::gLogger.getTRTLogger());
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine((void*)buff, size, nullptr));

        delete buff;

        context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    }

    virtual cv::Mat infer(cv::Mat frame) = 0;


protected:
    string onnxModelPath;
    string inputName;
    string outputName;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; 
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    samplesCommon::ManagedBuffer mInput{}; 
    samplesCommon::ManagedBuffer mOutput{}; 

};


class OnnxOCRdetect : OnnxOCR
{
public:
    OnnxOCRdetect(string onnxModelPath_, string inputName_, string outputName_): OnnxOCR(onnxModelPath_, inputName_, outputName_) {}
    cv::Mat infer(cv::Mat frame);
};


class OnnxOCRec: OnnxOCR
{
public:
    OnnxOCRec(string onnxModelPath_, string inputName_, string outputName_): OnnxOCR(onnxModelPath_, inputName_, outputName_) {}
    cv::Mat  infer(cv::Mat frame);
};


cv::Mat OnnxOCRdetect::infer(cv::Mat frame)
{
    FrameWithRatio framez = resizeToRight(frame);
    Dims4 inputDims(1, C, framez.h, framez.w);
    Dims4 outputDims(1, 1, framez.h, framez.w);

    mInput.hostBuffer.resize(inputDims);
    mInput.deviceBuffer.resize(inputDims);

    float* hostDataBuffer = static_cast<float*>(mInput.hostBuffer.data());
    for (int i = 0; i < C; i++)
    {
        for(int j = 0; j < framez.h; j++)
        {
            cv::Vec3b *pr = framez.frame.ptr<cv::Vec3b>(j);
            for(int k = 0; k < framez.w; k++)
            {
                float p = (float)(pr[k][i]) / 255.0;
                p = p - MEAN[i];
                p = p / STD[i];

                hostDataBuffer[i * framez.h * framez.w + framez.w * j + k] = p ;
            }
        }
    }

    CHECK(cudaMemcpy(mInput.deviceBuffer.data(), mInput.hostBuffer.data(), mInput.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));

    context->setBindingDimensions(0, inputDims);
    //context->setBindingDimensions(1, outputDims);
    context->allInputShapesSpecified();

    mOutput.hostBuffer.resize(outputDims);
    mOutput.deviceBuffer.resize(outputDims);

    std::vector<void*> predicitonBindings = {mInput.deviceBuffer.data(), mOutput.deviceBuffer.data()};
    context->executeV2(predicitonBindings.data()); 

    CHECK(cudaMemcpy(mOutput.hostBuffer.data(), mOutput.deviceBuffer.data(), mOutput.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));
    const float* bufRaw = static_cast<const float*>(mOutput.hostBuffer.data());

    std::vector<float> prob(bufRaw, bufRaw + mOutput.hostBuffer.size());

    cv::Mat DBmat(framez.h, framez.w, CV_8UC1);

    for(int j = 0; j < framez.h; j++)
    {
        for(int k = 0; k < framez.w; k++)
        {
            if(prob[j * framez.w + k] > thresh)
            {
                DBmat.at<uchar>(j, k) = 255;
            }
            else
            {
                DBmat.at<uchar>(j, k) = 0;
            }
        }
    }

    return DBmat;
}


cv::Mat OnnxOCRec::infer(cv::Mat frame)
{
    int h = frame.rows;
    int w_ = frame.cols;

    int concatW = w_ % 4 == 0 ? 0 : 4 - (w_ % 4);
    int w = w_ + concatW;

    Dims4 inputDims(1, C, h, w);
    Dims3 outputDims(1, int(w / 4), 6625);

    mInput.hostBuffer.resize(inputDims);
    mInput.deviceBuffer.resize(inputDims);

    float* hostDataBuffer = static_cast<float*>(mInput.hostBuffer.data());
    for (int i = 0; i < C; i++)
    {
        for(int j = 0; j < h; j++)
        {
            cv::Vec3b *pr = frame.ptr<cv::Vec3b>(j);
            for(int k = 0; k < w; k++)
            {
                float p = (float)(pr[k][i]) / 255.0;
                p = p - 0.5;
                p = p / 0.5;
                if (j >= w_)
                {
                    hostDataBuffer[i * h * w + w * j + k] = 0.0;
                }
                else
                {
                    hostDataBuffer[i * h * w + w * j + k] = p ;
                }

            }
        }
    }

    CHECK(cudaMemcpy(mInput.deviceBuffer.data(), mInput.hostBuffer.data(), mInput.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));

    context->setBindingDimensions(0, inputDims);
    //context->setBindingDimensions(1, outputDims);
    context->allInputShapesSpecified();

    mOutput.hostBuffer.resize(outputDims);
    mOutput.deviceBuffer.resize(outputDims);

    std::vector<void*> predicitonBindings = {mInput.deviceBuffer.data(), mOutput.deviceBuffer.data()};
    context->executeV2(predicitonBindings.data()); 

    CHECK(cudaMemcpy(mOutput.hostBuffer.data(), mOutput.deviceBuffer.data(), mOutput.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));
    const float* bufRaw = static_cast<const float*>(mOutput.hostBuffer.data());

    std::vector<float> prob(bufRaw, bufRaw + mOutput.hostBuffer.size());

    cv::Mat textMat(int(w / 4), 6625, CV_32FC1);

    for(int j = 0; j < int(w / 4); j++)
    {
        for(int k = 0; k < 6625; k++)
        {
            textMat.at<float>(j, k) = prob[j * 6625 + k];
        }
    }

    return textMat;
}


OnnxOCRdetect *onnxDetect;
OnnxOCRec *onnxRec;


void init()
{
    onnxDetect = new OnnxOCRdetect("ONNXmobileEngine/detect/detect.engine", "x", "save_infer_model/scale_0.tmp_1");
    onnxRec    = new OnnxOCRec("ONNXmobileEngine/character_rec/rec.engine", "x", "save_infer_model/scale_0.tmp_1");

    ifstream file;
    file.open("ONNXmobileEngine/ppocr_keys_v1.txt");
    bool ifFileExit = file.good();
    if(! ifFileExit)
    {
        cerr << "no file exist!" << endl;
        exit(-1);
    }

    std::wstring_convert<std::codecvt_utf8<wchar_t>> crt;

    string stringTemp;
    while(getline(file, stringTemp))
    {
        # if TIME_COUNT
        charsetTest.push_back(stringTemp);
        # endif
        wstring data = crt.from_bytes(stringTemp);
        charset.push_back(data[0]);
    }

    file.close();
}


vector<RetBox> use(cv::Mat frame)
{
    cv::Mat Ret1 = onnxDetect->infer(frame);

    float ratio_w = (float)frame.cols / (float)Ret1.cols;
    float ratio_h = (float)frame.rows / (float)Ret1.rows;

    vector<RecBox> recBoxes = detectPost(Ret1);

    # if SHOW
    FrameWithRatio showTemp = resizeToRight(frame.clone());
    cv::Mat image = showTemp.frame;

    for (auto testRecBox : recBoxes)
    {
        cv::circle(image, cv::Point(testRecBox.cx, testRecBox.cy), 3, cv::Scalar(255, 0, 0));

        cv::line(image, testRecBox.pts[0], testRecBox.pts[1], cv::Scalar(0, 0, 255));
        cv::line(image, testRecBox.pts[1], testRecBox.pts[2], cv::Scalar(0, 0, 255));
        cv::line(image, testRecBox.pts[2], testRecBox.pts[3], cv::Scalar(0, 0, 255));
        cv::line(image, testRecBox.pts[3], testRecBox.pts[0], cv::Scalar(0, 0, 255));
    }

    cv::imshow("show", image);
    cv::waitKey(0);
    # endif

    vector<RetBox> retBoxes;

    for(int i = 0; i < recBoxes.size(); i++)
    {
        RetBox retBox;

        vector<cv::Point> pts = recBoxes[i].pts;
        cv::Mat textFrame = getRotateCropImage(frame, pts, ratio_w, ratio_h, recBoxes[i].w, recBoxes[i].h, recBoxes[i].cx, recBoxes[i].cy);
        cv::Mat lineMat      = resizeNormImageRec(textFrame);
        cv::Mat textMat = onnxRec->infer(lineMat);
        decode(textMat, retBox);

        retBox.cx = recBoxes[i].cx;
        retBox.cy = recBoxes[i].cy;
        retBox.score = recBoxes[i].score;

        retBoxes.push_back(retBox);
    }

    return retBoxes;
}
