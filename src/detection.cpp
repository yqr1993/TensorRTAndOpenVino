#include "detection.h"

#if !defined(IF_GPU)
#define IF_GPU 1
#endif


typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
}Bbox;


typedef struct
{
	#if IF_GPU
	// create GPU image heap
	cv::cuda::GpuMat cudaImg;
	cv::cuda::GpuMat cudaTempl;

	// create GPU result heap
	cv::cuda::GpuMat cudaResult;

	// create alg model
	cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(CV_8U, cv::TM_CCOEFF_NORMED);
	#endif

	// template heap
	cv::Mat scaleTemplate;

	// create cpu result heap
	cv::Mat result;

}ScaleAlg;


map<string, vector<ScaleAlg>> templates;


pthread_mutex_t vecLock = PTHREAD_MUTEX_INITIALIZER;


void nms_cpu(vector<Bbox> &bboxes, float threshold) {
    if (bboxes.empty()){
        return ;
    }

    sort(bboxes.begin(), bboxes.end(), [&](Bbox b1, Bbox b2){return b1.score>b2.score;});

    vector<float> area(bboxes.size());
    for (int i=0; i<bboxes.size(); ++i){
        area[i] = (bboxes[i].xmax - bboxes[i].xmin + 1) * (bboxes[i].ymax - bboxes[i].ymin + 1);
    }

    for (int i=0; i<bboxes.size(); ++i){
        for (int j=i+1; j<bboxes.size(); ){
            float left = max(bboxes[i].xmin, bboxes[j].xmin);
            float right = min(bboxes[i].xmax, bboxes[j].xmax);
            float top = max(bboxes[i].ymin, bboxes[j].ymin);
            float bottom = min(bboxes[i].ymax, bboxes[j].ymax);
            float width = max(right - left + 1, 0.f);
            float height = max(bottom - top + 1, 0.f);
            float u_area = height * width;
            float iou = (u_area) / (area[i] + area[j] - u_area);
            if (iou>=threshold){
                bboxes.erase(bboxes.begin()+j);
                area.erase(area.begin()+j);
            }else{
                ++j;
            }
        }
    }
}


double iResize(cv::Mat icon, cv::Mat &Dst, int h, int w, int th, int tw, int i, int scale)
{
	double the_min_side_scaled = 0.0;
	double fix_scale_var = (double)i - (double)scale / 2;
	double templateScale = 0.0;

	if(th >= tw)
	{
		the_min_side_scaled = (double)tw + fix_scale_var;
		templateScale = the_min_side_scaled / (double)tw;
		resize(icon, Dst, cv::Size((int)(the_min_side_scaled), (int)((double)th * templateScale)));
	}
	else
	{
		the_min_side_scaled = (double)th + fix_scale_var;
		templateScale = the_min_side_scaled / (double)th;
		resize(icon, Dst, cv::Size((int)((double)tw * templateScale), (int)(the_min_side_scaled)));
	}

	return templateScale;

}


int loadTarget(string fn, int scale, int Volume)
{
	map<string, vector<ScaleAlg>>::iterator iter = templates.find(fn);
	if(iter != templates.end())
	{
		return -4;
	}

	if(templates.size() > Volume)
	{
		return -2;
	}

	cv::Mat mat = cv::imread(fn);

	int th = mat.rows;
	int tw = mat.cols;

	vector<ScaleAlg> multiScales;

	for(int i = 0; i < scale; i++)
	{
		// create template
		cv::Mat scaleTemplate;
		double s = iResize(mat, scaleTemplate, frame_height, frame_width, th, tw, i, scale);

		// alg struct
		ScaleAlg scaleAlg;

		// template rgb heap
		scaleAlg.scaleTemplate = scaleTemplate;

		#if IF_GPU
		scaleAlg.cudaTempl.upload(scaleTemplate);
		#endif

		multiScales.push_back(scaleAlg);
	}

	templates.insert(pair<string, vector<ScaleAlg>>(fn, multiScales));

	return templates.size();

}


vector<cv::Rect> detect(cv::Mat frame, string fn, float OV)
{
	// get the target template
	map<string, vector<ScaleAlg>>::iterator iter = templates.find(fn);

	// target obj
	vector<ScaleAlg> target;

	// returns
	vector<cv::Rect> candicate;

	if(iter != templates.end())
	{
		target = iter->second;
	}
	else
	{
		printf("The target file has not been load!\n");
		return candicate;
	}

	vector<Bbox> bboxS;

	//parallel for detect
	cv::parallel_for_(cv::Range(0, target.size()), [&](const cv::Range& range) {

		const int begin = range.start;
        const int end = range.end;

		for (int i = begin; i < end; i++)
		{
			ScaleAlg scaleAlgTar = target[i];

			double stw = scaleAlgTar.scaleTemplate.cols;
			double sth = scaleAlgTar.scaleTemplate.rows;

			#if IF_GPU

			// execute func
			scaleAlgTar.cudaImg.upload(frame);
			scaleAlgTar.alg->match(scaleAlgTar.cudaImg, scaleAlgTar.cudaTempl, scaleAlgTar.cudaResult);
    		scaleAlgTar.cudaResult.download(scaleAlgTar.result);

			#else

			// execute func
			matchTemplate(frame, scaleAlgTar.scaleTemplate, scaleAlgTar.result, CV_TM_CCOEFF_NORMED);

			#endif


			for(int i = 0; i < scaleAlgTar.result.rows; i++)
			{
				for(int j = 0; j < scaleAlgTar.result.cols; j++)
				{
					if(scaleAlgTar.result.ptr<float>(i)[j] > OV)
					{
						Bbox box{(float)j, (float)i, (float)(j + stw), (float)(i + sth), scaleAlgTar.result.ptr<float>(i)[j]};
						pthread_mutex_lock(&vecLock);
						bboxS.push_back(box);
						pthread_mutex_unlock(&vecLock);
					}

				}
			}
		}
	});

	nms_cpu(bboxS, 0.5);

	for(auto b : bboxS)
	{
		candicate.push_back(cv::Rect(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin));
	}

	return candicate;
}

void delTarget()
{
	map<string, vector<ScaleAlg>>::iterator iter;
    for (iter = templates.begin(); iter != templates.end(); ++iter)
    {
		int scale = iter->second.size();
		for(int i = 0; i < scale; i ++)
		{
			(iter->second)[i].cudaImg.release();
			(iter->second)[i].cudaTempl.release();

			(iter->second)[i].cudaResult.release();
			(iter->second)[i].alg.release();

			(iter->second)[i].scaleTemplate.release();
			(iter->second)[i].result.release();

		}

		vector<ScaleAlg> noneVec;

		(iter->second).swap(noneVec);
	}

	map<string, vector<ScaleAlg>> noneMap;

	templates.swap(noneMap);

	malloc_trim(0);
}
