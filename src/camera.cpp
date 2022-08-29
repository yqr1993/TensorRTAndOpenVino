#include "camera.h"


pthread_t m_Thread_ID;

pthread_mutex_t camLock;

cv::VideoCapture* cam;

cv::Mat image = cv::Mat::zeros(frame_height, frame_width, CV_8UC3);


int initCam(int camera_id, string camera_ip, int camera_ifset) {
    if (camera_ip != "none")
    {
        cam = new cv::VideoCapture(camera_ip, cv::CAP_V4L2, {
        cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        cv::CAP_PROP_FRAME_WIDTH, frame_width,
        cv::CAP_PROP_FRAME_HEIGHT, frame_height,
    });
    }
    else
    {
        cam = new cv::VideoCapture(camera_id, cv::CAP_V4L2, {
        cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        cv::CAP_PROP_FRAME_WIDTH, frame_width,
        cv::CAP_PROP_FRAME_HEIGHT, frame_height,
    });
    }

    // set camera
    if (camera_ifset)
    {
        cam->set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
        cam->set(cv::CAP_PROP_EXPOSURE, 50);
        cam->set(cv::CAP_PROP_GAIN, 0);
        cam->set(cv::CAP_PROP_BUFFERSIZE, 3);
    }

    m_Thread_ID = 0;

    camLock = PTHREAD_MUTEX_INITIALIZER;

    if(cam->isOpened())
    {
        pthread_create(&m_Thread_ID, nullptr, &imageRead, nullptr);
        printf("the camera has been connected in width : %d, height: %d\n", frame_width, frame_height);
        return 1;
    }
    else
    {
        printf("the camera has not been connected!\n");
        return -1;
    }
}


void releaseCam() {
    cam->release();
    delete cam;
    pthread_join(m_Thread_ID, NULL);
}


void* imageRead(void *arg)
{   
    while(cam->isOpened())
    {   
        cam->grab();
        pthread_mutex_lock(&camLock);
        cam->retrieve(image);
        pthread_mutex_unlock(&camLock);
    }

    return nullptr;
}


cv::Mat imageCapture()
{
    pthread_mutex_lock(&camLock);
    cv::Mat imageDst = image.clone();
    pthread_mutex_unlock(&camLock);
    //gamma_correction(image, imageDst, 2.0);

    return imageDst;

}


void saveRGB(cv::Mat mat, string fileName)
{
    imwrite(fileName, mat);
}


void gamma_correction(cv::Mat& src, cv::Mat& dst, float K) {
    uchar LUT[256];
    src.copyTo(dst);
    for (int i = 0; i < 256; i++) {
        float f = i / 255.0;
        f = pow(f, K);
        LUT[i] = cv::saturate_cast<uchar>(f * 255.0);
    }

    if (dst.channels() == 1) {
        cv::MatIterator_<uchar> it = dst.begin<uchar>();
        cv::MatIterator_<uchar> it_end = dst.end<uchar>();
        for (; it != it_end; ++it) {
            *it = LUT[(*it)];
        }
    }
    else {
        cv::MatIterator_<cv::Vec3b> it = dst.begin<cv::Vec3b>();
        cv::MatIterator_<cv::Vec3b> it_end = dst.end<cv::Vec3b>();
        for (; it != it_end; ++it) {
            (*it)[0] = LUT[(*it)[0]];
            (*it)[1] = LUT[(*it)[1]];
            (*it)[2] = LUT[(*it)[2]];
        }
    }
}
