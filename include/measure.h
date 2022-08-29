#pragma once
#include "configMade.h"


void myGetSingleMarkerObjectPoints(float markerLength_x, float markerLength_y, cv::OutputArray _objPoints);

void myEstimatePoseSingleMarkers(cv::InputArrayOfArrays _corners, float markerLength_x, float markerLength_y,
                                 cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
                                 cv::OutputArray _rvecs, cv::OutputArray _tvecs);

double calcuD(std::vector<cv::Vec2d> tvecs);
double calcuS(std::vector<cv::Vec2d> tvecs);
double calcuV(std::vector<cv::Vec2d> tvecs);
double calcuAlpha(std::vector<cv::Vec2d> rvecs);

XYZ mPos(cv::Rect pos, float fx, float cx, float fy, float cy, float Zc);
