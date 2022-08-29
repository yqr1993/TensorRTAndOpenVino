#include "measure.h"


void myGetSingleMarkerObjectPoints(float markerLength_x, float markerLength_y, cv::OutputArray _objPoints) {
    _objPoints.create(4, 1, CV_32FC3);
    cv::Mat objPoints = _objPoints.getMat();
    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.ptr< cv::Vec3f >(0)[0] = cv::Vec3f(-markerLength_x / 2.f, markerLength_y / 2.f, 0);
    objPoints.ptr< cv::Vec3f >(0)[1] = cv::Vec3f(markerLength_x / 2.f, markerLength_y / 2.f, 0);
    objPoints.ptr< cv::Vec3f >(0)[2] = cv::Vec3f(markerLength_x / 2.f, -markerLength_y / 2.f, 0);
    objPoints.ptr< cv::Vec3f >(0)[3] = cv::Vec3f(-markerLength_x / 2.f, -markerLength_y / 2.f, 0);
}


void myEstimatePoseSingleMarkers(cv::InputArrayOfArrays _corners, float markerLength_x, float markerLength_y,
    cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
    cv::OutputArray _rvecs, cv::OutputArray _tvecs) {

    cv::Mat markerObjPoints;
    myGetSingleMarkerObjectPoints(markerLength_x, markerLength_y, markerObjPoints);
    int nMarkers = (int)_corners.total();
    _rvecs.create(nMarkers, 1, CV_64FC3);
    _tvecs.create(nMarkers, 1, CV_64FC3);

    cv::Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();

    // for each marker, calculate its pose
    cv::parallel_for_(cv::Range(0, nMarkers), [&](const cv::Range& range) {
        const int begin = range.start;
        const int end = range.end;

        for (int i = begin; i < end; i++) {
            solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs, rvecs.at<cv::Vec3d>(i),
                tvecs.at<cv::Vec3d>(i));
        }
        });
}


double calcuD(std::vector<cv::Vec3d> tvecs)
{
    if (tvecs.size() != 1)
    {
        return -10.0;
    }

    return tvecs[0][2];
}


double calcuS(std::vector<cv::Vec3d> tvecs)
{
    if (tvecs.size() != 1)
    {
        return -10.0;
    }

    return tvecs[0][0];
}


double calcuV(std::vector<cv::Vec3d> tvecs)
{
    if (tvecs.size() != 1)
    {
        return -10.0;
    }

    return tvecs[0][1];
}


double calcuAlpha(std::vector<cv::Vec3d> rvecs)
{
    if (rvecs.size() != 1)
    {
        return -10.0;
    }

    cv::Mat rotM;
    cv::Rodrigues(rvecs[0], rotM);
    double theta = atan2(-rotM.at<double>(2, 0), sqrt(rotM.at<double>(2, 1) * rotM.at<double>(2, 1) + rotM.at<double>(2, 2) * rotM.at<double>(2, 2))) * 180 / 3.141592625;

    return theta;

}


XYZ mPos(cv::Rect pos, float fx, float cx, float fy, float cy, float Zc)
{
	double Xp = (double)pos.x + pos.width / 2;
	double Yp = (double)pos.y + pos.height / 2;

	// uv to camera
	double Xc = Zc * (Xp - cx) / fx;
	double Yc = Zc * (Yp - cy) / fy;

	//undistort
	//double r2 = Xc * Xc + Yc * Yc;
	//double Xcrt = Xc * (1 + k1 * pow(r2, 1) + k2 * pow(r2, 2) + k3 * pow(r2, 3));
	//double Ycrt = Yc * (1 + k1 * pow(r2, 1) + k2 * pow(r2, 2) + k3 * pow(r2, 3));

	//Xcrt = Xcrt + (2 * p1 * Xcrt * Ycrt + p2 * (r2 + 2 * Xcrt * Xcrt));
	//Ycrt = Ycrt + (p1 * (r2 + 2 * Ycrt * Ycrt) + 2 * p2 * Xcrt * Ycrt);

    //cout << xCorrected << " " << yCorrected << endl;
    
	return XYZ(Xc, Yc, Zc);
}