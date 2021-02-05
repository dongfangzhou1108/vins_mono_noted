#pragma once 

#include <vector>
#include "../parameters.h"
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
using namespace Eigen;
#include <ros/console.h>

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
class InitialEXRotation
{
public:
	InitialEXRotation();
    /**
     * @brief  iterated calc ric, judge whether extrinsic rotation calibed successfully, by judging calc time and singular value
     * @param {vector<pair<Vector3d,Vector3d>>} corres
     * @param {Quaterniond} delta_q_imu
     * @param {Matrix3d} &calib_ric_result
     * @return {*}
     */    
    bool CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result);
private:
	/**
  * @brief  calc relative rotation between two feature frame
  * @param {constvector<pair<Vector3d,Vector3d>>} &corres
  * @return {*}
  */ 
 Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres);

    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    /**
     * @brief  F matrix SVD decomposition
     */    
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    int frame_count; //will be add in func CalibrationExRotation() when doing initialization

    vector< Matrix3d > Rc; //camera roatation for calib extrinsic
    vector< Matrix3d > Rimu; //IMU rotation for calib extrinsic
    vector< Matrix3d > Rc_g;
    Matrix3d ric;
};


