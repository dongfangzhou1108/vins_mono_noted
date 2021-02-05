#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

/**
 * @brief  ImageFrame points, time, pose, IMU  preintegration, whether is key frame
 */
class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
		//key is feature_id, value is Eigen::Matrix(7,1) filled with un_pts, pts and un_pts_velo
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t; //image_msg time
        Matrix3d R; //here R is  R_cam0_camk * R_camk_imuk = R_cam0_imuk(代表第k帧IMU到参考相机帧的旋转矩阵)
        Vector3d T; //here T is trans from camk to cam0(代表相机第k帧到参考帧的平移)
        IntegrationBase *pre_integration; //IMU integration
        bool is_key_frame; //whether is key frame
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);