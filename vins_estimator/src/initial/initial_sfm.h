#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;



struct SFMFeature
{
    bool state; //all begin with false
    int id; //feature_id
    vector<pair<int,Vector2d>> observation; //pair of frame count and feature un_pts
    double position[3]; //trangulate 3d point
    double depth;
};

/**
 * @brief  construct reproject error
 */
struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
	/**
  * @brief  visual SFM
  * @param frame_num = frame_count +1 is the number of frame in the sliding window
  * @param	q the array of quaternion from arbitrarily frame to lth
  *	@param T the array of t vector from arbitrarily frame to lth
  * @param sfm_f feature seperated in feature id
  * @return {*}
  */ 
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	/**
  * @brief  using feature have triangulated to PnP for clac pose, only use the feather first time observed in i th image
  * @param {Matrix3d} &R_initial
  * @param {Vector3d} &P_initial
  * @param {int} i
  * @param {vector<SFMFeature>} &sfm_f
  * @return {*}
  */ 
 bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	/**
  * @brief  triangulate for one point
  */ 
 void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	/**
  * @brief  trangulate all feathers between two frames in the last frame coordinate
  */ 
 void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;
};