#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

/**
 * @brief feature class for one point
 */
class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;
    Vector3d point; //un_pts in normalized plane
    Vector2d uv; //pixel
    Vector2d velocity; //un_pts velocity in x, y axis
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

/**
 * @brief  feature class for FeaturePerFrame with same feature_id
 */
class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame; //the first frame which this feature occured
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    /**
     * @brief  initialize the class with feature_id and start frame
     * @param {int} _feature_id
     * @param {int} _start_frame
     * @return {*}
     */    
    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    /**
     * @brief  we calc the number of feature in the slide window,
	 * 					 which only have used more than 2 times and start_frame smaller than (WINDOW_SIZE-2);
     */    
    int getFeatureCount();

    /**
     * @brief  to update list<FeaturePerId> feature;
	 * 					through clac the parallax between second new and third new feature;
	 * 					to judge whether claced parralx bigger than MIN_PARALLAX, then jude it whether be the keyFrame;
     * @param {int} frame_count
     * @param {constmap<int,vector<pair<int,Eigen::Matrix<double,7,1>>>>} &image
     * @param {double} td
     * @return {*}
     */    
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    /**
     * @brief  get correspondence feature between two image
     */    
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
	/**
 * @brief  FeaturePerId.estimated_depth = 1.0/x
 * @param x retrun 1.0/FeaturePerId.estimated_depth
 * @return {*}
 */
    void clearDepth(const VectorXd &x);
    /**
     * @brief  retrun 1.0/FeaturePerId.estimated_depth (得到逆深度向量)
     */    
    VectorXd getDepthVector();
    /**
     * @brief  using unscaled t to calc triangulate point which have not trangulated and get it depth for every id of feature
     * @param  Ps position of imu by median integrations
     * @param  tic trans from cam to imu(body) 
     * @param  ric rotate from cam to body
     * @return {*}
     */    
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;
    int last_track_num;

  private:
    /**
     * @brief  calc the parallax between second and last second newest un_pts for one feature_id
     * @param {constFeaturePerId} &it_per_id
     * @param {int} frame_count
     * @return {*}
     */    
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif