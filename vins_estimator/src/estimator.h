#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>


class Estimator
{
  public:
    /**
     * @brief  when system begin, will set solver_flag INITIAL in func clearState()
     */    
    Estimator();

    /**
     * @brief set rotation of extrinsic for FeatureManager and td(initial value of time offset)
	 * 				  with	other factor unknown ProjectionFactor and  ProjectionTdFactor with 1.5 * focal_length Matrix2d
     * @param {*}
     * @return {*}
     */    
    void setParameter();

    // interface
    /**
     * @brief  for IMU data preintegration
     * @param {double} t
     * @param {constVector3d} &linear_acceleration
     * @param {constVector3d} &angular_velocity
     * @return {*}
     */    
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    /**
     * @brief  for visual SFM, calc v, s, g by align cam and imu data by using func visualInitialAlign()
     */    
    bool initialStructure();
    /**
     * @brief  first
     */    
    bool visualInitialAlign();
    /**
     * @brief to return whether we can start initialization ,
	 * 				by calc the parallax between one and the last feature, judge it whether is bigger than 30
     * @param {Matrix3d} &relative_R
     * @param {Vector3d} &relative_T
     * @param {int} &l
     * @return {*}
     */    
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
	/**
     * @brief  first triangulate all feature and update every id of feature depth using linear method in func triangulate(),
	 * 					and then optimize the p, v, q, ba, bg, inverse depth of feature in the func optimization();
     */
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    /**
     * @brief  optimize Rs, Ps, Vs, Bas, Bgs, inverse depth by ceres
     */    
    void optimization();
    /**
     * @brief  trans Eigen data:Ps, Rs, Vs, Bas, Bgs, td to double vector
     */    
    void vector2double();
    /**
     * @brief  trans double vector to Eigen data:Ps, Rs, Vs, Bas, Bgs, td
     */    
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0, //after get feather parallax enough (as key frame), or after initialization successfully
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM]; //extrinsic
    Vector3d tic[NUM_OF_CAM]; //extrinsic

	// optimized variable
    Vector3d Ps[(WINDOW_SIZE + 1)]; //position of median integration, after optimizing will change
    Vector3d Vs[(WINDOW_SIZE + 1)]; //velocity of median integtaion, after optimizing will change
    Matrix3d Rs[(WINDOW_SIZE + 1)]; //rotation of median integration, after optimizing will change
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0; //last_R/R0: last/first rotation in sliding window
    Vector3d back_P0, last_P, last_P0; //last_P/P0: last/first position in sliding window
    std_msgs::Header Headers[(WINDOW_SIZE + 1)]; //from img_msg->header in the slide window

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; //array of object to preintegration in the number of  WINDOW_SIZE+1
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count; //initialize 0, and will keep 0 in almost 50 times func processIMU() done then update
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager; //FeatureManager
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu; //initialize false
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame; //all the features data
    IntegrationBase *tmp_pre_integration;

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
