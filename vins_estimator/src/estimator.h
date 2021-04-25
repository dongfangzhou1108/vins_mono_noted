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
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
	/**
     * @brief step1：将new feature，加入FeatureManager，计算slide window倒数2，3帧之间的视差，确定边缘化方式．
	 * 				  step2：通过KLT计算基础矩阵E=t^R，恢复R，与IMU预积分量构成手眼标定问题，同时满足frame_count和奇异值足够小的条件，标定ric．
	 * 				  step3：如果solver_flag为INITIAL，视觉sfm + 视觉惯性对齐．
	 * 				  step4：
     */
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
	/**
     * @brief step1：选择滑动窗口中，与最后一帧的视差达到阈值的lth帧，作为世界坐标系，计算T_lth_last．
	 * 				  step2：sfm，实际维护的位姿是T_lth_kth，在三角化的过程中，我们需要T_kth_lth，通过多次三角化和PnP提供位姿初值，最后非线性优化BA．
	 * 								  TODO:ceres计算的3D点，存在(1)未优化的点(0,0,0)和(2)深度很大的outlier，没有排除这些点．
	 * 								  利用BA计算的3D点，通过PnP，确定all_image_frame的关键帧，计算all_image_frame所有帧的位姿．
	 * 				  step3：visual Inertial align：视觉惯导对齐．
     */
    bool initialStructure();
	/**
     * @brief step1：jacobian_bias * d_bg = 2 * delta_q，其中delta_q由pre_integration和frame_pose计算，优化bg，重新计算预积分量．
	 * 	　　　 step2：delta_alpha = 0，delta_beta = 0，其中delta_alpha和delta_beta由pre_integration和frame_pose计算，优化velo + gravity + scale．
	 * 								  其次，没有办法优化tic，因为delta_alpha中，tic的相关项：(R_IMUkth_IMUk+1th - I) * tic，该项很小，能观性差．
	 * 								  TODO:在线性化求解过程中，(1)Ax = b左右同时乘1000，(2)计算frame的delta_t时除100，为了增加数值稳定性？
	 * 							      TODO:没有优化ba，是否可以再优化v + g + s后，优化ba．
	 *                step3：仍基于delta_alpha = 0，delta_beta = 0，修正gravity，同时优化velo + scale．
	 * 								  思路是，通过上一步计算g的大致方向，因为g的大小固定，进行4次两自由度的切平面方向优化．
	 *                step4：更新状态，根据计算的scale，更新position + velocity + feature depth，调整yaw，使得重力方向竖直朝下，R0的yaw = 0．
	 * 								  TODO:为什么要保证feature的start_frame <= 7，即不使用最新三帧的feature进行状态估计？
     */
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
	/**
     * @brief step1：AddParameterBlock．
	 * 				  step2：AddResidualBlock：(1)MarginalizationFactor，(2)IMUFactor，(3)ProjectionFactor．
	 * 								 TODO:relocalization_info重定位．
	 * 				  step3：
     */
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
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
    vector<double *> last_marginalization_parameter_blocks; //滑动窗口中0-9帧pose + {velo + ba + bg} + 外参．

    map<double, ImageFrame> all_image_frame;
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
