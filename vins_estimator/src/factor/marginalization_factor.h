#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    // 通过设定的factor的CostFunction，计算raw_jacobians数组 + jacobians + 残差 Eigen向量．
	// 关于LossFunction的处理参见ceres官方文档．
    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks; // 各个factors的参数块
    std::vector<int> drop_set; // 边缘化参数在_parameter_blocks中的idx

    double **raw_jacobians; // jacobians数组
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

/**
 * @brief 边缘化目的：删除滑动窗口状态量时，防止减少约束以及丢失信息．
 */
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
	// 构建factors信息，将各factors的parameter的地址作为key，更新parameter_block_size和 parameter_block_idx．
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
	// ResidualBlockInfo::Evaluate()，维护parameter_block_data．
    void preMarginalize();
	// 计算marg变量和保留变量的维度，分成四个线程，计算当前所有状态量的H=J.trans()*J矩阵和J.trans()*residual．
	// 从H矩阵恢复保留变量的jacobian和residual．
    void marginalize();
	// 为keep_block系列变量赋值，last_marginalization_parameter_blocks赋值．
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;
    int m, n; // m：marg size，n：保留变量size．
    std::unordered_map<long, int> parameter_block_size; //global size : 键值对，键是factor的parameter地址，值是parameter占据内存大小 ．
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size : 在addResidualBlockInfo时，键是drop_set参数地址，值是0．marginalize时记录parameter idx．
    std::unordered_map<long, double *> parameter_block_data; //键值对，键是factor的parameter地址，值是parameter数组．

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
