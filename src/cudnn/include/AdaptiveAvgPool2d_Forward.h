#pragma once
#include "Utils.cuh"
#include <vector>
#include <cudnn.h>

using std::vector;

class AdaptiveAvgPool2d_Forward : public Profiler {
    public:
        cudnnHandle_t handle;
        bool is_UVM;

        cudnnTensorDescriptor_t input_descriptor;
        cudnnPoolingDescriptor_t avgpool_descriptor;
        cudnnTensorDescriptor_t output_descriptor;

        long input_n, input_c, input_h, input_w;
        int output_n, output_c, output_h, output_w;
        double input_ratio, output_ratio;

        long k_size_h, k_size_w;
        long stride_h, stride_w;
        long padding_h, padding_w;

        const float alpha = 1.f;
        const float beta = 0.f;

        float *input_data;
        float *output_data;

        AdaptiveAvgPool2d_Forward(cudnnHandle_t handle, vector<double> &args, bool is_UVM);
        ~AdaptiveAvgPool2d_Forward();
        float Run();
};
