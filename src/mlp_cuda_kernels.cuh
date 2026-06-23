#pragma once

#include <cuda_runtime.h>

#ifndef CUDA_INPUT_DIM
#define CUDA_INPUT_DIM 4096
#endif
#ifndef CUDA_HIDDEN1
#define CUDA_HIDDEN1 64
#endif
#ifndef CUDA_HIDDEN2
#define CUDA_HIDDEN2 32
#endif
#ifndef CUDA_OUTPUT_DIM
#define CUDA_OUTPUT_DIM 1
#endif

__global__ void matmul_kernel(const float* input, const float* weights, float* output, int batch_size, int in_dim, int out_dim);
__global__ void bias_relu_kernel(float* values, const float* bias, int total_values, int units);
__global__ void bias_sigmoid_kernel(float* values, const float* bias, int total_values, int units);
__global__ void binary_cross_entropy_kernel(const float* predictions, const float* labels, float* losses, int total_values);
__global__ void output_gradient_kernel(const float* predictions, const float* labels, float* gradients, int total_values);
__global__ void hidden_gradient_kernel(const float* next_gradient, const float* next_weights, const float* activation, float* gradient, int batch_size, int units, int next_units);
__global__ void weight_gradient_kernel(const float* input, const float* gradient, float* weight_gradient, int batch_size, int in_dim, int out_dim);
__global__ void bias_gradient_kernel(const float* gradient, float* bias_gradient, int batch_size, int units);
__global__ void sgd_update_kernel(float* parameters, const float* gradients, float learning_rate, int total_values);

typedef struct { float* w1; float* b1; float* w2; float* b2; float* w3; float* b3; } CudaMlpWeights;
typedef struct { float* a1; float* a2; float* output; } CudaMlpActivations;
typedef struct {
    float* d1; float* d2; float* d3; float* losses;
    float* gw1; float* gb1; float* gw2; float* gb2; float* gw3; float* gb3;
} CudaMlpGradients;

void cuda_forward_pass(const float* d_input, const CudaMlpWeights* weights, CudaMlpActivations* activations, int batch_size);
void cuda_backward_update(const float* d_input, const float* d_labels, CudaMlpWeights* weights, const CudaMlpActivations* activations, CudaMlpGradients* gradients, int batch_size, float learning_rate);
