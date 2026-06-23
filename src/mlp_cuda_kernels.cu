#include "mlp_cuda_kernels.cuh"

#include <math.h>

#define CUDA_BLOCK_SIZE 256
#define BLOCKS(total) (((total) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE)

__global__ void matmul_kernel(const float* input, const float* weights, float* output, int batch_size, int in_dim, int out_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_dim) return;
    int row = index / out_dim;
    int col = index % out_dim;
    float sum = 0.0f;
    for (int k = 0; k < in_dim; ++k) sum += input[row * in_dim + k] * weights[k * out_dim + col];
    output[index] = sum;
}

__global__ void bias_relu_kernel(float* values, const float* bias, int total_values, int units) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_values) {
        float value = values[index] + bias[index % units];
        values[index] = value > 0.0f ? value : 0.0f;
    }
}

__global__ void bias_sigmoid_kernel(float* values, const float* bias, int total_values, int units) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_values) values[index] = 1.0f / (1.0f + expf(-(values[index] + bias[index % units])));
}

__global__ void binary_cross_entropy_kernel(const float* predictions, const float* labels, float* losses, int total_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_values) {
        float p = fminf(fmaxf(predictions[index], 1.0e-7f), 1.0f - 1.0e-7f);
        losses[index] = -(labels[index] * logf(p) + (1.0f - labels[index]) * logf(1.0f - p));
    }
}

__global__ void output_gradient_kernel(const float* predictions, const float* labels, float* gradients, int total_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_values) gradients[index] = predictions[index] - labels[index];
}

__global__ void hidden_gradient_kernel(const float* next_gradient, const float* next_weights, const float* activation, float* gradient, int batch_size, int units, int next_units) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch_size * units) {
        int row = index / units;
        int unit = index % units;
        float sum = 0.0f;
        for (int next = 0; next < next_units; ++next) sum += next_gradient[row * next_units + next] * next_weights[unit * next_units + next];
        gradient[index] = activation[index] > 0.0f ? sum : 0.0f;
    }
}

__global__ void weight_gradient_kernel(const float* input, const float* gradient, float* weight_gradient, int batch_size, int in_dim, int out_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < in_dim * out_dim) {
        int in = index / out_dim;
        int out = index % out_dim;
        float sum = 0.0f;
        for (int row = 0; row < batch_size; ++row) sum += input[row * in_dim + in] * gradient[row * out_dim + out];
        weight_gradient[index] = sum / (float)batch_size;
    }
}

__global__ void bias_gradient_kernel(const float* gradient, float* bias_gradient, int batch_size, int units) {
    int unit = blockIdx.x * blockDim.x + threadIdx.x;
    if (unit < units) {
        float sum = 0.0f;
        for (int row = 0; row < batch_size; ++row) sum += gradient[row * units + unit];
        bias_gradient[unit] = sum / (float)batch_size;
    }
}

__global__ void sgd_update_kernel(float* parameters, const float* gradients, float learning_rate, int total_values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_values) parameters[index] -= learning_rate * gradients[index];
}

void cuda_forward_pass(const float* input, const CudaMlpWeights* w, CudaMlpActivations* a, int batch) {
    matmul_kernel<<<BLOCKS(batch * CUDA_HIDDEN1), CUDA_BLOCK_SIZE>>>(input, w->w1, a->a1, batch, CUDA_INPUT_DIM, CUDA_HIDDEN1);
    bias_relu_kernel<<<BLOCKS(batch * CUDA_HIDDEN1), CUDA_BLOCK_SIZE>>>(a->a1, w->b1, batch * CUDA_HIDDEN1, CUDA_HIDDEN1);
    matmul_kernel<<<BLOCKS(batch * CUDA_HIDDEN2), CUDA_BLOCK_SIZE>>>(a->a1, w->w2, a->a2, batch, CUDA_HIDDEN1, CUDA_HIDDEN2);
    bias_relu_kernel<<<BLOCKS(batch * CUDA_HIDDEN2), CUDA_BLOCK_SIZE>>>(a->a2, w->b2, batch * CUDA_HIDDEN2, CUDA_HIDDEN2);
    matmul_kernel<<<BLOCKS(batch), CUDA_BLOCK_SIZE>>>(a->a2, w->w3, a->output, batch, CUDA_HIDDEN2, CUDA_OUTPUT_DIM);
    bias_sigmoid_kernel<<<BLOCKS(batch), CUDA_BLOCK_SIZE>>>(a->output, w->b3, batch, CUDA_OUTPUT_DIM);
}

void cuda_backward_update(const float* input, const float* labels, CudaMlpWeights* w, const CudaMlpActivations* a, CudaMlpGradients* g, int batch, float lr) {
    output_gradient_kernel<<<BLOCKS(batch), CUDA_BLOCK_SIZE>>>(a->output, labels, g->d3, batch);
    hidden_gradient_kernel<<<BLOCKS(batch * CUDA_HIDDEN2), CUDA_BLOCK_SIZE>>>(g->d3, w->w3, a->a2, g->d2, batch, CUDA_HIDDEN2, CUDA_OUTPUT_DIM);
    hidden_gradient_kernel<<<BLOCKS(batch * CUDA_HIDDEN1), CUDA_BLOCK_SIZE>>>(g->d2, w->w2, a->a1, g->d1, batch, CUDA_HIDDEN1, CUDA_HIDDEN2);
    weight_gradient_kernel<<<BLOCKS(CUDA_HIDDEN2), CUDA_BLOCK_SIZE>>>(a->a2, g->d3, g->gw3, batch, CUDA_HIDDEN2, CUDA_OUTPUT_DIM);
    bias_gradient_kernel<<<BLOCKS(CUDA_OUTPUT_DIM), CUDA_BLOCK_SIZE>>>(g->d3, g->gb3, batch, CUDA_OUTPUT_DIM);
    weight_gradient_kernel<<<BLOCKS(CUDA_HIDDEN1 * CUDA_HIDDEN2), CUDA_BLOCK_SIZE>>>(a->a1, g->d2, g->gw2, batch, CUDA_HIDDEN1, CUDA_HIDDEN2);
    bias_gradient_kernel<<<BLOCKS(CUDA_HIDDEN2), CUDA_BLOCK_SIZE>>>(g->d2, g->gb2, batch, CUDA_HIDDEN2);
    weight_gradient_kernel<<<BLOCKS(CUDA_INPUT_DIM * CUDA_HIDDEN1), CUDA_BLOCK_SIZE>>>(input, g->d1, g->gw1, batch, CUDA_INPUT_DIM, CUDA_HIDDEN1);
    bias_gradient_kernel<<<BLOCKS(CUDA_HIDDEN1), CUDA_BLOCK_SIZE>>>(g->d1, g->gb1, batch, CUDA_HIDDEN1);
    sgd_update_kernel<<<BLOCKS(CUDA_INPUT_DIM * CUDA_HIDDEN1), CUDA_BLOCK_SIZE>>>(w->w1, g->gw1, lr, CUDA_INPUT_DIM * CUDA_HIDDEN1);
    sgd_update_kernel<<<BLOCKS(CUDA_HIDDEN1), CUDA_BLOCK_SIZE>>>(w->b1, g->gb1, lr, CUDA_HIDDEN1);
    sgd_update_kernel<<<BLOCKS(CUDA_HIDDEN1 * CUDA_HIDDEN2), CUDA_BLOCK_SIZE>>>(w->w2, g->gw2, lr, CUDA_HIDDEN1 * CUDA_HIDDEN2);
    sgd_update_kernel<<<BLOCKS(CUDA_HIDDEN2), CUDA_BLOCK_SIZE>>>(w->b2, g->gb2, lr, CUDA_HIDDEN2);
    sgd_update_kernel<<<BLOCKS(CUDA_HIDDEN2), CUDA_BLOCK_SIZE>>>(w->w3, g->gw3, lr, CUDA_HIDDEN2);
    sgd_update_kernel<<<BLOCKS(CUDA_OUTPUT_DIM), CUDA_BLOCK_SIZE>>>(w->b3, g->gb3, lr, CUDA_OUTPUT_DIM);
}
