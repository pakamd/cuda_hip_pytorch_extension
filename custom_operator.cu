#include <cuda.h>
#include <ATen/ATen.h>

__global__ void custom_operator_my(const float *input,
                                   float *output,
                                   const int height,
                                   const int width)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    output[y * width + x] = input[y * width + x] * 2;
}

at::Tensor my_custom_operator_mul2(at::Tensor input)
{
    auto output = at::zeros_like(input);
    // one pixel per block
    constexpr int pixel_per_block = 1;
    custom_operator_my<<<dim3(input.size(1) / pixel_per_block, input.size(0) / pixel_per_block), dim3(pixel_per_block, pixel_per_block)>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    return output;
}