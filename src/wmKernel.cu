#include "wmKernel.cuh"

__global__ void applyKernel()
{

}

void applyKernelToImgAsync()
{
    applyKernel<<<1,1>>>();
}




__global__ void scaleElement(cufftComplex* array, int width, int height, float factor)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < width * height)
    {
        array[idx].x /= factor;
        array[idx].y /= factor;
    }
}
void scaleComplexAsync(cufftComplex* array, int width, int height, float factor)
{
    size_t threadPerBlock = 1024;
    size_t blocks = (width*height + threadPerBlock - 1)/threadPerBlock;
    scaleElement<<<blocks, threadPerBlock>>>(array, width, height, factor);
}