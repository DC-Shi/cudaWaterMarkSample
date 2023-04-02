#ifndef WATERMARK_KERNEL
#define WATERMARK_KERNEL


#include <cufft.h>
#include "fileio.h"
#include <iostream>

__global__ void applyKernel();
void applyKernelToImgAsync();

void scaleComplexAsync(cufftComplex* array, int width, int height, float factor);

#endif