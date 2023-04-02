#ifndef WATERMARK_KERNEL
#define WATERMARK_KERNEL


#include <cufft.h>
#include "fileio.h"
#include <iostream>

__global__ void applyKernel();
void applyKernelToImgAsync();

cufftComplex * convertImgToBytes(ColoredImageType grayImage);
ColoredImageType convertBytesToImg(uint8_t* grayArray, const int width, const int height);
ColoredImageType convertBytesToImg(cufftComplex* grayArray, const int width, const int height);

#endif