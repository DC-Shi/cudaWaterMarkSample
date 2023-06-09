#ifndef WATERMARK_KERNEL
#define WATERMARK_KERNEL


#include <cufft.h>
#include "fileio.h"
#include <iostream>

void applyKernelToImgAsync(cufftComplex* array, int width, int height);

void scaleComplexAsync(cufftComplex* array, int width, int height, float factor);
void compareTwoImg(ColoredImageType a, ColoredImageType b);

#endif