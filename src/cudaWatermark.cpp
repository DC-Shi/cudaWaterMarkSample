/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif
#include <opencv2/imgcodecs.hpp>
#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

#include <chrono> // Add timing
#include "fileio.h"
#include <cufft.h>


cufftComplex * convertImgToBytes(ColoredImageType grayImage);
ColoredImageType convertBytesToImg(uint8_t* grayArray, const int width, const int height);
ColoredImageType convertBytesToImg(cufftComplex* grayArray, const int width, const int height);

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    std::vector<std::string> imgNames{"color.png", "sloth.png"};

    for (auto imgName : imgNames)
    {
      std::cout << "==== Dealing with image " << imgName << "====" << std::endl;
      // Step 1: get the filename and load into image.
      std::string filename = parseArgs(argc, (const char **)argv, "input", imgName);
      int channels = -1;
      ColoredImageType img = loadImage(filename, channels);
      int width = FreeImage_GetWidth(img);
      int height = FreeImage_GetHeight(img);
      std::cerr << "INFO: The image dimension is " <<
          width << "x" << height << "x" << channels <<
          " (W*H*C)." << std::endl;

      // Step 2: split the image into channels, save for each file.
      auto imgStack = imageChannelSplit(img, channels);

      saveSlice(imgStack, filename, "r", 0);
      saveSlice(imgStack, filename, "g", 1);
      saveSlice(imgStack, filename, "b", 2);

      // Step 3: make the data onto GPU
      // Image that stores on CPU
      // float *imgR, *imgG, *imgB;
      cufftComplex *devImgR, *devImgG, *devImgB;
      devImgR = convertImgToBytes(imgStack[0]);
      devImgG = convertImgToBytes(imgStack[1]);
      devImgB = convertImgToBytes(imgStack[2]);

      printf("Debugprint: devImgR, (%f, %fi), (%f, %fi)\n", 
      devImgR[0].x, devImgR[0].y, 
      devImgR[1].x, devImgR[1].y
      );

      checkCudaErrors( cudaGetLastError() );

      // Step 4: Make FFT
      cufftHandle fftPlan;
      cufftResult stat;
      cufftPlan2d(&fftPlan, width, height, CUFFT_C2C);
      stat = cufftExecC2C(fftPlan, devImgR, devImgR, CUFFT_FORWARD);
      stat = cufftExecC2C(fftPlan, devImgG, devImgG, CUFFT_FORWARD);
      stat = cufftExecC2C(fftPlan, devImgB, devImgB, CUFFT_FORWARD);
      if (stat != CUFFT_SUCCESS) {
          printf("cufftExecR2C3 error %d\n",stat);
          return 1;
      }
      // 4.1 Save the FFT's result, real part.
      // Need to wait for FFT to be completed
      cudaDeviceSynchronize();

      std::cerr << "DEBUG: devImgR: " << devImgR << std::endl;

        printf("Debugprint: devImgR, (%f, %fi), (%f, %fi)\n", 
        devImgR[0].x, devImgR[0].y, 
        devImgR[1].x, devImgR[1].y
        );

      std::cout << "done fft forward" << std::endl;
        std::string fftFilename = filename;
        fftFilename.insert(filename.length()-4, "_fft");
      std::cout << "done name forward" << std::endl;
        ColoredImageType fftR = convertBytesToImg(devImgR, width, height);
      std::cout << "done fftr forward" << std::endl;
        ColoredImageType fftG = convertBytesToImg(devImgG, width, height);
      std::cout << "done fftg forward" << std::endl;
        ColoredImageType fftB = convertBytesToImg(devImgB, width, height);
      std::cout << "done fftb forward" << std::endl;
      GrayscaleImageStack fftImage {fftR, fftG, fftB};
      saveSlice(fftImage, fftFilename, "r", 0);
      saveSlice(fftImage, fftFilename, "g", 1);
      saveSlice(fftImage, fftFilename, "b", 2);

  // 5. for each ffted channel, add watermark to the corners.
  // 5.1 save the changed ffted channel to image file.
  // 6. for each channel, do iFFT, into normal space.
  // 6.1 save iffted changed ffted channel to image file.
  // 7. Combine 3 channels into one, and save into RGB file.
  // 8. Compare initial image and watermarked image, pixel by pixel, show the max diff, avg diff. Estimate the result should be only a few digit off.

      // Free resources
      FreeImage_Unload(img);
      for (auto imgC : imgStack)
        FreeImage_Unload(imgC);
      FreeImage_Unload(fftR);
      FreeImage_Unload(fftG);
      FreeImage_Unload(fftB);
      // free(imgR);
      // free(imgG);
      // free(imgB);
      cudaFree(devImgR);
      cudaFree(devImgG);
      cudaFree(devImgB);
      // cudaFree(devCImgR);
      // cudaFree(devCImgG);
      // cudaFree(devCImgB);
      cufftDestroy(fftPlan);
    }
  }
  catch (...)
  {
    std::cerr << "ERR: Program error! An unknow type of exception occurred. \n";
    std::cerr << "ERR: Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}


cufftComplex * convertImgToBytes(ColoredImageType grayImage)
{
    BYTE* bits = FreeImage_GetBits(grayImage);
    int width = FreeImage_GetWidth(grayImage);
    int height = FreeImage_GetHeight(grayImage);
    int pitch = FreeImage_GetPitch(grayImage);
    cufftComplex* imageArray;
    cudaMallocManaged(&imageArray, width * height * sizeof(cufftComplex));
    for (int y = 0; y < height; y++)
    {
        BYTE* pixel = (BYTE*)bits + y * pitch;
        for (int x = 0; x < width; x++)
        {
            // We write the result to column-major array
            imageArray[x * height + y].x = pixel[x];
            imageArray[x * height + y].y = 0;
        }
    }

    return imageArray;
}


ColoredImageType convertBytesToImg(float* grayArray, const int width, const int height)
{
    // Create a new 8-bit grayscale image from column-major array.
    FIBITMAP* grayImage = FreeImage_Allocate(width, height, 8);
    BYTE* bits = FreeImage_GetBits(grayImage);
    int pitch = FreeImage_GetPitch(grayImage);

    float minValue = 100;
    float maxValue = -100;

    // Set pixel values from imageArray
    for (int y = 0; y < height; y++) {
        BYTE* pixel = (BYTE*)bits + y * pitch;
        for (int x = 0; x < width; x++) {
            float value = grayArray[x * height + y];
            pixel[x] = (uint8_t)value;

            if (minValue > value) minValue = value;
            if (maxValue < value) maxValue = value;
        }
    }

    std::cerr << "DEBUG: convertBytesToImg, Complex, min value: " <<
        minValue << " , max value: " << maxValue << std::endl;

    return grayImage;
}


ColoredImageType convertBytesToImg(cufftComplex* grayArray, const int width, const int height)
{
    // Create a new 8-bit grayscale image from column-major array.
    FIBITMAP* grayImage = FreeImage_Allocate(width, height, 8);
    BYTE* bits = FreeImage_GetBits(grayImage);
    int pitch = FreeImage_GetPitch(grayImage);

    float minValue = 100;
    float maxValue = -100;
    std::cerr << "DEBUG: convertBytesToImg, Complex, ptr: " <<
        grayArray << " , pitch: " << pitch << " , width: " << width << " , height: " << height << std::endl;

    // Set pixel values from imageArray
    for (int y = 0; y < height; y++) {
        BYTE* pixel = (BYTE*)bits + y * pitch;
        for (int x = 0; x < width; x++) {
          // printf("(%d, %d)\n", x, y);
            float value = grayArray[x * height + y].x;
          // printf("(%d, %d)\n", x, y);
            pixel[x] = (uint8_t)value;
          // printf("(%d, %d)\n", x, y);

            if (minValue > value) minValue = value;
            if (maxValue < value) maxValue = value;
        }
    }

    std::cerr << "DEBUG: convertBytesToImg, Complex, min value: " <<
        minValue << " , max value: " << maxValue << std::endl;

    return grayImage;
}

