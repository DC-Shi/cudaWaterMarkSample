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
#include "wmKernel.cuh"



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
          printf("cufftExecC2C forward error %d\n",stat);
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

      saveImage(devImgR, devImgG, devImgB, width, height, filename, "fft", true);

      // 5. for each ffted channel, add watermark to the corners.
      applyKernel();
      // 5.1 save the changed ffted channel to image file.
      saveImage(devImgR, devImgG, devImgB, width, height, filename, "wmfft", true);
      // 6. for each channel, do iFFT, into normal space.
      stat = cufftExecC2C(fftPlan, devImgR, devImgR, CUFFT_INVERSE);
      stat = cufftExecC2C(fftPlan, devImgG, devImgG, CUFFT_INVERSE);
      stat = cufftExecC2C(fftPlan, devImgB, devImgB, CUFFT_INVERSE);
      if (stat != CUFFT_SUCCESS) {
          printf("cufftExecC2C back error %d\n",stat);
          return 1;
      }
      
      // 6.1 save iffted changed ffted channel to image file.
      // Need to wait for FFT to be completed
      scaleComplexAsync(devImgR, width, height, width*height);
      scaleComplexAsync(devImgG, width, height, width*height);
      scaleComplexAsync(devImgB, width, height, width*height);
      cudaDeviceSynchronize();
      saveImage(devImgR, devImgG, devImgB, width, height, filename, "wm_ifft", true);

      
      std::cerr << "DEBUG: devImgR: " << devImgR << std::endl;

        printf("Debugprint: devImgR, (%f, %fi), (%f, %fi)\n", 
        devImgR[0].x, devImgR[0].y, 
        devImgR[1].x, devImgR[1].y
        );
  // 7. Combine 3 channels into one, and save into RGB file.
  // 8. Compare initial image and watermarked image, pixel by pixel, show the max diff, avg diff. Estimate the result should be only a few digit off.

      // Free resources
      FreeImage_Unload(img);
      for (auto imgC : imgStack)
        FreeImage_Unload(imgC);
        
      checkCudaErrors( cudaFree(devImgR) );
      checkCudaErrors( cudaFree(devImgG) );
      checkCudaErrors( cudaFree(devImgB) );
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
