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

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    std::string filename = parseArgs(argc, (const char **)argv, "input", "sloth.png");
    int channels = -1;
    ColoredImageType img = loadImage(filename, channels);
    std::cout << "The image has " << channels << " channels." << std::endl;
    
    auto imgStack = imageChannelSplit(img, channels);

    saveSlice(imgStack, filename, "r", 0);
    saveSlice(imgStack, filename, "g", 1);
    saveSlice(imgStack, filename, "b", 2);

    exit(EXIT_SUCCESS);

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
    }

    if (filePath)
    {
      sFilename = filePath;
    }
    else
    {
      sFilename = "Lena.pgm";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
      std::cout << "convFilterNPP opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    }
    else
    {
      std::cout << "convFilterNPP unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }

    // Loop over the value of conv kernel, generate different results.
    for (int convValue = 1; convValue < 15; convValue++)
    {
      std::string sResultFilename = sFilename;

      std::string::size_type dot = sResultFilename.rfind('.');

      if (dot != std::string::npos)
      {
        sResultFilename = sResultFilename.substr(0, dot);
      }

      sResultFilename += "_convFilter_" + std::to_string(convValue) + ",1.pgm";

      if (checkCmdLineFlag(argc, (const char **)argv, "output"))
      {
        char *outputFilePath;
        getCmdLineArgumentString(argc, (const char **)argv, "output",
                                &outputFilePath);
        sResultFilename = outputFilePath;
      }

      // declare a host image object for an 8-bit grayscale image
      npp::ImageCPU_8u_C1 oHostSrc;
      // load gray-scale image from disk
      npp::loadImage(sFilename, oHostSrc);
      // declare a device image and copy construct from the host image,
      // i.e. upload host to device
      npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

      // Start the timer for recording processing times
      auto timerStart = std::chrono::high_resolution_clock::now();

      // create struct with box-filter mask size
      NppiSize oMaskSize = {5, 5};

      NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
      NppiPoint oSrcOffset = {0, 0};

      // create struct with ROI size
      NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
      // allocate device image of appropriately reduced size
      npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
      // set anchor point inside the mask to (oMaskSize.width / 2,
      // oMaskSize.height / 2) It should round down when odd
      NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};
     
      // Try to change these values to see the difference
      npp::ImageCPU_32s_C1 hostKernel(5,5);
      for(int x = 0 ; x < 5; x++){
          for(int y = 0 ; y < 5; y++){
              hostKernel.pixels(x,y)[0].x = convValue;
          }
      }
      // Set the middle pixel to be least weight.
      hostKernel.pixels(3,3)[0].x = 1;
      // Create device side kernel
      npp::ImageNPP_32s_C1 pKernel(hostKernel);

      // We now use Convolution fileter!
      NPP_CHECK_NPP(nppiFilter_8u_C1R(
          oDeviceSrc.data(), oDeviceSrc.pitch(),
          oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI,
          pKernel.data(),
          oMaskSize, oAnchor, 25));

      // declare a host image for the result
      npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
      // and copy the device result data into it
      oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

      auto timerStop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart);
      std::cout << "Iteration " << convValue << " takes " << duration.count() << " ms " ;

      saveImage(sResultFilename, oHostDst);
      std::cout << "Saved image: " << sResultFilename << std::endl;

      // If free them now, it will only function once.
      // nppiFree(oDeviceSrc.data());
      // nppiFree(oDeviceDst.data());
    }

    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  // Planned procedures:
  // 1. Check input param, find whether we provide the input image, otherwise using default image.
  // 2. Load the image using OpenCV, into Mat, let's assume the image is colored jpeg file.
  // 3. Split by channel, Mat into MatChannel[3], where the order is BGR
  // 3.1 Save the each channel into image file.
  // 4. for each channel, do FFT, into new cuda array
  // 4.1 convert cuda array to mat and save to image file.
  // 5. for each ffted channel, add watermark to the corners.
  // 5.1 save the changed ffted channel to image file.
  // 6. for each channel, do iFFT, into normal space.
  // 6.1 save iffted changed ffted channel to image file.
  // 7. Combine 3 channels into one, and save into RGB file.
  // 8. Compare initial image and watermarked image, pixel by pixel, show the max diff, avg diff. Estimate the result should be only a few digit off.

  // Functions needed:
  // Mat load_image(path)
  // void save_image(path, Mat), void save_image(path, Mat, 'r''g''b')
  // Mat array2Mat(cudaArray)
  // cudaArray mat2Array(Mat)

  return 0;
}
