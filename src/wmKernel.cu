#include "wmKernel.cuh"

__global__ void applyKernel()
{

}

void applyKernelToImgAsync()
{
    applyKernel<<<1,1>>>();
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

