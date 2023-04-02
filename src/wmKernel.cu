#include "wmKernel.cuh"

__global__ void applyKernel()
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0)
    {
        printf("Dummy kernel not implemented\n");
    }
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


void compareTwoImg(ColoredImageType a, ColoredImageType b)
{
    std::cout << "NOTIMPLEMENTEDCOMPARE" << std::endl;

    int widthA = FreeImage_GetWidth(a);
    int heightA = FreeImage_GetHeight(a);
    int widthB = FreeImage_GetWidth(b);
    int heightB = FreeImage_GetHeight(b);

    if ((widthA != widthB) || (heightA != heightB))
    {
        std::cout << "Compare result: Two image does not match in size." << std::endl;
        return;
    }
    
    float totalErr = 0;
    float maxErr = 0;

    for (int y = 0; y < heightA; y++)
    {
        for (int x = 0; x < widthA; x++)
        {
            RGBQUAD pixelColorA;
            FreeImage_GetPixelColor(a, x, y, &pixelColorA);
            RGBQUAD pixelColorB;
            FreeImage_GetPixelColor(a, x, y, &pixelColorB);

            auto diffR = abs((float)pixelColorA.rgbRed - pixelColorB.rgbRed);
            auto diffG = abs((float)pixelColorA.rgbGreen - pixelColorB.rgbGreen);
            auto diffB = abs((float)pixelColorA.rgbBlue - pixelColorB.rgbBlue);
            totalErr += diffR;
            totalErr += diffG;
            totalErr += diffB;

            maxErr = max(maxErr, (float)diffR);
            maxErr = max(maxErr, (float)diffG);
            maxErr = max(maxErr, (float)diffB);
        }
    }

    std::cout << "Compare result: Two image max diff " << maxErr << " , average err per pixel " << totalErr/heightA/widthA << std::endl;

}