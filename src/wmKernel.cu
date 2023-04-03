#include "wmKernel.cuh"

__global__ void applyKernel(cufftComplex* array, int width, int height)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < width*height)
    {
        // This is column-major
        int y = idx % height;
        int x = idx / height;
        // draw a line
        if (100 <= x && x <= 120)
            if (40 <= y && y <= 180)
            {
                array[idx].x = 255;
            }

        // draw a circle, the (0,0) is left-bottom
        float center1X = 80;
        float center1Y = 80;
        float distance1Sq = (x-center1X)*(x-center1X) + (y-center1Y)*(y-center1Y);

        if (400 <= distance1Sq && distance1Sq <= 1600)
        {
            array[idx].x = 255;
        }

        // draw another circle, the (0,0) is left-bottom
        float center2X = 180;
        float center2Y = 80;
        float distance2Sq = (x-center2X)*(x-center2X) + (y-center2Y)*(y-center2Y);

        if (400 <= distance2Sq && distance2Sq <= 1600)
        {
            float cosTheta = (x-center2X) / sqrt(distance2Sq);
            if (cosTheta < 0.8)
                array[idx].x = 255;
        }

        // draw a line
        if (240 <= x && x <= 320)
            if (70 <= y && y <= 90)
            {
                array[idx].x = 255;
            }

        // draw another circle, the (0,0) is left-bottom
        float center3X = 380;
        float center3Y = 80;
        float distance3Sq = (x-center3X)*(x-center3X) + (y-center3Y)*(y-center3Y);

        if (400 <= distance3Sq && distance3Sq <= 1600)
        {
            float cosTheta = (x-center3X) / sqrt(distance3Sq);
            float sinTheta = (y-center3Y) / sqrt(distance3Sq);
            if (cosTheta < 0.7 && sinTheta >= 0)
                array[idx].x = 255;
            if (cosTheta > -0.7 && sinTheta <= 0)
                array[idx].x = 255;
            if (340 <= x && x <= 420)
                if (70 <= y && y <= 90)
                {
                    array[idx].x = 255;
                }
        }
        if (350 <= x && x <= 410)
            if (70 <= y && y <= 90)
            {
                array[idx].x = 255;
            }
    }
}

void applyKernelToImgAsync(cufftComplex* array, int width, int height)
{
    size_t threadPerBlock = 1024;
    size_t blocks = (width*height + threadPerBlock - 1)/threadPerBlock;
    applyKernel<<<blocks, threadPerBlock>>>(array, width, height);
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
            FreeImage_GetPixelColor(b, x, y, &pixelColorB);

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