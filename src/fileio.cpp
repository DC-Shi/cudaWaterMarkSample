#include "fileio.h"
#include <iostream>
#include <helper_string.h>
#include <cuda_runtime.h>

/// @brief Parse from commandline
/// @param argc How many arguments
/// @param argv Arguments array
/// @param option Which option should be parsed
/// @param defaultValue The default value for this option, empty means the commandline must provide this option to continue
/// @return 
std::string parseArgs(const int argc, const char ** argv, const char * option, const std::string defaultValue)
{
    std::string retFilename;
    char *filePath;

    if (checkCmdLineFlag(argc, (const char **)argv, option))
    {
        getCmdLineArgumentString(argc, (const char **)argv, option, &filePath);
    }
    else
    {
        if (defaultValue.empty())
        {
            std::cerr << "ERR: You must provide value of the option <-" << option
                << "> to continue the program, exit." << std::endl;
            exit(EXIT_FAILURE);

        }

        std::cerr << "INFO: Searching for " << defaultValue <<
            " for this option <" << option << "> in folders." <<std::endl;
        filePath = sdkFindFilePath(defaultValue.data(), argv[0]);
    }

    if (filePath)
    {
        retFilename = filePath;
    }
    else
    {
        retFilename = defaultValue;
    }

    // Try to open the file to see whether it is good.
    int file_errors = 0;
    std::ifstream infile(retFilename.data(), std::ifstream::in);

    if (infile.good())
    {
        std::cerr << "INFO: File opened: <" << retFilename.data()
            << "> successfully!" << std::endl;
        file_errors = 0;
        infile.close();
    }
    else
    {
        std::cerr << "ERR: unable to open: <" << retFilename.data() << ">"
                << std::endl;
        file_errors++;
        infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }
    
    return retFilename;
}

/// @brief Load the image from file path
/// @param imgPath Image file path
/// @param channels How many channels does this image have
/// @return The image object
ColoredImageType loadImage(const std::string imgPath, int& channels)
{
    // Decide format upon filename
    FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(imgPath.data());
    if (!fif)
    {
        std::cerr << "ERR: FreeImage: unknown format" << imgPath << std::endl;
        exit(EXIT_FAILURE);
    }
    // Check whether it can support read from such file.
    if (!FreeImage_FIFSupportsReading(fif))
    {
        std::cerr << "ERR: FreeImage: not supported format <" << fif << ">" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Load image with given format
    ColoredImageType inputImage = FreeImage_Load(fif, imgPath.data(), PNG_DEFAULT);
    if (!inputImage)
    {
        std::cerr << "ERR: FreeImage: failed to load image: " << imgPath << std::endl;
        exit(EXIT_FAILURE);
    }

    // Calculate type and bpp to get how many channels in this image.
    FREE_IMAGE_TYPE internalType = FreeImage_GetImageType(inputImage);
    auto   m_bpp = FreeImage_GetBPP(inputImage);

    std::cerr << "INFO: " << imgPath << " has type:" << internalType << std::endl;
    if(internalType == FIT_BITMAP)
    {
        //standard bitmap
        if(m_bpp == 8)
            channels = 1;
        else if(m_bpp == 24)
            channels = 3;
        else if(m_bpp == 32)
            channels = 4;    
        else
            {}
    }

    return inputImage;
}


/// @brief Save image to file
/// @param img The image ready to save
/// @param imgPath Image file path
/// @return Whether the save is successful
bool saveImage(ColoredImageType img, const std::string imgPath)
{
    return FreeImage_Save(FIF_PNG, img, imgPath.data());
}


/// @brief Save one slice of the image, the grayscale, into particular colored images.
/// @param img The grayscale image ready to save
/// @param imgPath Image file path
/// @param layer Which layer does this belongs to, should be 'r', 'g', or 'b'
/// @param idx Which layer to be saved in the grayscale stack
/// @return Whether the save is successful
bool saveSlice(GrayscaleImageStack img, const std::string imgPath, const std::string channel, const int idx)
{
    std::string filename = imgPath;
    filename.insert(filename.length()-4, "_"+channel);

    auto grayImg = img[idx];
    int width = FreeImage_GetWidth(grayImg);
    int height = FreeImage_GetHeight(grayImg);
    int pitch = FreeImage_GetPitch(grayImg);
    std::cerr << "INFO: saveSlice: " << width << "x" << height <<
        " into: " << filename << std::endl;

    // ColoredImageType rgbImage = FreeImage_ConvertTo24Bits(img[idx]);
    ColoredImageType rgbImage = FreeImage_Allocate(width, height, 24);
    if (channel == "r")
    {
        FreeImage_SetChannel(rgbImage, grayImg, FICC_RED);
    }
    else if (channel == "g")
    {
        FreeImage_SetChannel(rgbImage, grayImg, FICC_GREEN);
    }
    else if (channel == "b")
    {
        FreeImage_SetChannel(rgbImage, grayImg, FICC_BLUE);
    }
    else
    {
        std::cerr << "ERR: FreeImage: Cannot recognize channel <" << channel << ">" << std::endl;
        return false;
    }

    bool result = saveImage(rgbImage, filename);

    // Free the image locally allocated.
    FreeImage_Unload(rgbImage);

    return result;
}

/// @brief Split the image by channels and save it to an array
/// @param img The image object that has 1 or more channels
/// @param channels The channels this image has
/// @return Splited grayscale image
GrayscaleImageStack imageChannelSplit(const ColoredImageType img, const int channels)
{    
    GrayscaleImageStack ret; // = new GrayscaleImageStack();
    for (int i = 1; i <= channels; i++)
        ret.push_back(FreeImage_GetChannel(img, static_cast<FREE_IMAGE_COLOR_CHANNEL>(i)));
    
    return ret;
}

/// @brief Merge image with multiple grayscale images
/// @param imgs Grayscale images array
/// @param channels How many channels does this image have
/// @return The merged color image
ColoredImageType imageChannelMerge(const GrayscaleImageStack imgs, const int channels)
{
    int width = FreeImage_GetWidth(imgs[0]);
    int height = FreeImage_GetHeight(imgs[0]);
    ColoredImageType rgbImage = FreeImage_Allocate(width, height, 24);
    if (channels <= 2)
    {
        FreeImage_SetChannel(rgbImage, imgs[1], FICC_GREEN);
    }
    if (channels <= 3)
    {
        FreeImage_SetChannel(rgbImage, imgs[2], FICC_BLUE);
    }
    if (channels <= 1)
    {
        FreeImage_SetChannel(rgbImage, imgs[0], FICC_RED);
    }

    return rgbImage;
}



/// @brief Save cufftComplex to file
/// @param devImgR The complex array in red channel to be saved
/// @param devImgG The complex array in green channel to be saved
/// @param devImgB The complex array in blue channel to be saved
/// @param width Image width
/// @param height Image height
/// @param imgPath Image filename
/// @param suffix suffix of the filename
/// @param real Whether we save real part or the imaginary part, default is the real part
/// @return Whether the save is successful
bool saveImage(cufftComplex* devImgR, cufftComplex* devImgG, cufftComplex* devImgB,
    const int width, const int height, 
    const std::string imgPath, const std::string suffix, const bool real)
{    
    std::string fftFilename = imgPath;
    fftFilename.insert(imgPath.length()-4, "_"+suffix);
    ColoredImageType tmpfftR = convertBytesToImg(devImgR, width, height, real);
    ColoredImageType tmpfftG = convertBytesToImg(devImgG, width, height, real);
    ColoredImageType tmpfftB = convertBytesToImg(devImgB, width, height, real);
    GrayscaleImageStack fftImage {tmpfftR, tmpfftG, tmpfftB};
    saveSlice(fftImage, fftFilename, "r", 0);
    saveSlice(fftImage, fftFilename, "g", 1);
    saveSlice(fftImage, fftFilename, "b", 2);

    // Free the image locally generated.
    for (auto imgC : fftImage)
        FreeImage_Unload(imgC);
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


ColoredImageType convertBytesToImg(cufftComplex* grayArray, const int width, const int height, const bool real)
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
          float value;
          if (real)
            value = grayArray[x * height + y].x;
          else
            value = grayArray[x * height + y].y;
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

