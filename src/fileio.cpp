#include "fileio.h"
#include <iostream>
#include <helper_string.h>

/// @brief Parse from commandline
/// @param argc How many arguments
/// @param argv Arguments array
/// @param option Which option should be parsed
/// @param defaultValue The default value for this option, NULL means the commandline must provide this option to continue
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
            std::cerr << "ERR: You must provide the option <" << defaultValue
                << "> to continue the program, exit." << std::endl;
            exit(EXIT_FAILURE);

        }
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

    std::cout << imgPath << "type:"<<internalType<<std::endl;
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
    std::cout << "INFO: saveSlice: " << width << "x" << height <<
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
ColoredImageType imageChannelMerge(const GrayscaleImageStack imgs, const int channels);
