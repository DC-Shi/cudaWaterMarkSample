#ifndef WATERMARK_FILEIO
#define WATERMARK_FILEIO

#include <string>
#include <FreeImage.h>
#include <vector>

using ColoredImageType = FIBITMAP*;
using GrayscaleImageStack = std::vector<FIBITMAP*>;

/// @brief Parse from commandline
/// @param argc How many arguments
/// @param argv Arguments array
/// @param option Which option should be parsed
/// @param defaultValue The default value for this option, NULL means the commandline must provide this option to continue
/// @return 
std::string parseArgs(const int argc, const char ** argv, const char * option, const std::string defaultValue);

/// @brief Load the image from file path
/// @param imgPath Image file path
/// @param channels How many channels does this image have
/// @return The image object
ColoredImageType loadImage(const std::string imgPath, int& channels);


/// @brief Save image to file
/// @param img The image ready to save
/// @param imgPath Image file path
/// @return Whether the save is successful
bool saveImage(ColoredImageType img, const std::string imgPath);


/// @brief Save one slice of the image, the grayscale, into particular colored images.
/// @param img The grayscale image ready to save
/// @param imgPath Image file path
/// @param layer Which layer does this belongs to, should be 'r', 'g', or 'b'
/// @param idx Which layer to be saved in the grayscale stack
/// @return Whether the save is successful
bool saveSlice(GrayscaleImageStack img, const std::string imgPath, const std::string layer, const int idx = 0);

/// @brief Split the image by channels and save it to an array
/// @param img The image object that has 1 or more channels
/// @param channels The channels this image has
/// @return Splited grayscale image
GrayscaleImageStack imageChannelSplit(const ColoredImageType img, const int channels);

/// @brief Merge image with multiple grayscale images
/// @param imgs Grayscale images array
/// @param channels How many channels does this image have
/// @return The merged color image
ColoredImageType imageChannelMerge(const GrayscaleImageStack imgs, const int channels);


#endif