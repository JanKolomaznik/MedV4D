#ifndef SNAKE_SEGMENTATION_FILTER_H
#define SNAKE_SEGMENTATION_FILTER_H

#include "Imaging.h"

namespace M4D
{

namespace Imaging
{

class SnakeSegmentationFilter: public M4D::Imaging::APipeFilter
{
public:
	
};

//include implementation
#include "Imaging/filters/ConvolutionFilter.tcc"

#endif //SNAKE_SEGMENTATION_FILTER_H
