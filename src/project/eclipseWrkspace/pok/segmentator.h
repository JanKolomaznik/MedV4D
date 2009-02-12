#ifndef SEGMENTATOR_H_
#define SEGMENTATOR_H_

#include "itkImage.h"
#include "itkThresholdSegmentationLevelSetImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkZeroCrossingImageFilter.h"

// We define the pixel type and dimension of the image to be read.
// We also choose to use the \doxygen{OrientedImage} in order to make sure
// that the image orientation information contained in the direction cosines
// of the DICOM header are read in and passed correctly down the image processing
// pipeline.

typedef float InternalPixelType;
typedef signed short ReadWritePixelType;

typedef itk::Image< ReadWritePixelType, 3 >   ReadWriteImageType;
typedef itk::Image< InternalPixelType, 3 >   InternalImageType;

// threshold filter used to threshold final output to zeros and ones
typedef itk::BinaryThresholdImageFilter<InternalImageType, ReadWriteImageType>
    ThresholdingFilterType;

// filter that created initial level set
typedef  itk::FastMarchingImageFilter< InternalImageType, InternalImageType >
    FastMarchingFilterType;

// filter that performs actual levelset segmentation
typedef  itk::ThresholdSegmentationLevelSetImageFilter< 
					InternalImageType, InternalImageType, InternalPixelType >
	ThresholdSegmentationLevelSetImageFilterType;

class Segmentator
{
public:
	Segmentator(float upperThreshold, float lowerThreshold, float curvatureScaling,
			float seedX, float seedY, float seedZ, double initialDistance,
			const ReadWriteImageType::SizeType &imageSize);

	ThresholdingFilterType::Pointer thresholder;
	FastMarchingFilterType::Pointer fastMarching;
	ThresholdSegmentationLevelSetImageFilterType::Pointer thresholdSegmentation;
	
	void PrintInfo(void);
	
private:	
	void SetupFastMarchingFilter(
			float seedX, float seedY, float seedZ, 
			double initialDistance, const ReadWriteImageType::SizeType &imageSize);
	void SetupBinaryThresholder(void);
	void SetupLevelSetSegmentator(
			float upperThreshold, float lowerThreshold, float curvatureScaling);
};

#endif /*SEGMENTATOR_H_*/
