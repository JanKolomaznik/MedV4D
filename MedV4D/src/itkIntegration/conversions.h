#ifndef CONVERSIONS_H_
#define CONVERSIONS_H_

#include <stdio.h>	// memcpy

#include "Imaging/ImageFactory.h"

namespace M4D
{
namespace ITKIntegration
{

template< typename MedvedImageType, typename ITKImageType>
static void ConvertMedevedImagePropsToITKImageProps(
	typename ITKImageType::RegionType &region,
	typename ITKImageType::SpacingType &spacing,
	const MedvedImageType &inMedImage)
{
	typename ITKImageType::RegionType::IndexType &index = 
		(typename ITKImageType::RegionType::IndexType &) region.GetIndex();
	typename ITKImageType::RegionType::SizeType &size  = 
		(typename ITKImageType::RegionType::SizeType &) region.GetSize();
	
	for( unsigned i=0; i < MedvedImageType::Dimension; ++i ) {
		size[i] = inMedImage.GetDimensionExtents( i ).maximum -
			inMedImage.GetDimensionExtents( i ).minimum;
		index[i] = inMedImage.GetDimensionExtents( i ).minimum;
		spacing[i] = inMedImage.GetDimensionExtents( i ).elementExtent;
	}
}

template< typename ITKImageType, typename MedvedImageType>
static void CopyITKToMedvedImage(
	const ITKImageType &itkImage, MedvedImageType &medImage)
{
	const typename ITKImageType::RegionType::IndexType &index = itkImage.GetLargestPossibleRegion().GetIndex();
	const typename ITKImageType::RegionType::SizeType &size  = itkImage.GetLargestPossibleRegion().GetSize();
	const typename ITKImageType::SpacingType &spacing = itkImage.GetSpacing();
		
	// set output medved properties and allocate the data buffer	
	Vector< int32, MedvedImageType::Dimension > minimums;
	Vector< int32, MedvedImageType::Dimension > maximums;
	Vector< float32, MedvedImageType::Dimension > voxelExtents;

	for( unsigned i=0; i < MedvedImageType::Dimension; i++ ) {
		minimums[i] = index[i];
		maximums[i] = index[i] + size[i];
		voxelExtents[i] = spacing[i];
	}
	
	// alloc data
	//medImage.UpgradeToExclusiveLock();
	Imaging::ImageFactory::ChangeImageSize( 
			medImage, minimums, maximums, voxelExtents);
	//medImage.DowngradeFromExclusiveLock();
		
	// set buffer properties of ITK out image
	typename MedvedImageType::PointType strides;
	typename MedvedImageType::SizeType medvedSize;
	
	typename MedvedImageType::Element *dataPointer = 
		medImage.GetPointer(medvedSize, strides);

	// copy content
	memcpy(
		dataPointer, 
		itkImage.GetBufferPointer(), 
		itkImage.GetPixelContainer()->Size() * sizeof(typename ITKImageType::PixelType)
	);
}

}
}
#endif /*CONVERSIONS_H_*/
