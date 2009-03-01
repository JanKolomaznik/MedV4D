#ifndef CONVERSIONS_H_
#define CONVERSIONS_H_

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

}
}
#endif /*CONVERSIONS_H_*/
