/////////////// NOT USED !!!!! Bud left in repo ... ///////////////////

#ifndef ITKFILTER_H_
#error File itkFilter.cxx cannot be included directly!
#else

namespace M4D
{
namespace ITKIntegration
{

/////////////////////////////////////////////////////////////////////////////////
template<typename PixelType, uint16 dimension>
void
ITKImageWrapper<PixelType, dimension>
	::SetupAccordingMedvedImage(const MedvedImageType &medImage)
{
	typename MedvedImageType::PointType strides;
	typename MedvedImageType::SizeType size;
	size_t sizeOfData = 1;	// size in elements (not in bytes) 
	typename MedvedImageType::Element *dataPointer = medImage.GetPointer(size, strides);
	// count num of elems
	for( uint32 i=0; i< MedvedImageType::Dimension; i++)
		sizeOfData *= size[i];
	
	this->GetPixelContainer()->SetImportPointer(
			dataPointer, 
			(typename itkImageType::PixelContainer::ElementIdentifier) sizeOfData);
	
	
	typename itkImageType::RegionType::SizeType &regionSize = 
		(typename itkImageType::RegionType::SizeType &) 
			this->GetLargestPossibleRegion().GetSize();
	
	typename itkImageType::RegionType::IndexType &regionIndex = 
		(typename itkImageType::RegionType::IndexType &) 
			this->GetLargestPossibleRegion().GetIndex();
	
	typename itkImageType::SpacingType &spacing = 
		(typename itkImageType::SpacingType &) this->GetSpacing();
	
	// copy info from input medved image into input ITK image
	for(uint32 i=0; i<MedvedImageType::Dimension; i++)
	{
		regionSize[i] = medImage.GetDimensionExtents(i).maximum -
			medImage.GetDimensionExtents(i).minimum;
		regionIndex[i] = medImage.GetDimensionExtents(i).minimum;
		spacing[i] = medImage.GetDimensionExtents(i).elementExtent;		
	}
}

/////////////////////////////////////////////////////////////////////////////////
}
}
#endif
