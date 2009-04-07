/**
 * @ingroup imaging 
 * @author Venca_X
 * @file itkFilter.tcc 
 * @{ 
 **/

#ifndef ITKFILTER_H_
#error File itkFilter.cxx cannot be included directly!
#else

namespace M4D
{
namespace ITKIntegration
{
/////////////////////////////////////////////////////////////////////////////////
template< typename InputImageType, typename OutputImageType >
ITKFilter<InputImageType, OutputImageType>::ITKFilter()
	: PredecessorType( new typename PredecessorType::Properties() )
	, inITKImage( ITKInputImageType::New() )
	, outITKImage( NULL)
{
//	// set our dataContainerWraps to ITKImages
//	m_inputITKImage.SetPixelContainer( & m_inputDatCnt);
//	m_outputITKImage.SetPixelContainer( & m_outputDatCnt);
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputImageType, typename OutputImageType >
void
ITKFilter<InputImageType, OutputImageType>::SetOutImageSize(
		const typename ITKOutputImageType::RegionType &region,
		const typename ITKOutputImageType::SpacingType &spacing)
{
	// set itkOutImage Properties
	outITKImage->SetLargestPossibleRegion(region);
	outITKImage->SetBufferedRegion(region);	// buferred is the whole region
	outITKImage->SetSpacing(spacing);
	
	// set output medved properties and allocate the data buffer	
	int32 minimums[ OutputImageType::Dimension ];
	int32 maximums[ OutputImageType::Dimension ];
	float32 voxelExtents[ OutputImageType::Dimension ];

	for( unsigned i=0; i < OutputImageType::Dimension; i++ ) {
		minimums[i] = region.GetIndex()[i];
		maximums[i] = region.GetIndex()[i] + region.GetSize()[i];
		voxelExtents[i] = spacing[i];
	}
	this->SetOutputImageSize( minimums, maximums, voxelExtents );
	
	// set buffer properties of ITK out image
	typename OutputImageType::PointType strides;
	typename OutputImageType::SizeType size;
	size_t sizeOfData = 1;	// size in elements (not in bytes) 
	const OutputImageType &outMedImage = this->GetOutputImage();
	typename OutputImageType::Element *dataPointer = 
		outMedImage.GetPointer(size, strides);
	// count num of elems
	for( uint32 i=0; i< InputImageType::Dimension; i++)
		sizeOfData *= size[i];
		
	outITKImage->GetPixelContainer()->SetImportPointer(
			dataPointer, 
			(typename ITKInputImageType::PixelContainer::ElementIdentifier) sizeOfData);
}

///////////////////////////////////////////////////////////////////////////////
template< typename InputImageType, typename OutputImageType >
void
ITKFilter<InputImageType, OutputImageType>::PrepareOutputDatasets(void)
{
	PredecessorType::PrepareOutputDatasets();	
	SetupInITKImageAccordingInMedevedImage();	
}
///////////////////////////////////////////////////////////////////////////////
//template< typename InputImageType, typename OutputImageType >
//void
//ITKFilter<InputImageType, OutputImageType>
//	::SetupOutMedvedImageAccordingOutputITKImage(void)
//{
//	// craete outMedvedImage buffer pointing to outITKImage buffer
////	Vector< int32, OutputImageType::Dimension > 	size;
////	Vector< float32, OutputImageType::Dimension >	elementSize;
//	
//	const typename ITKInputImageType::RegionType::SizeType &regionSize = 
//		outITKImage->GetLargestPossibleRegion().GetSize();
//	
////	const typename ITKInputImageType::RegionType::IndexType &regionIndex = 
////		outITKImage->GetLargestPossibleRegion().GetIndex();
//	
//	const typename ITKInputImageType::SpacingType &spacing =
//		outITKImage->GetSpacing();
//	
//	Vector< int32, OutputImageType::Dimension > 	size;
//	Vector< float32, OutputImageType::Dimension >	elementSize;
//	
//	// copy output ITK image info into out medved image 
//	for(uint32 i=0; i<OutputImageType::Dimension; i++)
//	{
//		size[i] = regionSize[i];
//		elementSize[i] = spacing[i];
//	}
//	
//	M4D::Imaging::ImageFactory::AssignNewDataToImage< 
//			typename OutputImageType::Element, OutputImageType::Dimension>(
//				outITKImage->GetBufferPointer(),
//				this->GetOutputImage(), size, elementSize );
//	
////	typename OutputImageType::Ptr newOutImage = typename OutputImageType::Ptr(
////			new OutputImageType(container));
//	
//	// put new image into output connection
//	//this->_outputPorts[0].GetConnection()->PutDataset( newOutImage);
//}
///////////////////////////////////////////////////////////////////////////////
template< typename InputImageType, typename OutputImageType >
void
ITKFilter<InputImageType, OutputImageType>
	::SetupInITKImageAccordingInMedevedImage(void)
{
	const InputImageType &inMedImage = *this->in;
	typename InputImageType::PointType strides;
	typename InputImageType::SizeType size;
	size_t sizeOfData = 1;	// size in elements (not in bytes) 
	typename InputImageType::Element *dataPointer = 
		inMedImage.GetPointer(size, strides);
	// count num of elems
	for( uint32 i=0; i< InputImageType::Dimension; i++)
		sizeOfData *= size[i];
	
	inITKImage->GetPixelContainer()->SetImportPointer(
			dataPointer, 
			(typename ITKInputImageType::PixelContainer::ElementIdentifier) sizeOfData);	
	
	typename ITKInputImageType::RegionType::SizeType &regionSize = 
		(typename ITKInputImageType::RegionType::SizeType &) 
			inITKImage->GetLargestPossibleRegion().GetSize();
	
	typename ITKInputImageType::RegionType::IndexType &regionIndex = 
		(typename ITKInputImageType::RegionType::IndexType &) 
			inITKImage->GetLargestPossibleRegion().GetIndex();
	
	typename ITKInputImageType::SpacingType &spacing = 
		(typename ITKInputImageType::SpacingType &) inITKImage->GetSpacing();
	
	// copy info from input medved image into input ITK image
	for(uint32 i=0; i<InputImageType::Dimension; i++)
	{
		regionSize[i] = inMedImage.GetDimensionExtents(i).maximum -
			inMedImage.GetDimensionExtents(i).minimum;
		regionIndex[i] = inMedImage.GetDimensionExtents(i).minimum;
		spacing[i] = inMedImage.GetDimensionExtents(i).elementExtent;		
	}
	
	inITKImage->SetBufferedRegion( inITKImage->GetLargestPossibleRegion() );
}
///////////////////////////////////////////////////////////////////////////////
}
}
#endif
/** @} */
