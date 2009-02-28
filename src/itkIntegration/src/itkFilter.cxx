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
ITKFilter<InputImageType, OutputImageType>::PrepareOutputDatasets(void)
{
	PredecessorType::PrepareOutputDatasets();
	
	SetupInITKImageAccordingInMedevedImage();
	SetupOutMedvedImageAccordingOutputITKImage();
	//const InputImageType &in = this->GetInputImage();			
	//m_inputITKImageWrapper.SetupAccordingMedvedImage(in);
	// init input dataContainerWrap according changed input image
	//m_inputDatCnt.SetData( this->GetInputImage() );
	// output image has to be set according output of the last filter
	// within ITK pipeline
	
}
///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
ITKFilter<InputImageType, OutputImageType>
	::SetOutputImageSizeAccordingITK(ITKOutputImageType *itkImage)
{
	int32 *maximums = (int32 *) &itkImage->GetLargestPossibleRegion().GetSize();
	int32 *minimums = (int32 *) &itkImage->GetLargestPossibleRegion().GetIndex();
	float32 *voxelExtents = (float32 *)itkImage->GetSpacing();

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputImageType, typename OutputImageType >
void
ITKFilter<InputImageType, OutputImageType>
	::SetupOutMedvedImageAccordingOutputITKImage(void)
{
	// craete outMedvedImage buffer pointing to outITKImage buffer
	//OutputImageType &outMedImage = this->GetOutputImage();
	Vector< int32, OutputImageType::Dimension > 	size;
	Vector< float32, OutputImageType::Dimension >	elementSize;
	
	const typename ITKInputImageType::RegionType::SizeType &regionSize = 
		outITKImage->GetLargestPossibleRegion().GetSize();
	
//	const typename ITKInputImageType::RegionType::IndexType &regionIndex = 
//		outITKImage->GetLargestPossibleRegion().GetIndex();
	
	const typename ITKInputImageType::SpacingType &spacing =
		outITKImage->GetSpacing();
	
	// copy info from input medved image into input ITK image
	for(uint32 i=0; i<InputImageType::Dimension; i++)
	{
//		outMedImage.GetDimensionExtents(i).maximum = regionSize[i];
//		outMedImage.GetDimensionExtents(i).minimum = regionIndex[i];
//		outMedImage.GetDimensionExtents(i).elementExtent = spacing[i];
		
		size[i] = regionSize[i];
		elementSize[i] = spacing[i];
	}
	
	typedef M4D::Imaging::ImageDataTemplate<typename OutputImageType::Element> BufferType;
	typename BufferType::Ptr container =
		M4D::Imaging::ImageFactory::CreateImageDataBuffer< 
			typename OutputImageType::Element, OutputImageType::Dimension>(
				outITKImage->GetBufferPointer(),
				size, elementSize );
	
	typename OutputImageType::Ptr newOutImage = typename OutputImageType::Ptr(
			new OutputImageType(container));
	
	// put new image into output connection
	this->_outputPorts[0].GetConnection()->PutDataset( newOutImage);
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputImageType, typename OutputImageType >
void
ITKFilter<InputImageType, OutputImageType>
	::SetupInITKImageAccordingInMedevedImage(void)
{
	const InputImageType &inMedImage = this->GetInputImage();
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
}
///////////////////////////////////////////////////////////////////////////////
}
}
#endif
/** @} */
