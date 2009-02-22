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
	: PredecessorType( PredecessorType::Properties)
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
	
	
	m_inputITKImage.GetPixelContainer.SetImportPointer(
					data->GetData(), (TElementIdentifier) data->GetSize());
	
	// init input dataContainerWrap according changed input image
	m_inputDatCnt.SetData( this->GetInputImage() );
	// output image has to be set according output of the last filter
	// within ITK pipeline
}
///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
ITKFilter<InputImageType, OutputImageType>
	::SetOutputImageSize(ITKOutputImageType *image)
{
	int32 *minimums = image->GetLargestPossibleRegion().GetSize();
	int32 *minimums = image->GetLargestPossibleRegion().GetIndex();
	float32 *voxelExtents = image->GetSpacing();

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}
///////////////////////////////////////////////////////////////////////////////
}
}
#endif
/** @} */