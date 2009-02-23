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
	
	typename ITKInputImageType::RegionType region;
	typename ITKInputImageType::SpacingType spacing;
	// copy info from input medved image into input ITK image
	for(uint32 i=0; i<InputImageType::Dimension; i++)
	{
		region.GetIndex()[i] = in->GetDimensionExtents(i).minimum;
		region.GetSize()[i] = in->GetDimensionExtents(i).maximum;
		spacing[i] = in->GetDimensionExtents(i).elementExtent;		
	}
	// init input dataContainerWrap according changed input image
	//m_inputDatCnt.SetData( this->GetInputImage() );
	// output image has to be set according output of the last filter
	// within ITK pipeline
	
}
///////////////////////////////////////////////////////////////////////////////

template< typename InputImageType, typename OutputImageType >
void
ITKFilter<InputImageType, OutputImageType>
	::SetOutputImageSize(ITKOutputImageType &itkImage)
{
	int32 *maximums = itkImage.GetLargestPossibleRegion().GetSize();
	int32 *minimums = itkImage.GetLargestPossibleRegion().GetIndex();
	float32 *voxelExtents = itkImage.GetSpacing();

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}
///////////////////////////////////////////////////////////////////////////////
}
}
#endif
/** @} */