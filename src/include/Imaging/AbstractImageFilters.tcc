#ifndef _ABSTRACT_IMAGE_FILTERS_H
#error File AbstractImageFilters.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{


template< typename InputImageType, typename OutputImageType >
ImageFilter< InputImageType, OutputImageType >::ImageFilter()
{
	M4D::Imaging::InputPort *in = new InputPortType();
	M4D::Imaging::OutputPort *out = new OutputPortType();

	//TODO - check whether OK
	_inputPorts.AddPort( in );
	_outputPorts.AddPort( out );
}

template< typename InputImageType, typename OutputImageType >
const InputImageType&
ImageFilter< InputImageType, OutputImageType >::GetInputImage()const
{
	return _inputPorts.GetPortTyped< InputPortType >( 0 ).GetImage();
}

template< typename InputImageType, typename OutputImageType >
void
ImageFilter< InputImageType, OutputImageType >::ReleaseInputImage()const
{
	//TODO
	_inputPorts.GetPortTyped< InputPortType >( 0 ).ReleaseDatasetLock();
}

template< typename InputImageType, typename OutputImageType >
void
ImageFilter< InputImageType, OutputImageType >::ReleaseOutputImage()const
{
	//TODO
	_outputPorts.GetPortTyped< OutputPortType >( 0 ).ReleaseDatasetLock();
}

template< typename InputImageType, typename OutputImageType >
OutputImageType&
ImageFilter< InputImageType, OutputImageType >::GetOutputImage()const
{
	return _outputPorts.GetPortTyped< OutputPortType >( 0 ).GetImage();
}

template< typename InputImageType, typename OutputImageType >
void
ImageFilter< InputImageType, OutputImageType >
::SetOutputImageSize( 
		size_t 		minimums[ ], 
		size_t 		maximums[ ], 
		float32		elementExtents[ ]
	    )
{
	_outputPorts.GetPortTyped< OutputPortType >( 0 ).SetImageSize( minimums, maximums, elementExtents );
}

template< typename InputImageType, typename OutputImageType >
void
ImageFilter< InputImageType, OutputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	//TODO
	PredecessorType::BeforeComputation( utype );	

	in = &( this->GetInputImage() );
	out = &( this->GetOutputImage() );
}

template< typename InputImageType, typename OutputImageType >
void
ImageFilter< InputImageType, OutputImageType >
::AfterComputation( bool successful )
{
	this->ReleaseInputImage();
	this->ReleaseOutputImage();

	//TODO
	PredecessorType::AfterComputation( successful );	
}
//******************************************************************************
//******************************************************************************

template< typename InputElementType, typename OutputImageType >
ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::ImageSliceFilter() : _sliceComputationNeighbourCount( 0 )
{

}

template< typename InputElementType, typename OutputImageType >
bool
ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::ExecutionThreadMethod()
{
	//TODO
	ExecutionOnWholeThreadMethod();
	return true;
}

template< typename InputElementType, typename OutputImageType >
bool
ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::ExecutionOnWholeThreadMethod()
{
	//const Image< InputElementType, 3 > &in = this->GetInputImage();
	//OutputImageType &out = this->GetOutputImage();
	
	//TODO - better implementation	
	for( 
		size_t i = this->in->GetDimensionExtents( 2 ).minimum; 
		i < this->in->GetDimensionExtents( 2 ).maximum;
		++i
	) {
		ProcessSlice( 	*(this->in), 
				*(this->out),
				this->in->GetDimensionExtents( 0 ).minimum,
				this->in->GetDimensionExtents( 1 ).minimum,
				this->in->GetDimensionExtents( 0 ).maximum,
				this->in->GetDimensionExtents( 1 ).maximum,
				i 
				);

	}
	return true;
}

template< typename InputElementType, typename OutputImageType >
void
ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::PrepareOutputDatasets()
{
	//TODO
}

template< typename InputElementType, typename OutputImageType >
void
ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	//TODO
	PredecessorType::BeforeComputation( utype );	
}

//******************************************************************************
//******************************************************************************

template< typename InputElementType, typename OutputImageType >
ImageVolumeFilter< Image< InputElementType, 3 >, OutputImageType >
::ImageVolumeFilter()
{

}

template< typename InputElementType, typename OutputImageType >
bool
ImageVolumeFilter< Image< InputElementType, 3 >, OutputImageType >
::ExecutionThreadMethod()
{
	//TODO
	ExecutionOnWholeThreadMethod();
	return true;
}

template< typename InputElementType, typename OutputImageType >
bool
ImageVolumeFilter< Image< InputElementType, 3 >, OutputImageType >
::ExecutionOnWholeThreadMethod()
{
	//const Image< InputElementType, 3 > &in = this->GetInputImage();
	//OutputImageType &out = this->GetOutputImage();

	//TODO - better implementation	
	
	ProcessVolume( 
			*(this->in),
			*(this->out),
			this->in->GetDimensionExtents( 0 ).minimum,
			this->in->GetDimensionExtents( 1 ).minimum,
			this->in->GetDimensionExtents( 2 ).minimum,
			this->in->GetDimensionExtents( 0 ).maximum,
			this->in->GetDimensionExtents( 1 ).maximum,
			this->in->GetDimensionExtents( 2 ).maximum
		     );

	return true;
}

template< typename InputElementType, typename OutputImageType >
void
ImageVolumeFilter< Image< InputElementType, 3 >, OutputImageType >
::PrepareOutputDatasets()
{
	//TODO
}

template< typename InputElementType, typename OutputImageType >
void
ImageVolumeFilter< Image< InputElementType, 3 >, OutputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	//TODO
	PredecessorType::BeforeComputation( utype );	
}

//******************************************************************************
//******************************************************************************

template< typename InputElementType, typename OutputImageType >
ImageVolumeFilter< Image< InputElementType, 4 >, OutputImageType >
::ImageVolumeFilter()
{

}

template< typename InputElementType, typename OutputImageType >
bool
ImageVolumeFilter< Image< InputElementType, 4 >, OutputImageType >
::ExecutionThreadMethod()
{
	//TODO
	ExecutionOnWholeThreadMethod();
	return true;
}

template< typename InputElementType, typename OutputImageType >
bool
ImageVolumeFilter< Image< InputElementType, 4 >, OutputImageType >
::ExecutionOnWholeThreadMethod()
{
	//const Image< InputElementType, 4 > &in = this->GetInputImage();
	//OutputImageType &out = this->GetOutputImage();
	
	//TODO - better implementation	
	for( 
		size_t i = this->in->GetDimensionExtents( 3 ).minimum; 
		i < this->in->GetDimensionExtents( 3 ).maximum;
		++i
	) {
		ProcessVolume( 
			*(this->in),
			*(this->out),
			this->in->GetDimensionExtents( 0 ).minimum,
			this->in->GetDimensionExtents( 1 ).minimum,
			this->in->GetDimensionExtents( 2 ).minimum,
			this->in->GetDimensionExtents( 0 ).maximum,
			this->in->GetDimensionExtents( 1 ).maximum,
			this->in->GetDimensionExtents( 2 ).maximum,
			i
		     );
		

	}

	return true;
}

template< typename InputElementType, typename OutputImageType >
void
ImageVolumeFilter< Image< InputElementType, 4 >, OutputImageType >
::PrepareOutputDatasets()
{
	//TODO
}

template< typename InputElementType, typename OutputImageType >
void
ImageVolumeFilter< Image< InputElementType, 4 >, OutputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	//TODO
	PredecessorType::BeforeComputation( utype );	
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_SIMPLE_IMAGE_FILTER_H*/

