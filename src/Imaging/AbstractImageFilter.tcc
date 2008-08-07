#ifndef _ABSTRACT_IMAGE_FILTER_H
#error File AbstractImageFilter.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{


template< typename InputImageType, typename OutputImageType >
AbstractImageFilter< InputImageType, OutputImageType >::AbstractImageFilter( typename AbstractImageFilter< InputImageType, OutputImageType >::Properties * prop )
:	AbstractPipeFilter( prop ), 
	in( NULL ), _inTimestamp( Common::DefaultTimeStamp ), _inEditTimestamp( Common::DefaultTimeStamp ), 
	out( NULL ), _outTimestamp( Common::DefaultTimeStamp ), _outEditTimestamp( Common::DefaultTimeStamp )
{
	M4D::Imaging::InputPort *inPort = new InputPortType();
	M4D::Imaging::OutputPort *outPort = new OutputPortType();

	//TODO check if OK
	_inputPorts.AddPort( inPort );
	_outputPorts.AddPort( outPort );
}

template< typename InputImageType, typename OutputImageType >
const InputImageType&
AbstractImageFilter< InputImageType, OutputImageType >::GetInputImage()const
{
	_inputPorts.GetPortTyped< InputPortType >( 0 ).LockDataset();
	return _inputPorts.GetPortTyped< InputPortType >( 0 ).GetImage();
}

template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >::ReleaseInputImage()const
{
	_inputPorts.GetPortTyped< InputPortType >( 0 ).ReleaseDatasetLock();
}

template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >::ReleaseOutputImage()const
{
	_outputPorts.GetPortTyped< OutputPortType >( 0 ).ReleaseDatasetLock();
}

template< typename InputImageType, typename OutputImageType >
OutputImageType&
AbstractImageFilter< InputImageType, OutputImageType >::GetOutputImage()const
{
	_outputPorts.GetPortTyped< OutputPortType >( 0 ).LockDataset();
	return _outputPorts.GetPortTyped< OutputPortType >( 0 ).GetImage();
}

template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >
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
AbstractImageFilter< InputImageType, OutputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );	
	
	this->in = &(this->GetInputImage());
	this->out = &(this->GetOutputImage());

	D_PRINT( "Input Image : " << this->in );
	D_PRINT( "Output Image : " << this->out );

	Common::TimeStamp inTS = in->GetStructureTimestamp();
	Common::TimeStamp outTS = out->GetStructureTimestamp();

	//Check whether structure of images changed
	if ( 
		!inTS.IdenticalID( _inTimestamp )
		|| inTS != _inTimestamp
		|| !outTS.IdenticalID( _outTimestamp )
		|| outTS != _outTimestamp 
	) {
		utype = AbstractPipeFilter::RECALCULATION;
		_inTimestamp = inTS;
		_outTimestamp = outTS;
		PrepareOutputDatasets();
	}
	if( utype == AbstractPipeFilter::ADAPTIVE_CALCULATION ) {
		Common::TimeStamp inEditTS = in->GetModificationManager().GetLastStoredTimestamp();
		Common::TimeStamp outEditTS = out->GetModificationManager().GetActualTimestamp();
		if( 
			!inEditTS.IdenticalID( _inEditTimestamp ) 
			|| !outEditTS.IdenticalID( _outEditTimestamp )
			|| inEditTS > _inEditTimestamp 
			|| outEditTS != _outEditTimestamp
		) {
			utype = AbstractPipeFilter::RECALCULATION;
		}
	}
}

template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

}


template< typename InputImageType, typename OutputImageType >
void
AbstractImageFilter< InputImageType, OutputImageType >
::AfterComputation( bool successful )
{
	//We store actual timestamps of input and output - for next execution
	_inEditTimestamp = in->GetModificationManager().GetActualTimestamp();
	_outEditTimestamp = out->GetModificationManager().GetActualTimestamp();

	this->ReleaseInputImage();
	this->ReleaseOutputImage();

	PredecessorType::AfterComputation( successful );	
}
//******************************************************************************
//******************************************************************************


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
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
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
	PredecessorType::PrepareOutputDatasets();
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
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
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
	PredecessorType::PrepareOutputDatasets();
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

#endif /*_ABSTRACT_IMAGE_FILTER_H*/

