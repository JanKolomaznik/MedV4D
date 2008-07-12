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

}

template< typename InputImageType, typename OutputImageType >
void
ImageFilter< InputImageType, OutputImageType >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();
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
::ImageSliceFilter( unsigned neighbourCount, unsigned grouping ) 
	: _sliceComputationNeighbourCount( neighbourCount ), _computationGrouping( grouping );
{
	//TODO - check intervals of parameters - throw exceptions
}

template< typename InputElementType, typename OutputImageType >
bool
ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	for( size_t i = 0; i < _actualComputationGroups.size(); ++i )
	{
		ComputationRecord & record = _actualComputationGroups[ i ];

		//Wait until input area is ready
		if ( !record.inputBBox->WaitWhileDirty() ) {
			//TODO - exit
		}

		for( int32 slice = record.firstSlice; slice <= record.lastSlice; ++slice )
		{
			ProcessSlice( 	*(this->in), 
					*(this->out),
					this->in->GetDimensionExtents( 0 ).minimum,
					this->in->GetDimensionExtents( 1 ).minimum,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
					slice 
					);
		}

		record.writerBBox->SetModified();
	}

	_actualComputationGroups.clear();

	/*
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

	}*/
	return true;
}

template< typename InputElementType, typename OutputImageType >
void
ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	//TODO
	PredecessorType::BeforeComputation( utype );	

	switch ( utype ) {
	case AbstractPipeFilter::RECALCULATION:

		DL_PRINT( 5, "SliceFilter recalculation" );

		const DimensionExtents & dimExtents = this->in->GetDimensionExtents( 2 );
		unsigned groupCount = ( dimExtents.maximum - dimExtents.minimum ) / _computationGrouping;
		for( unsigned i = 0; i < groupCount; ++i ) {
			ComputationRecord record;
			record.firstSlice = dimExtents.minimum + (i*_computationGrouping);
			record.lastSlice = dimExtents.minimum + ((i+1)*_computationGrouping) - 1;
			record.inputBBox = this->in->GetDirtyBBox( 
				this->in->GetDimensionExtents( 0 ).minimum,
				this->in->GetDimensionExtents( 1 ).minimum,
				this->in->GetDimensionExtents( 0 ).maximum,
				this->in->GetDimensionExtents( 1 ).maximum,
				record.firstSlice - _sliceComputationNeighbourCount,
				record.lastSlice + _sliceComputationNeighbourCount
				);
			record.writerBBox = GetComputationGroupWriterBBox( record );

			_actualComputationGroups.push_back( record );
		}

		ComputationRecord record;
		record.firstSlice = dimExtents.minimum + (groupCount*_computationGrouping) - _sliceComputationNeighbourCount;
		record.lastSlice = dimExtents.maximum - 1;
		record.inputBBox = this->in->GetDirtyBBox( 
			this->in->GetDimensionExtents( 0 ).minimum,
			this->in->GetDimensionExtents( 1 ).minimum,
			this->in->GetDimensionExtents( 0 ).maximum,
			this->in->GetDimensionExtents( 1 ).maximum,
			record.firstSlice - _sliceComputationNeighbourCount,
			record.lastSlice + _sliceComputationNeighbourCount
			);
		record.writerBBox = GetComputationGroupWriterBBox( record );

		_actualComputationGroups.push_back( record );

		break;

	case AbstractPipeFilter::ADAPTIVE_CALCULATION:
		DL_PRINT( 5, "SliceFilter adaptive calculation" );
		break;
	default:
		ASSERT( false );
	}
}

//******************************************************************************
//******************************************************************************

template< typename InputElementType, typename OutputElementType >
IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::IdenticalExtentsImageSliceFilter( unsigned neighbourCount, unsigned grouping ) 
	: PredecessorType( neighbourCount, grouping )
{

}

template< typename InputElementType, typename OutputElementType >
void
IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	//TODO
	PredecessorType::BeforeComputation( utype );	
}

WriterBBoxInterface *
IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::GetComputationGroupWriterBBox( ComputationRecord & record )
{
	return this->out->SetDirtyBBox( this->in->GetDimensionExtents( 0 ).minimum,
			this->in->GetDimensionExtents( 1 ).minimum,
			this->in->GetDimensionExtents( 0 ).maximum,
			this->in->GetDimensionExtents( 1 ).maximum,
			record.firstSlice,
			record.lastSlice
			);
}

template< typename InputElementType, typename OutputElementType >
void
IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

	size_t minimums[3];
	size_t maximums[3];
	float32 voxelExtents[3];

	for( unsigned i=0; i < 3; ++i ) {
		const DimensionExtents & dimExt = this->in->GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum;
		voxelExtents[i] = dimExt.elementExtent;
	}
	this->SetOutputImageSize( minimums, maximums, voxelExtents );
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

#endif /*_SIMPLE_IMAGE_FILTER_H*/

