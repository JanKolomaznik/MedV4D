#ifndef _ABSTRACT_IMAGE_FILTERS_H
#error File AbstractImageFilters.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{


template< typename InputImageType, typename OutputImageType >
ImageFilter< InputImageType, OutputImageType >::ImageFilter()
:	in( NULL ), _inTimestamp( Common::DefaultTimeStamp ), _inEditTimestamp( Common::DefaultTimeStamp ), 
	out( NULL ), _outTimestamp( Common::DefaultTimeStamp ), _outEditTimestamp( Common::DefaultTimeStamp )
{
	M4D::Imaging::InputPort *inPort = new InputPortType();
	M4D::Imaging::OutputPort *outPort = new OutputPortType();

	//TODO - check whether OK
	_inputPorts.AddPort( inPort );
	_outputPorts.AddPort( outPort );
}

template< typename InputImageType, typename OutputImageType >
const InputImageType&
ImageFilter< InputImageType, OutputImageType >::GetInputImage()const
{
	//TODO - exceptions
	_inputPorts.GetPortTyped< InputPortType >( 0 ).LockDataset();
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
	//TODO - exceptions
	_outputPorts.GetPortTyped< OutputPortType >( 0 ).LockDataset();
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
	
	//TODO - check
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
	//We store actual timestamps of input and output - for next execution
	_inEditTimestamp = in->GetModificationManager().GetActualTimestamp();
	_outEditTimestamp = out->GetModificationManager().GetActualTimestamp();

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
	: _sliceComputationNeighbourCount( neighbourCount ), _computationGrouping( grouping )
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
		SliceComputationRecord & record = _actualComputationGroups[ i ];

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
		{
			DL_PRINT( 5, "SliceFilter recalculation" );

			const DimensionExtents & dimExtents = this->in->GetDimensionExtents( 2 );
			unsigned groupCount = ( dimExtents.maximum - dimExtents.minimum ) / _computationGrouping;
			for( unsigned i = 0; i < groupCount; ++i ) {
				SliceComputationRecord record;
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
				record.writerBBox = &( GetComputationGroupWriterBBox( record ) );

				_actualComputationGroups.push_back( record );
			}

			SliceComputationRecord record;
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
			record.writerBBox = &( GetComputationGroupWriterBBox( record ) );

			_actualComputationGroups.push_back( record );
		}
		break;

	case AbstractPipeFilter::ADAPTIVE_CALCULATION:
		{
			DL_PRINT( 5, "SliceFilter adaptive calculation" );
			const ModificationManager &manager = this->in->GetModificationManager();
			ModificationManager::ConstChangeReverseIterator iterator; 
			for( 	iterator = manager.ChangesReverseBegin(); 
				iterator != manager.ChangesReverseEnd() /*&& ((*iterator)->GetTimeStamp()) < this->_inEditTimestamp*/; 
				++iterator 
			){
				if ( this->_inEditTimestamp >= (**iterator).GetTimeStamp() ) {
					break;
				}

				const ModificationBBox & BBox = (**iterator).GetBoundingBox();
				SliceComputationRecord record;
				
				BBox.GetInterval( 3, record.firstSlice, record.lastSlice );

				record.inputBBox = this->in->GetDirtyBBox( 
					this->in->GetDimensionExtents( 0 ).minimum,
					this->in->GetDimensionExtents( 1 ).minimum,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
					record.firstSlice - _sliceComputationNeighbourCount,
					record.lastSlice + _sliceComputationNeighbourCount
					);
				record.writerBBox = &( GetComputationGroupWriterBBox( record ) );

				_actualComputationGroups.push_back( record );
			}
		}
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

template< typename InputElementType, typename OutputElementType >
WriterBBoxInterface &
IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::GetComputationGroupWriterBBox(  SliceComputationRecord & record )
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

