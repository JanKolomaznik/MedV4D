#ifndef _ABSTRACT_IMAGE_FILTERS_H
#error File AbstractImageFilters.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{


template< typename InputImageType, typename OutputImageType >
ImageFilter< InputImageType, OutputImageType >::ImageFilter( typename ImageFilter< InputImageType, OutputImageType >::Properties * prop )
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
ImageFilter< InputImageType, OutputImageType >::GetInputImage()const
{
	_inputPorts.GetPortTyped< InputPortType >( 0 ).LockDataset();
	return _inputPorts.GetPortTyped< InputPortType >( 0 ).GetImage();
}

template< typename InputImageType, typename OutputImageType >
void
ImageFilter< InputImageType, OutputImageType >::ReleaseInputImage()const
{
	_inputPorts.GetPortTyped< InputPortType >( 0 ).ReleaseDatasetLock();
}

template< typename InputImageType, typename OutputImageType >
void
ImageFilter< InputImageType, OutputImageType >::ReleaseOutputImage()const
{
	_outputPorts.GetPortTyped< OutputPortType >( 0 ).ReleaseDatasetLock();
}

template< typename InputImageType, typename OutputImageType >
OutputImageType&
ImageFilter< InputImageType, OutputImageType >::GetOutputImage()const
{
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

	PredecessorType::AfterComputation( successful );	
}
//******************************************************************************
//******************************************************************************

template< typename InputElementType, typename OutputImageType >
ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::ImageSliceFilter( typename ImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >::Properties *prop ) 
	: PredecessorType( prop )
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
		if ( !(record.inputBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
			for( size_t j = i; j < _actualComputationGroups.size(); ++j )
			{
				_actualComputationGroups[ i ].writerBBox->SetState( MS_CANCELED );
			}
			//TODO clear _actualComputationGroups
			return false;
		}

		for( int32 slice = record.firstSlice; slice <= record.lastSlice; ++slice )
		{
			//TODO check result
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

	unsigned computationGrouping = static_cast<Properties*>(this->_properties)->_computationGrouping;
	unsigned sliceComputationNeighbourCount = static_cast<Properties*>(this->_properties)->_sliceComputationNeighbourCount;

	switch ( utype ) {
	case AbstractPipeFilter::RECALCULATION:
		{
			DL_PRINT( 5, "SliceFilter recalculation" );

			const DimensionExtents & dimExtents = this->in->GetDimensionExtents( 2 );
			unsigned groupCount = ( dimExtents.maximum - dimExtents.minimum ) / computationGrouping;
			for( unsigned i = 0; i < groupCount; ++i ) {
				SliceComputationRecord record;
				record.firstSlice = dimExtents.minimum + (i*computationGrouping);
				record.lastSlice = dimExtents.minimum + ((i+1)*computationGrouping) - 1;
				record.inputBBox = this->in->GetDirtyBBox( 
					this->in->GetDimensionExtents( 0 ).minimum,
					this->in->GetDimensionExtents( 1 ).minimum,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
					record.firstSlice - sliceComputationNeighbourCount,
					record.lastSlice + sliceComputationNeighbourCount
					);
				record.writerBBox = &( GetComputationGroupWriterBBox( record ) );

				_actualComputationGroups.push_back( record );
			}

			SliceComputationRecord record;
			record.firstSlice = dimExtents.minimum + (groupCount*computationGrouping) - sliceComputationNeighbourCount;
			record.lastSlice = dimExtents.maximum - 1;
			record.inputBBox = this->in->GetDirtyBBox( 
				this->in->GetDimensionExtents( 0 ).minimum,
				this->in->GetDimensionExtents( 1 ).minimum,
				this->in->GetDimensionExtents( 0 ).maximum,
				this->in->GetDimensionExtents( 1 ).maximum,
				record.firstSlice - sliceComputationNeighbourCount,
				record.lastSlice + sliceComputationNeighbourCount
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
					record.firstSlice - sliceComputationNeighbourCount,
					record.lastSlice + sliceComputationNeighbourCount
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
::IdenticalExtentsImageSliceFilter( typename IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >::Properties *prop ) 
	: PredecessorType( prop )
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

//******************************************************************************
//******************************************************************************

template< typename InputImageType, typename OutputImageType >
ImageFilterWholeAtOnce< InputImageType, OutputImageType >
::ImageFilterWholeAtOnce( typename ImageFilterWholeAtOnce< InputImageType, OutputImageType >::Properties *prop ) 
	: PredecessorType( prop )
{

}


template< typename InputImageType, typename OutputImageType >
bool
ImageFilterWholeAtOnce< InputImageType, OutputImageType >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	if ( !( _readerBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
		_writerBBox->SetState( MS_CANCELED );
		return false;
	}

	if( ProcessImage( *(this->in), *(this->out) ) ) {
		_writerBBox->SetModified();
		return true;
	} else {
		_writerBBox->SetState( MS_CANCELED );
		return false;
	}
}

template< typename InputImageType, typename OutputImageType >
void
ImageFilterWholeAtOnce< InputImageType, OutputImageType >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );
	
	//This kind of filter computes always on whole dataset
	utype = AbstractPipeFilter::RECALCULATION;

	_readerBBox = ApplyReaderBBox( *(this->in) );
	_writerBBox = ApplyWriterBBox( *(this->out) );
	
}

template< typename InputImageType, typename OutputImageType >
void
ImageFilterWholeAtOnce< InputImageType, OutputImageType >
::AfterComputation( bool successful )
{
	_readerBBox = ReaderBBoxInterface::Ptr();
	_writerBBox = NULL;

	PredecessorType::AfterComputation( successful );
}
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
template< typename ElementType, unsigned dim >
ReaderBBoxInterface::Ptr
ApplyReaderBBox(  const Image< ElementType, dim > &in );

template< typename ElementType, unsigned dim >
ReaderBBoxInterface::Ptr
ApplyReaderBBoxFunc(  const Image< ElementType, 2 > &in )
{
	return in.GetDirtyBBox( 
				in.GetDimensionExtents( 0 ).minimum,
				in.GetDimensionExtents( 1 ).minimum,
				in.GetDimensionExtents( 0 ).maximum,
				in.GetDimensionExtents( 1 ).maximum
			);
}

template< typename ElementType, unsigned dim >
ReaderBBoxInterface::Ptr
ApplyReaderBBoxFunc(  const Image< ElementType, 3 > &in )
{
	return in.GetDirtyBBox( 
				in.GetDimensionExtents( 0 ).minimum,
				in.GetDimensionExtents( 1 ).minimum,
				in.GetDimensionExtents( 2 ).minimum,
				in.GetDimensionExtents( 0 ).maximum,
				in.GetDimensionExtents( 1 ).maximum,
				in.GetDimensionExtents( 2 ).maximum
			);
}

template< typename ElementType, unsigned dim >
ReaderBBoxInterface::Ptr
ApplyReaderBBoxFunc(  const Image< ElementType, 4 > &in )
{
	return in.GetDirtyBBox( 
				in.GetDimensionExtents( 0 ).minimum,
				in.GetDimensionExtents( 1 ).minimum,
				in.GetDimensionExtents( 2 ).minimum,
				in.GetDimensionExtents( 3 ).minimum,
				in.GetDimensionExtents( 0 ).maximum,
				in.GetDimensionExtents( 1 ).maximum,
				in.GetDimensionExtents( 2 ).maximum,
				in.GetDimensionExtents( 3 ).maximum
			);
}

template< typename ElementType, unsigned dim >
WriterBBoxInterface *
ApplyWriterBBoxFunc(  const Image< ElementType, dim > &out );

template< typename ElementType, unsigned dim >
WriterBBoxInterface *
ApplyWriterBBoxFunc(  const Image< ElementType, 2 > &out )
{
	return out.SetDirtyBBox( 
				out.GetDimensionExtents( 0 ).minimum,
				out.GetDimensionExtents( 1 ).minimum,
				out.GetDimensionExtents( 0 ).maximum,
				out.GetDimensionExtents( 1 ).maximum
			);
}

template< typename ElementType, unsigned dim >
WriterBBoxInterface *
ApplyWriterBBoxFunc(  const Image< ElementType, 3 > &out )
{
	return out.SetDirtyBBox( 
				out.GetDimensionExtents( 0 ).minimum,
				out.GetDimensionExtents( 1 ).minimum,
				out.GetDimensionExtents( 2 ).minimum,
				out.GetDimensionExtents( 0 ).maximum,
				out.GetDimensionExtents( 1 ).maximum,
				out.GetDimensionExtents( 2 ).maximum
			);
}

template< typename ElementType, unsigned dim >
WriterBBoxInterface *
ApplyWriterBBoxFunc(  const Image< ElementType, 4 > &out )
{
	return out.SetDirtyBBox( 
				out.GetDimensionExtents( 0 ).minimum,
				out.GetDimensionExtents( 1 ).minimum,
				out.GetDimensionExtents( 2 ).minimum,
				out.GetDimensionExtents( 3 ).minimum,
				out.GetDimensionExtents( 0 ).maximum,
				out.GetDimensionExtents( 1 ).maximum,
				out.GetDimensionExtents( 2 ).maximum,
				out.GetDimensionExtents( 3 ).maximum
			);
}
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
template< typename InputImageType, typename OutputImageType >
ReaderBBoxInterface::Ptr
ImageFilterWholeAtOnce< InputImageType, OutputImageType >
::ApplyReaderBBox( const InputImageType &in )
{
	return ApplyReaderBBoxFunc< InputImageType::Element, InputImageType::Dimension >( in );
}

template< typename InputImageType, typename OutputImageType >
WriterBBoxInterface *
ImageFilterWholeAtOnce< InputImageType, OutputImageType >
::ApplyWriterBBox( OutputImageType &out )
{
	return ApplyWriterBBoxFunc< OutputImageType::Element, OutputImageType::Dimension >( out );
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_ABSTRACT_IMAGE_FILTERS_H*/

