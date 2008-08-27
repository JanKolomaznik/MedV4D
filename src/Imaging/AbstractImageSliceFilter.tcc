#ifndef _ABSTRACT_IMAGE_SLICE_FILTER_H
#error File AbstractImageSliceFilter.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{

template< typename InputElementType, typename OutputImageType >
AbstractImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::AbstractImageSliceFilter( typename AbstractImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >::Properties *prop ) 
	: PredecessorType( prop )
{
	//TODO - check intervals of parameters - throw exceptions
}

template< typename InputElementType, typename OutputImageType >
bool
AbstractImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE /*utype*/ )
{
	size_t i = 0;
	for( i = 0; i < _actualComputationGroups.size(); ++i )
	{
		SliceComputationRecord & record = _actualComputationGroups[ i ];

		//Wait until input area is ready
		if ( !(record.inputBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
			goto cleanup;
		}

		for( int32 slice = record.firstSlice; slice <= record.lastSlice; ++slice )
		{
			bool result = ProcessSlice( 	
					*(this->in), 
					*(this->out),
					this->in->GetDimensionExtents( 0 ).minimum,
					this->in->GetDimensionExtents( 1 ).minimum,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
					slice 
					);
			if( !result ){
				goto cleanup;
			}
		}

		record.writerBBox->SetModified();
	}

	_actualComputationGroups.clear();

	return true;

cleanup:
	for( size_t j = i; j < _actualComputationGroups.size(); ++j )
	{
		_actualComputationGroups[ j ].writerBBox->SetState( MS_CANCELED );
	}
	_actualComputationGroups.clear();
	return false;
}

template< typename InputElementType, typename OutputImageType >
void
AbstractImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
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
			uint32 groupCount = ( dimExtents.maximum - dimExtents.minimum ) / computationGrouping;
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
AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::AbstractImageSliceFilterIExtents( typename AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >::Properties *prop ) 
	: PredecessorType( prop )
{

}

template< typename InputElementType, typename OutputElementType >
void
AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	//TODO
	PredecessorType::BeforeComputation( utype );	
}

template< typename InputElementType, typename OutputElementType >
WriterBBoxInterface &
AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
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
AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

	int32 minimums[3];
	int32 maximums[3];
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


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_IMAGE_SLICE_FILTER_H*/

