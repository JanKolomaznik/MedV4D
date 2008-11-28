/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AImageSliceFilter.tcc 
 * @{ 
 **/

#ifndef _A_IMAGE_SLICE_FILTER_H
#error File AImageSliceFilter.tcc cannot be included directly!
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
AImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::AImageSliceFilter( typename AImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >::Properties *prop ) 
	: PredecessorType( prop )
{
	//TODO - check intervals of parameters - throw exceptions
}

template< typename InputElementType, typename OutputImageType >
bool
AImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE /*utype*/ )
{
	size_t i = 0;
	unsigned sliceComputationNeighbourCount = GetProperties()._sliceComputationNeighbourCount;
	for( i = 0; i < _actualComputationGroups.size(); ++i )
	{
		SliceComputationRecord & record = _actualComputationGroups[ i ];

		//Wait until input area is ready
		if ( !(record.inputBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
			goto cleanup;
		}

		for( int32 slice = record.firstSlice; slice <= record.lastSlice; ++slice )
		{
			int32 minZ = Max( slice - (int32)sliceComputationNeighbourCount, this->in->GetDimensionExtents( 2 ).minimum );
			int32 maxZ = Min( slice + (int32)sliceComputationNeighbourCount + 1, this->in->GetDimensionExtents( 2 ).maximum );

			ImageRegion< InputImageType, 3 > region = 
				this->in->GetSubRegion(	
					this->in->GetDimensionExtents( 0 ).minimum,
					this->in->GetDimensionExtents( 1 ).minimum,
					minZ,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
					maxZ );

			bool result = ProcessSlice( region, *(this->out), slice - minZ );

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
AImageSliceFilter< Image< InputElementType, 3 >, OutputImageType >
::MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype )
{

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
					record.firstSlice - sliceComputationNeighbourCount,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
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
				record.firstSlice - sliceComputationNeighbourCount,
				this->in->GetDimensionExtents( 0 ).maximum,
				this->in->GetDimensionExtents( 1 ).maximum,
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
					record.firstSlice - sliceComputationNeighbourCount,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
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
template< typename InputElementType, typename OutputElementType >
AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::AImageSliceFilterIExtents( typename AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >::Properties *prop ) 
	: PredecessorType( prop )
{
	//TODO - check intervals of parameters - throw exceptions
}

template< typename InputElementType, typename OutputElementType >
bool
AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE /*utype*/ )
{
	unsigned sliceComputationNeighbourCount = GetProperties()._sliceComputationNeighbourCount;
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
			int32 minZ = Max( slice - (int32)sliceComputationNeighbourCount, this->in->GetDimensionExtents( 2 ).minimum );
			int32 maxZ = Min( slice + (int32)sliceComputationNeighbourCount + 1, this->in->GetDimensionExtents( 2 ).maximum );

			ImageRegion< InputElementType, 3 > region1 = 
				this->in->GetSubRegion(	
					this->in->GetDimensionExtents( 0 ).minimum,
					this->in->GetDimensionExtents( 1 ).minimum,
					minZ,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
					maxZ );

			ImageRegion< OutputElementType, 3 > regionTmp = 
				this->out->GetSubRegion(	
					this->out->GetDimensionExtents( 0 ).minimum,
					this->out->GetDimensionExtents( 1 ).minimum,
					slice,
					this->out->GetDimensionExtents( 0 ).maximum,
					this->out->GetDimensionExtents( 1 ).maximum,
					slice+1 );

			ImageRegion< OutputElementType, 2 > region2 = regionTmp.GetSlice( 0 );

			bool result = ProcessSlice( region1, region2, slice - minZ );

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

template< typename InputElementType, typename OutputElementType >
void
AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype )
{

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
					record.firstSlice - sliceComputationNeighbourCount,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
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
				record.firstSlice - sliceComputationNeighbourCount,
				this->in->GetDimensionExtents( 0 ).maximum,
				this->in->GetDimensionExtents( 1 ).maximum,
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
					record.firstSlice - sliceComputationNeighbourCount,
					this->in->GetDimensionExtents( 0 ).maximum,
					this->in->GetDimensionExtents( 1 ).maximum,
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

template< typename InputElementType, typename OutputElementType >
WriterBBoxInterface &
AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::GetComputationGroupWriterBBox(  SliceComputationRecord & record )
{
	return this->out->SetDirtyBBox( this->in->GetDimensionExtents( 0 ).minimum,
			this->in->GetDimensionExtents( 1 ).minimum,
			this->in->GetDimensionExtents( 0 ).maximum,
			record.firstSlice,
			this->in->GetDimensionExtents( 1 ).maximum,
			record.lastSlice
			);
}

template< typename InputElementType, typename OutputElementType >
void
AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
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

//template< typename InputElementType, typename OutputElementType >
//AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
//::AImageSliceFilterIExtents( typename AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >::Properties *prop ) 
//	: PredecessorType( prop )
//{
//
//}
//
//template< typename InputElementType, typename OutputElementType >
//WriterBBoxInterface &
//AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
//::GetComputationGroupWriterBBox(  SliceComputationRecord & record )
//{
//	return this->out->SetDirtyBBox( this->in->GetDimensionExtents( 0 ).minimum,
//			this->in->GetDimensionExtents( 1 ).minimum,
//			this->in->GetDimensionExtents( 0 ).maximum,
//			this->in->GetDimensionExtents( 1 ).maximum,
//			record.firstSlice,
//			record.lastSlice
//			);
//}
//
//template< typename InputElementType, typename OutputElementType >
//void
//AImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
//::PrepareOutputDatasets()
//{
//	PredecessorType::PrepareOutputDatasets();
//
//	int32 minimums[3];
//	int32 maximums[3];
//	float32 voxelExtents[3];
//
//	for( unsigned i=0; i < 3; ++i ) {
//		const DimensionExtents & dimExt = this->in->GetDimensionExtents( i );
//
//		minimums[i] = dimExt.minimum;
//		maximums[i] = dimExt.maximum;
//		voxelExtents[i] = dimExt.elementExtent;
//	}
//	this->SetOutputImageSize( minimums, maximums, voxelExtents );
//}

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_A_IMAGE_SLICE_FILTER_H*/


/** @} */

