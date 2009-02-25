
#ifndef SNAKE_SEGMENTATION_FILTER_H
#error File SnakeSegmentationFilter.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template < typename ElementType >
const typename SnakeSegmentationFilter< ElementType >::InputImageType&
SnakeSegmentationFilter< ElementType >::GetInputImage( uint32 idx )const
{
	return this->GetInputDataSet< InputImageType >( idx );
}

template < typename ElementType >
void 
SnakeSegmentationFilter< ElementType >::ReleaseInputImage( uint32 idx )const
{
	this->ReleaseInputDataSet( idx );
}

/*template < typename ElementType >
typename SnakeSegmentationFilter< ElementType >::OutputDataset&
SnakeSegmentationFilter< ElementType >::GetOutputGDataset()const
{
	return this->GetOuputDataSet< OutputDataset >( 0 );
}*/

template < typename ElementType >
void 
SnakeSegmentationFilter< ElementType >::ReleaseOutputGDataset()const
{
	this->ReleaseOutputDataSet( 0 );
}
	
template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >::ExecutionThreadMethod()
{
	//TODO locking
	for( int32 i = _minSlice; i < _maxSlice; ++i ) {

		//ProcessSlice()

	}
}

template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >::PrepareOutputDatasets()
{
	int32 _minSlice = in[0]->GetDimensionExtents(2).minimum;
	int32 _maxSlice = in[0]->GetDimensionExtents(2).maximum;
	for( unsigned i=1; i<InCount; ++i ) {
		_minSlice = Max( _minSlice, in[i]->GetDimensionExtents(2).minimum );
		_maxSlice = Max( _maxSlice, in[i]->GetDimensionExtents(2).maximum );
	}

	this->out->UpgradeToExclusiveLock();
		//GeometryDataSetFactory::ChangeSliceCount( (*this->out), _minSlice, _maxSlice );
	this->out->DowngradeFromExclusiveLock();
}

template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	utype = AbstractPipeFilter::RECALCULATION;
	this->_callPrepareOutputDatasets = true;

	for( unsigned i = 0; i < InCount; ++i ) {
		in[ i ] = &(this->GetInputImage( i ));
	}
	//out = &(this->GetOutputGDataset());
	
}

template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >::MarkChanges( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	for( unsigned i=0; i < InCount; ++i ) {
		readerBBox[i] = in[i]->GetWholeDirtyBBox();
	}
}

template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >::AfterComputation( bool successful )
{
	for( unsigned i=0; i < InCount; ++i ) {
	/*	_inTimestamp[ i ] = in[ i ]->GetStructureTimestamp();
		_inEditTimestamp[ i ] = in[ i ]->GetEditTimestamp();*/
		
		this->ReleaseInputImage( i );
	}
	PredecessorType::AfterComputation( successful );	
}

/*
template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >
::ProcessSlice( const typename SnakeSegmentationFilter< ElementType >::RegionType	&region, 
		typename SnakeSegmentationFilter< ElementType >::CurveType		&initialization, 
		typename SnakeSegmentationFilter< ElementType >::OutputDataset::ObjectsInSlice &slice 
		)
{
	slice.clear();
	slice.push_back( initialization );
}*/
	
} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*SNAKE_SEGMENTATION_FILTER_H*/

