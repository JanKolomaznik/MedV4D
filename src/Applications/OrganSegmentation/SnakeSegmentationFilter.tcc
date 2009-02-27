
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
SnakeSegmentationFilter< ElementType >
::SnakeSegmentationFilter() : PredecessorType( new Properties() )
{
	//TODO check if OK
	for( unsigned i = 0; i < InCount; ++i ) {
		InputPortType *inPort = new InputPortType();
		_inputPorts.AddPort( inPort );
	}
	
	OutputPortType *outPort = new OutputPortType();
	_outputPorts.AddPort( outPort );

}

template < typename ElementType >
SnakeSegmentationFilter< ElementType >
::SnakeSegmentationFilter( typename SnakeSegmentationFilter< ElementType >::Properties *prop ) 
: PredecessorType( prop ) 
{
	//TODO check if OK
	for( unsigned i = 0; i < InCount; ++i ) {
		InputPortType *inPort = new InputPortType();
		_inputPorts.AddPort( inPort );
	}
	
	OutputPortType *outPort = new OutputPortType();
	_outputPorts.AddPort( outPort );
}

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

template < typename ElementType >
typename SnakeSegmentationFilter< ElementType >::OutputDatasetType&
SnakeSegmentationFilter< ElementType >::GetOutputGDataset()const
{
	return this->GetOutputDataSet< OutputDatasetType >( 0 );
}

template < typename ElementType >
void 
SnakeSegmentationFilter< ElementType >::ReleaseOutputGDataset()const
{
	this->ReleaseOutputDataSet( 0 );
}
	
template < typename ElementType >
bool
SnakeSegmentationFilter< ElementType >::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	if( !CanContinue() ) return false;

	//TODO locking
	
	CurveType initialization;
	initialization.SetCyclic();
	initialization.AddPoint( typename CurveType::PointType( -5, -5 ) );
	initialization.AddPoint( typename CurveType::PointType( 5, -5 ) );
	initialization.AddPoint( typename CurveType::PointType( 5, 5 ) );
	initialization.AddPoint( typename CurveType::PointType( -5, 5 ) );

	//std::cout << "Min slice = " << _minSlice << "; Max slice = " << _maxSlice << "\n";

	float32 center = ((float32)(_maxSlice + _minSlice))/2.0f;
	float32 s = 6.0f/(_maxSlice - _minSlice);

	Coordinates trans = GetFirstPoint();
	Coordinates relMove = (1.0f/(GetSecondSlice()-GetFirstSlice()) ) * (GetSecondPoint() - GetFirstPoint());
	for( int32 i = _minSlice; i < _maxSlice; ++i ) {
		float32 scaleFactor = 4.0f - s*Abs( i - center );
		CurveType pom = initialization;
		pom.Move(trans);
		pom.Scale( Vector< float32, 2>( scaleFactor ), trans );
		trans += relMove;

		ProcessSlice( pom, this->out->GetSlice( i ) );
	}

	return true;
}

template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >::PrepareOutputDatasets()
{
	_minSlice = in[0]->GetDimensionExtents(2).minimum;
	_maxSlice = in[0]->GetDimensionExtents(2).maximum;
	for( unsigned i=1; i<InCount; ++i ) {
		_minSlice = Max( _minSlice, in[i]->GetDimensionExtents(2).minimum );
		_maxSlice = Max( _maxSlice, in[i]->GetDimensionExtents(2).maximum );
	}

	this->out->UpgradeToExclusiveLock();
		GeometryDataSetFactory::ChangeSliceCount( (*this->out), _minSlice, _maxSlice );
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
	out = &(this->GetOutputGDataset());
	
}

template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >::MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype )
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


/*template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >
::ProcessSlice( //const typename SnakeSegmentationFilter< ElementType >::RegionType	&region, 
		typename SnakeSegmentationFilter< ElementType >::CurveType		&initialization, 
		typename SnakeSegmentationFilter< ElementType >::OutputDatasetType::ObjectsInSlice &slice 
		)*/

template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >
::ProcessSlice( 
			typename SnakeSegmentationFilter< ElementType >::CurveType &initialization, 
			typename SnakeSegmentationFilter< ElementType >::ObjectsInSlice &slice 
			)
{
	slice.clear();
	slice.push_back( initialization );
	slice[0].Sample( 5 );
}
	
} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*SNAKE_SEGMENTATION_FILTER_H*/

