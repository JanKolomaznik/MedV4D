
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

template < typename ElementType >
void 
SnakeSegmentationFilter< ElementType >::ComputeStatistics( Vector<int32, 3> p, float32 &E, float32 &var )
{
	static const int32 Radius = 5;
	float32 sum = 0.0f;
	Vector<int32, 3> i = p;
	for( i[0] = p[0] - Radius; i[0] <= p[0] + Radius; ++i[0] ) {
		for( i[1] = p[1] - Radius; i[1] <= p[1] + Radius; ++i[1] ) {
			sum += in[0]->GetElement( i );
		}
	}
	E = sum / Sqr(2*Radius +1);
	for( i[0] = p[0] - Radius; i[0] <= p[0] + Radius; ++i[0] ) {
		for( i[1] = p[1] - Radius; i[1] <= p[1] + Radius; ++i[1] ) {
			sum += Sqr( in[0]->GetElement( i )) ;
		}
	}
	var = sum / Sqr(2*Radius +1) - Sqr( E );
}
	
/*
template < typename ElementType >
bool
SnakeSegmentationFilter< ElementType >::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	if( !CanContinue() ) return false;

	//TODO locking
	
	Vector<int32, 3> p( 
		GetInsidePoint()[0]/in[0]->GetDimensionExtents(0).elementExtent,
		GetInsidePoint()[1]/in[0]->GetDimensionExtents(1).elementExtent,
		GetInsidePointSlice()
		);
	ComputeStatistics( p, _inEstimatedValue, _inVariation );
	p =  Vector<int32, 3>( 
		GetOutsidePoint()[0]/in[0]->GetDimensionExtents(0).elementExtent,
		GetOutsidePoint()[1]/in[0]->GetDimensionExtents(1).elementExtent,
		GetOutsidePointSlice()
		);
	ComputeStatistics( p, _outEstimatedValue, _outVariation );

	LOG( "In region stats : E = " << _inEstimatedValue << "; var = " << _inVariation );
	LOG( "Out region stats : E = " << _outEstimatedValue << "; var = " << _outVariation );
	
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
}*/

template < typename ElementType >
typename SnakeSegmentationFilter< ElementType >::CurveType
SnakeSegmentationFilter< ElementType >::CreateSquareControlPoints( float32 radius )
{
	CurveType result;
	result.SetCyclic( true );
	result.AddPoint( Coordinates( radius, radius ) );
	result.AddPoint( Coordinates( radius, -radius ) );
	result.AddPoint( Coordinates( -radius, -radius ) );
	result.AddPoint( Coordinates( -radius, radius ) );

	return result;
}

template < typename ElementType >
bool
SnakeSegmentationFilter< ElementType >::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	if( !CanContinue() ) return false;

	//TODO locking
	
	CurveType northSpline = CreateSquareControlPoints( 0.5 );
	CurveType southSpline = northSpline;

	northSpline.Move( GetSecondPoint() );
	southSpline.Move( GetFirstPoint() );
	unsigned stepCount = (_maxSlice - _minSlice) / 2;
	for( unsigned step = 0; step < stepCount; ++step ) {
		ProcessSlice( _minSlice + step, southSpline );
		ProcessSlice( _maxSlice - step - 1, northSpline );
	}
	/*if( (_minSlice + stepCount) == (_maxSlice - stepCount-1) ) {//TODO check !!!
		ProcessSlice( _minSlice + stepCount, southSpline );
	}*/
	return true;
}

template < typename ElementType >
void
SnakeSegmentationFilter< ElementType >
::ProcessSlice( 
		int32	sliceNumber,
		typename SnakeSegmentationFilter< ElementType >::CurveType &initialization 
		)
{
	static const unsigned ResultSampleRate = 5;
	//Initialization and setup
	SnakeAlgorithm algorithm;
	
	algorithm.Initialize( initialization );
	//****************************************
	//**** COMPUTATION ***********************

	while( 40 > algorithm.Step() ) {
		/* empty */
	}


	//****************************************
	//Result processing
	const CurveType &result = algorithm.GetCurrentCurve();

	ObjectsInSlice &slice = this->out->GetSlice( sliceNumber );
	slice.clear();
	slice.push_back( result );
	slice[0].Sample( ResultSampleRate );
	initialization = result;
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

