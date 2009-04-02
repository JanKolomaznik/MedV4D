
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

template < typename ElementType, typename SecondElementType >
SnakeSegmentationFilter< ElementType, SecondElementType >
::SnakeSegmentationFilter() : PredecessorType( new Properties() )
{
	this->_name = "SnakeSegmentationFilter";

	InputPortType *inPort = new InputPortType();
	_inputPorts.AppendPort( inPort );

	EdgePortType *edgePort = new EdgePortType();
	_inputPorts.AppendPort( edgePort );
	
	OutputPortType *outPort = new OutputPortType();
	_outputPorts.AppendPort( outPort );

}

template < typename ElementType, typename SecondElementType >
SnakeSegmentationFilter< ElementType, SecondElementType >
::SnakeSegmentationFilter( typename SnakeSegmentationFilter< ElementType, SecondElementType >::Properties *prop ) 
: PredecessorType( prop ) 
{
	this->_name = "SnakeSegmentationFilter";

	InputPortType *inPort = new InputPortType();
	_inputPorts.AppendPort( inPort );

	EdgePortType *edgePort = new EdgePortType();
	_inputPorts.AppendPort( edgePort );
	
	OutputPortType *outPort = new OutputPortType();
	_outputPorts.AppendPort( outPort );
}


template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >::PrepareOutputDatasets()
{
	_minSlice = in->GetDimensionExtents(2).minimum;
	_maxSlice = in->GetDimensionExtents(2).maximum;

	_minSlice = Max( _minSlice, inEdge->GetDimensionExtents(2).minimum );
	_maxSlice = Min( _maxSlice, inEdge->GetDimensionExtents(2).maximum );

	this->out->UpgradeToExclusiveLock();
		GeometryDataSetFactory::ChangeSliceCount( (*this->out), _minSlice, _maxSlice );
	this->out->DowngradeFromExclusiveLock();
}

template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >::BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	utype = AbstractPipeFilter::RECALCULATION;
	this->_callPrepareOutputDatasets = true;

	in = &(this->GetInputDataSet< InputImageType >( 0 ));
	inEdge = &(this->GetInputDataSet< EdgeImageType >( 1 ));
	
	out = &(this->GetOutputDataSet<OutputDatasetType>(0));

	_northPole[ 0 ] = GetSecondPoint()[ 0 ];
	_northPole[ 1 ] = GetSecondPoint()[ 1 ];
	_northPole[ 2 ] = GetSecondSlice() * in->GetElementExtents()[2];

	_southPole[ 0 ] = GetFirstPoint()[ 0 ];
	_southPole[ 1 ] = GetFirstPoint()[ 1 ];
	_southPole[ 2 ] = GetFirstSlice() * in->GetElementExtents()[2];
	
}

template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >::MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype )
{
	readerBBox[0] = in->GetWholeDirtyBBox();
	readerBBox[1] = inEdge->GetWholeDirtyBBox();
	
	writerBBox = &(out->SetWholeDirtyBBox());
}

template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >::AfterComputation( bool successful )
{
	for( unsigned i=0; i < InCount; ++i ) {
	/*	_inTimestamp[ i ] = in[ i ]->GetStructureTimestamp();
		_inEditTimestamp[ i ] = in[ i ]->GetEditTimestamp();*/
		
		this->ReleaseInputDataSet( i );
	}
	this->ReleaseOutputDataSet( 0 );
	PredecessorType::AfterComputation( successful );	
}


template < typename ElementType, typename SecondElementType >
typename SnakeSegmentationFilter< ElementType, SecondElementType >::CurveType
SnakeSegmentationFilter< ElementType, SecondElementType >::CreateSquareControlPoints( float32 radius )
{
	CurveType result;
	result.SetCyclic( true );
	result.AddPoint( Coordinates( radius, radius ) );
	result.AddPoint( Coordinates( radius, -radius ) );
	result.AddPoint( Coordinates( -radius, -radius ) );
	result.AddPoint( Coordinates( -radius, radius ) );
	
	return result;
}

template < typename ElementType, typename SecondElementType >
typename SnakeSegmentationFilter< ElementType, SecondElementType >::CurveType
SnakeSegmentationFilter< ElementType, SecondElementType >::CreateCircleControlPoints( float32 radius, int segments )
{
	CurveType result;
	result.SetCyclic( true );

	float angle = -2*PI / (float)segments;

	for( int i = 0; i < segments; ++i ) {
		Coordinates np = radius * Coordinates(sin(angle*i), cos(angle*i));
		result.AddPoint( np );
	}
	return result;
}

template < typename ElementType, typename SecondElementType >
bool
SnakeSegmentationFilter< ElementType, SecondElementType >::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	if( !CanContinue() ) return false;

	if ( !(readerBBox[0]->WaitWhileDirty() == MS_MODIFIED ) ) {
		writerBBox->SetState( MS_CANCELED );
		return false;
	}

	typename InputImageType::ElementExtentsType tmp =  in->GetElementExtents();
	_extent = Max( tmp[0], tmp[1] );
	
	bool result = false;

	if( GetSeparateSliceInit() ) {
		result = ParallelizableComputation();
	} else {
		result = SequentialComputation();
	}

	writerBBox->SetModified();
	return result;
}

template < typename ElementType, typename SecondElementType >
bool
SnakeSegmentationFilter< ElementType, SecondElementType >::SequentialComputation()
{
	CurveType northSpline = CreateCircleControlPoints( 6.0f, 12 );
	CurveType southSpline = northSpline;

	northSpline.Move( GetSecondPoint() );
	southSpline.Move( GetFirstPoint() );
	unsigned stepCount = (_maxSlice - _minSlice) / 2;
	for( unsigned step = 0; step < stepCount; ++step ) {
			D_PRINT( "Segmentation step " << step );
		ProcessSlice( _minSlice + step, southSpline );
		ProcessSlice( _maxSlice - step - 1, northSpline );
	}
	if( (_minSlice + stepCount) != (_maxSlice - stepCount-2) ) {//TODO check !!!
		ProcessSlice( _minSlice + stepCount, southSpline );
	}

	return true;
}

template < typename ElementType, typename SecondElementType >
bool
SnakeSegmentationFilter< ElementType, SecondElementType >::ParallelizableComputation()
{
	D_PRINT( "First slice = " << GetFirstSlice() << "; Second slice = " << GetSecondSlice() );
	for( int32 slice = _minSlice; slice < _maxSlice; ++slice ) {
			D_PRINT( "Segmentation in slice " << slice );
		ProcessSlice( slice );
	}

	return true;
}

template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >
::ProcessSlice( 
		int32	sliceNumber,
		typename SnakeSegmentationFilter< ElementType, SecondElementType >::CurveType &initialization 
		)
{
	static const unsigned ResultSampleRate = 8;

	ObjectsInSlice &slice = this->out->GetSlice( sliceNumber );
	slice.clear();
	
	typename InputImageType::ElementExtentsType tmp =  in->GetElementExtents();

	//Initialization and setup
	SnakeAlgorithm algorithm;

	
	//Distribution settings
	algorithm.SetZCoordinate( sliceNumber * tmp[2] );
	algorithm.SetTransformation( GetTransformation( _northPole, _southPole ) );
	algorithm.SetModel( GetProbabilityModel() );
	algorithm.SetBalance( GetShapeIntensityBalance() );

	/*algorithm.SetStepScale( 1.0 );
	algorithm.SetSampleRate( 5 );
	algorithm.SetMaxStepScale( 2.0 );
	algorithm.SetMaxSegmentLength( 20 );
	algorithm.SetMinSegmentLength( 10 );*/

	algorithm.SetSampleRate( 6 );
	algorithm.SetMaxStepScale( 1.2 );
	algorithm.SetStepScale( algorithm.GetMaxStepScale() / 2.0 );
	algorithm.SetMaxSegmentLength( GetPrecision() * _extent * 2.0 );
	algorithm.SetMinSegmentLength( GetPrecision() * _extent );

	algorithm.SetSelfIntersectionTestPeriod( 1 );
	algorithm.SetSegmentLengthsTestPeriod( 3 );

	algorithm.SetGamma( 0.8f );
	algorithm.SetImageEnergyBalance( 1.0f );
	algorithm.SetInternalEnergyBalance( 0.0f );
	algorithm.SetConstrainEnergyBalance( 0.0f );
	algorithm.SetRegionStat( in->GetSlice( sliceNumber ) );
	algorithm.SetRegionEdge( inEdge->GetSlice( sliceNumber ) );
	algorithm.SetAlpha( GetEdgeRegionBalance() );

	algorithm.SetCalmDownInterval( 20 );
	algorithm.SetMaxStepCount( 60 );


	
	
	algorithm.Initialize( initialization );
	//****************************************
	//**** COMPUTATION ***********************

	/*unsigned i = 0;
	while( 60 > i ) {
		i = algorithm.Step();
		if( i % 5 == 0 ) {
			const CurveType &pom = algorithm.GetCurrentCurve();
			slice.push_back( pom );
			slice[slice.size()-1].Sample( ResultSampleRate );
		}
	}*/

	if( algorithm.Converge() ) {
		const CurveType &result = algorithm.GetCurrentCurve();

		slice.push_back( result );
		slice[0].Sample( ResultSampleRate );
		initialization = result;
	} else {
		const CurveType &result = algorithm.GetCurrentCurve();
		slice.push_back( result );
	}

	slice[0].Sample( ResultSampleRate );
}

template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >
::ProcessSlice( int32 sliceNumber )
{
	static const unsigned ResultSampleRate = 8;

	ObjectsInSlice &slice = this->out->GetSlice( sliceNumber );
	slice.clear();

	typename InputImageType::ElementExtentsType tmp =  in->GetElementExtents();

	Transformation trans = GetTransformation( _northPole, _southPole );

	//Initialization and setup
	SnakeAlgorithm algorithm;
	
	//Distribution settings
	algorithm.SetZCoordinate( sliceNumber * tmp[2] );
	algorithm.SetTransformation( trans );
	algorithm.SetModel( GetProbabilityModel() );

	algorithm.SetRegionStat( in->GetSlice( sliceNumber ) );
	algorithm.SetRegionEdge( inEdge->GetSlice( sliceNumber ) );

	algorithm.SetStepScaleAlpha( 0.9f );
	algorithm.SetStepScaleBeta( 0.25f );
	//Computation
	
	float32 t = static_cast<float32>(sliceNumber - GetFirstSlice())/(GetSecondSlice() - GetFirstSlice());
	Coordinates center = GetFirstPoint() + t * (GetSecondPoint() - GetFirstPoint());

	Vector<float32, 3 > tmpPos = trans( Vector<float32, 3 >( 0, 0, sliceNumber * tmp[2] ) );
	tmpPos[0] = 0;
	tmpPos[1] = 0;

	CurveType init = CreateCircleControlPoints( Min(GetProbabilityModel()->GetLayerProbRadius( tmpPos ) * 2, 18.0f), 12 );
	tmpPos = trans.GetInversion( GetProbabilityModel()->GetLayerProbCenter( tmpPos ) );

	//init.Move( Coordinates( tmpPos[0], tmpPos[1] ) );
	init.Move( center );

	algorithm.SetSampleRate( 3 );
	algorithm.Initialize( init );

	FindInitialization( algorithm );

		//slice.push_back( algorithm.GetCurrentCurve() );
		//slice[slice.size()-1].Sample( ResultSampleRate );

	ComputeRawShape( algorithm );

		//slice.push_back( algorithm.GetCurrentCurve() );
		//slice[slice.size()-1].Sample( ResultSampleRate );

	FinishComputation( algorithm );

	slice.push_back( algorithm.GetCurrentCurve() );
	slice[slice.size()-1].Sample( ResultSampleRate );

	/*slice.push_back( init );
	slice[1].Sample( ResultSampleRate );*/
}

template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >
::FindInitialization( typename SnakeSegmentationFilter< ElementType, SecondElementType >::SnakeAlgorithm &algorithm )
{
	algorithm.SetBalance( 0.5 );

	algorithm.SetSampleRate( 3 );
	algorithm.SetMaxStepScale( 10.0f );
	algorithm.SetStepScale( algorithm.GetMaxStepScale() / 2.0 );
	algorithm.SetMaxSegmentLength( 30 );
	algorithm.SetMinSegmentLength( 10 );

	algorithm.SetSelfIntersectionTestPeriod( 3 );
	algorithm.SetSegmentLengthsTestPeriod( 4 );

	algorithm.SetGamma( 0.8f );
	algorithm.SetImageEnergyBalance( 1.0f );
	algorithm.SetInternalEnergyBalance( 0.0f );
	algorithm.SetConstrainEnergyBalance( 0.3f );
	algorithm.SetAlpha( 1.0f );



	unsigned i = 0;
	while( 30 > i && i >= 0 ) {
		i = algorithm.Step();
	}
}

template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >
::ComputeRawShape( typename SnakeSegmentationFilter< ElementType, SecondElementType >::SnakeAlgorithm &algorithm )
{
	algorithm.SetBalance( GetShapeIntensityBalance() );

	algorithm.SetSampleRate( 5 );
	algorithm.SetMaxStepScale( 1.5 );
	algorithm.SetStepScale( algorithm.GetMaxStepScale() / 2.0 );
	algorithm.SetMaxSegmentLength( GetPrecision() * _extent * 2.0 );
	algorithm.SetMinSegmentLength( GetPrecision() * _extent );

	algorithm.SetSelfIntersectionTestPeriod( 1 );
	algorithm.SetSegmentLengthsTestPeriod( 1 );

	algorithm.SetGamma( 0.8f );
	algorithm.SetImageEnergyBalance( 1.0f );
	algorithm.SetInternalEnergyBalance( 0.0f );
	algorithm.SetConstrainEnergyBalance( 0.0f );
	algorithm.SetAlpha( 0.75f );

	//algorithm.SetCalmDownInterval( 20 );
	//algorithm.SetMaxStepCount( 60 );


	unsigned i = 0;
	while( 55 > i && i >= 0 ) {
		i = algorithm.Step();
	}

}

template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >
::FinishComputation( typename SnakeSegmentationFilter< ElementType, SecondElementType >::SnakeAlgorithm &algorithm )
{

	algorithm.SetBalance( 0.9f );

	algorithm.SetSampleRate( 8 );
	algorithm.SetMaxStepScale( 1.0f );
	algorithm.SetStepScale( algorithm.GetMaxStepScale() / 2.0 );
	algorithm.SetMaxSegmentLength( GetPrecision() * _extent * 2.0 );
	algorithm.SetMinSegmentLength( GetPrecision() * _extent );

	algorithm.SetSelfIntersectionTestPeriod( 1 );
	algorithm.SetSegmentLengthsTestPeriod( 1 );

	algorithm.SetGamma( 0.8f );
	algorithm.SetImageEnergyBalance( 1.0f );
	algorithm.SetInternalEnergyBalance( 0.0f );
	algorithm.SetConstrainEnergyBalance( 0.0f );
	algorithm.SetAlpha( GetEdgeRegionBalance() * 0.5 );

	algorithm.SetCalmDownInterval( 20 );
	algorithm.SetMaxStepCount( 80 );


/*	unsigned i = 0;
	while( 75 > i && i >= 0 ) {
		i = algorithm.Step();
	}*/
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*SNAKE_SEGMENTATION_FILTER_H*/

