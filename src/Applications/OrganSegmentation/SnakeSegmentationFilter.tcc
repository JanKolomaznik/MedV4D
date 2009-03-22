
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
const typename SnakeSegmentationFilter< ElementType, SecondElementType >::InputImageType&
SnakeSegmentationFilter< ElementType, SecondElementType >::GetInputImage( uint32 idx )const
{
	return this->GetInputDataSet< InputImageType >( idx );
}

template < typename ElementType, typename SecondElementType >
void 
SnakeSegmentationFilter< ElementType, SecondElementType >::ReleaseInputImage( uint32 idx )const
{
	this->ReleaseInputDataSet( idx );
}

template < typename ElementType, typename SecondElementType >
typename SnakeSegmentationFilter< ElementType, SecondElementType >::OutputDatasetType&
SnakeSegmentationFilter< ElementType, SecondElementType >::GetOutputGDataset()const
{
	return this->GetOutputDataSet< OutputDatasetType >( 0 );
}

template < typename ElementType, typename SecondElementType >
void 
SnakeSegmentationFilter< ElementType, SecondElementType >::ReleaseOutputGDataset()const
{
	this->ReleaseOutputDataSet( 0 );
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
	
	out = &(this->GetOutputGDataset());

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
		
		this->ReleaseInputImage( i );
	}
	this->ReleaseOutputGDataset();
	PredecessorType::AfterComputation( successful );	
}

template < typename ElementType, typename SecondElementType >
void 
SnakeSegmentationFilter< ElementType, SecondElementType >::ComputeStatistics( Vector<int32, 3> p, float32 &E, float32 &var )
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
bool
SnakeSegmentationFilter< ElementType, SecondElementType >::ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype )
{
	if( !CanContinue() ) return false;

	if ( !(readerBBox[0]->WaitWhileDirty() == MS_MODIFIED ) ) {
		writerBBox->SetState( MS_CANCELED );
		return false;
	}
	
	CurveType northSpline = CreateSquareControlPoints( 3.0f );
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

	writerBBox->SetModified();
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
	float32 extent = Max( tmp[0], tmp[1] );

	//Initialization and setup
	SnakeAlgorithm algorithm;

	
	//Distribution settings
	algorithm.SetZCoordinate( sliceNumber * tmp[2] );
	algorithm.SetTransformation( GetTransformation( _northPole, _southPole ) );
	algorithm.SetModel( GetProbabilityModel() );
	algorithm.SetBalance( GetShapeIntensityBalance() );
	/*
	algorithm.SetInE( 1060 );
	algorithm.SetInVar( 900 );
	algorithm.SetOutE( 910 );
	algorithm.SetOutVar( 1600 );
	*/


	/*algorithm.SetStepScale( 1.0 );
	algorithm.SetSampleRate( 5 );
	algorithm.SetMaxStepScale( 2.0 );
	algorithm.SetMaxSegmentLength( 20 );
	algorithm.SetMinSegmentLength( 10 );*/

	algorithm.SetSampleRate( 6 );
	algorithm.SetMaxStepScale( 1.2 );
	algorithm.SetStepScale( algorithm.GetMaxStepScale() / 2.0 );
	algorithm.SetMaxSegmentLength( GetPrecision() * extent * 2.0 );
	algorithm.SetMinSegmentLength( GetPrecision() * extent );

	algorithm.SetSelfIntersectionTestPeriod( 1 );
	algorithm.SetSegmentLengthsTestPeriod( 3 );

	algorithm.SetGamma( 0.8f );
	algorithm.SetImageEnergyBalance( 1.0f );
	algorithm.SetInternalEnergyBalance( 0.0f );
	algorithm.SetConstrainEnergyBalance( 0.0f );
	//algorithm.SetRegionStatRegion( in[0]->GetSlice( sliceNumber ) );
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


/*template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >
::ProcessSlice( //const typename SnakeSegmentationFilter< ElementType, SecondElementType >::RegionType	&region, 
		typename SnakeSegmentationFilter< ElementType, SecondElementType >::CurveType		&initialization, 
		typename SnakeSegmentationFilter< ElementType, SecondElementType >::OutputDatasetType::ObjectsInSlice &slice 
		)*/

template < typename ElementType, typename SecondElementType >
void
SnakeSegmentationFilter< ElementType, SecondElementType >
::ProcessSlice( 
			typename SnakeSegmentationFilter< ElementType, SecondElementType >::CurveType &initialization, 
			typename SnakeSegmentationFilter< ElementType, SecondElementType >::ObjectsInSlice &slice 
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

