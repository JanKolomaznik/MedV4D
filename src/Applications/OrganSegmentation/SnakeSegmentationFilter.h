#ifndef SNAKE_SEGMENTATION_FILTER_H
#define SNAKE_SEGMENTATION_FILTER_H

#include "Imaging/Imaging.h"
#include <cmath>

namespace M4D
{

namespace Imaging
{

class ProbabilityFromModel
{
private:
	CanonicalProbModel 	*_model;
	float32 		_zCoord;
	Transformation		_transformation;
	Vector< float32, 3 >	_tmp;
	float32			_balance;
public:
	ProbabilityFromModel():_model( NULL ), _balance( 0.5 ) {}

	float32
	LogProbabilityRatio( int16 value, const Vector< float32, 2 > &pos )
	{ 
		_tmp[0] = pos[0];
		_tmp[1] = pos[1];
		_tmp[2] = _zCoord; //TODO - set with _zCoord

		//return _model->LogRatioProbabilityPosition( _transformation( _tmp ) ); 
		return _model->LogRatioProbabilityIntesityPosition( value, _transformation( _tmp ), _balance ); 
		//return _model->LogRatioProbabilityIntesity( value ); 
	}
	SIMPLE_GET_SET_METHODS( CanonicalProbModel*, Model, _model );
	SIMPLE_GET_SET_METHODS( float32, ZCoordinate, _zCoord );
	SIMPLE_GET_SET_METHODS( float32, Balance, _balance );
	SIMPLE_GET_SET_METHODS( Transformation, Transformation, _transformation );
};

class NormalDistribution
{
private:
	float32 _inE;
	float32	_inVar;

	float32 _outE;
	float32	_outVar;

	float32 _a;
	float32 _b;
	float32 _c;
public:
	NormalDistribution(): _inE( 0.0f ), _inVar( 1.0f ), _outE( 0.0f ), _outVar( 1.0f )
		{ Recalc(); }

	void
	SetInE( float32 E )
		{ _inE = E; Recalc(); }
	void
	SetInVar( float32 var )
		{ _inVar = var; Recalc(); }

	void
	SetOutE( float32 E )
		{ _outE = E; Recalc(); }
	void
	SetOutVar( float32 var )
		{ _outVar = var; Recalc(); }

	template< typename NType >
	float32
	LogProbabilityRatio( NType value )
	{ return Sqr(value) * _a + value * _b + _c; }

private:
	void
	Recalc()
	{ 
		_a = 1.0f / (2*_outVar) - 1.0f / (2*_inVar);
		_b = _inE / _inVar - _outE / _outVar;
		_c = log( _outVar / _inVar ) + Sqr( _outE )/(2*_outVar) - Sqr( _inE )/(2*_inVar);
	}

};


template < typename ElementType, typename SecondElementType >
class SnakeSegmentationFilter: public APipeFilter
{
public:
	typedef APipeFilter	PredecessorType;
	typedef	Imaging::Geometry::BSpline< float32, 2 >		CurveType;
	typedef SlicedGeometry< Imaging::Geometry::BSpline<float32,2> >	OutputDatasetType;
	typedef Image< ElementType, 3 >					InputImageType;
	typedef Image< SecondElementType, 3 >				EdgeImageType;
	typedef typename ImageTraits< InputImageType >::InputPort 	InputPortType;
	typedef typename ImageTraits< EdgeImageType >::InputPort 	EdgePortType;
	typedef OutputPortTyped< OutputDatasetType > 			OutputPortType;
	typedef ImageRegion< ElementType, 2 >				RegionType;
	typedef ImageRegion< SecondElementType, 2 >			EdgeRegionType;
	typedef typename OutputDatasetType::ObjectsInSlice		ObjectsInSlice;

	typedef Vector< float32, 2 >					Coordinates;


	static const unsigned InCount = 2;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): precision( -1 ), shapeIntensityBalance( 0.5 ), edgeRegionBalance( 0.5 ), probModel( NULL ), separateSliceInit(true) {}

		Coordinates		firstPoint;
		int32			firstSlice;
		Coordinates		secondPoint;
		int32			secondSlice;

		int32			precision;
		float32			shapeIntensityBalance;
		float32			edgeRegionBalance;
		CanonicalProbModel*	probModel;

		bool			separateSliceInit;
	};

	~SnakeSegmentationFilter() {}

	SnakeSegmentationFilter();
	SnakeSegmentationFilter( Properties * prop );

	GET_SET_PROPERTY_METHOD_MACRO( Coordinates, FirstPoint, firstPoint );
	GET_SET_PROPERTY_METHOD_MACRO( int32, FirstSlice, firstSlice );
	GET_SET_PROPERTY_METHOD_MACRO( Coordinates, SecondPoint, secondPoint );
	GET_SET_PROPERTY_METHOD_MACRO( int32, SecondSlice, secondSlice );

	GET_SET_PROPERTY_METHOD_MACRO( int32, Precision, precision );
	GET_SET_PROPERTY_METHOD_MACRO( float32, ShapeIntensityBalance, shapeIntensityBalance );
	GET_SET_PROPERTY_METHOD_MACRO( float32, EdgeRegionBalance, edgeRegionBalance );
	GET_SET_PROPERTY_METHOD_MACRO( CanonicalProbModel*, ProbabilityModel, probModel );
	GET_SET_PROPERTY_METHOD_MACRO( bool, SeparateSliceInit, separateSliceInit );

protected:

	typedef	ProbabilityFromModel					Distribution;
	//typedef NormalDistribution					Distribution;

	typedef M4D::Imaging::Algorithms::SegmentationEnergy< 
			CurveType,
			//M4D::Imaging::Algorithms::RegionImageEnergy< CurveType, RegionType, Distribution >,
			M4D::Imaging::Algorithms::UnifiedImageEnergy2< CurveType, RegionType, EdgeRegionType, Distribution >,
			M4D::Imaging::Algorithms::InternalCurveEnergy< CurveType >,
			M4D::Imaging::Algorithms::DummyEnergy3 >			EnergyModel;

	typedef M4D::Imaging::Algorithms::EnergicSnake< CurveType, EnergyModel >	SnakeAlgorithm;

	CurveType
	CreateSquareControlPoints( float32 radius );

	CurveType
	CreateCircleControlPoints( float32 radius, int segments );


	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );
	
	bool
	SequentialComputation();

	bool
	ParallelizableComputation();

	void
	PrepareOutputDatasets();
	
	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );
	
	void
	MarkChanges( AbstractPipeFilter::UPDATE_TYPE utype );

	void
	AfterComputation( bool successful );

	void
	ProcessSlice( 
			int32		sliceNumber,
			CurveType	&initialization 
			);

	void
	ProcessSlice( int32 sliceNumber );

	void
	FindInitialization( SnakeAlgorithm &algorithm );

	void
	ComputeRawShape( SnakeAlgorithm &algorithm );

	void
	FinishComputation( SnakeAlgorithm &algorithm );

	const InputImageType	*in;
	const EdgeImageType	*inEdge;
	OutputDatasetType	*out;
	int32 _minSlice;
	int32 _maxSlice;
	float32 _extent;

	Vector< float32, 3 >	_southPole;
	Vector< float32, 3 >	_northPole;

	ReaderBBoxInterface::Ptr readerBBox[ InCount ];
	WriterBBoxInterface		*writerBBox;

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "SnakeSegmentationFilter.tcc"

#endif //SNAKE_SEGMENTATION_FILTER_H
