#ifndef SNAKE_SEGMENTATION_FILTER_H
#define SNAKE_SEGMENTATION_FILTER_H

#include "Imaging.h"
#include <cmath>

namespace M4D
{

namespace Imaging
{

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


template < typename ElementType >
class SnakeSegmentationFilter: public APipeFilter
{
public:
	typedef APipeFilter	PredecessorType;
	typedef	Imaging::Geometry::BSpline< float32, 2 >		CurveType;
	typedef SlicedGeometry< Imaging::Geometry::BSpline<float32,2> >	OutputDatasetType;
	typedef Image< ElementType, 3 >					InputImageType;
	typedef typename ImageTraits< InputImageType >::InputPort 	InputPortType;
	typedef OutputPortTyped< OutputDatasetType > 			OutputPortType;
	typedef ImageRegion< ElementType, 2 >				RegionType;
	typedef typename OutputDatasetType::ObjectsInSlice		ObjectsInSlice;

	typedef Vector< float32, 2 >					Coordinates;


	static const unsigned InCount = 2;

	struct Properties : public PredecessorType::Properties
	{
		Coordinates	firstPoint;
		int32		firstSlice;
		Coordinates	secondPoint;
		int32		secondSlice;

		Coordinates	insidePoint;
		int32		insidePointSlice;
		Coordinates	outsidePoint;
		int32		outsidePointSlice;

		int32		precision;
	};

	~SnakeSegmentationFilter() {}

	SnakeSegmentationFilter();
	SnakeSegmentationFilter( Properties * prop );

	GET_SET_PROPERTY_METHOD_MACRO( Coordinates, FirstPoint, firstPoint );
	GET_SET_PROPERTY_METHOD_MACRO( int32, FirstSlice, firstSlice );
	GET_SET_PROPERTY_METHOD_MACRO( Coordinates, SecondPoint, secondPoint );
	GET_SET_PROPERTY_METHOD_MACRO( int32, SecondSlice, secondSlice );

	GET_SET_PROPERTY_METHOD_MACRO( Coordinates, InsidePoint, insidePoint );
	GET_SET_PROPERTY_METHOD_MACRO( int32, InsidePointSlice, insidePointSlice );
	GET_SET_PROPERTY_METHOD_MACRO( Coordinates, OutsidePoint, outsidePoint );
	GET_SET_PROPERTY_METHOD_MACRO( int32, OutsidePointSlice, outsidePointSlice );

	GET_SET_PROPERTY_METHOD_MACRO( int32, Precision, precision );
protected:

	typedef	NormalDistribution					Distribution;

	typedef M4D::Imaging::Algorithms::SegmentationEnergy< 
			CurveType,
			//M4D::Imaging::Algorithms::RegionImageEnergy< CurveType, RegionType, Distribution >,
			M4D::Imaging::Algorithms::UnifiedImageEnergy< CurveType, RegionType, RegionType, Distribution >,
			M4D::Imaging::Algorithms::InternalCurveEnergy< CurveType >,
			M4D::Imaging::Algorithms::DummyEnergy3 >			EnergyModel;

	typedef M4D::Imaging::Algorithms::EnergicSnake< CurveType, EnergyModel >	SnakeAlgorithm;

	void
	ComputeStatistics( Vector<int32, 3> p, float32 &E, float32 &var );

	CurveType
	CreateSquareControlPoints( float32 radius );

	const InputImageType&
	GetInputImage( uint32 idx )const;

	void
	ReleaseInputImage( uint32 idx )const;

	OutputDatasetType&
	GetOutputGDataset()const;

	void
	ReleaseOutputGDataset()const;

	bool
	ExecutionThreadMethod( AbstractPipeFilter::UPDATE_TYPE utype );
	
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
	ProcessSlice( 
			//const RegionType &region, 
			CurveType &initialization, 
			typename OutputDatasetType::ObjectsInSlice &slice 
			);

	const InputImageType	*in[ InCount ];
	OutputDatasetType	*out;
	int32 _minSlice;
	int32 _maxSlice;
	ReaderBBoxInterface::Ptr readerBBox[ InCount ];

	float32 _inEstimatedValue;
	float32	_inVariation;

	float32 _outEstimatedValue;
	float32	_outVariation;

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "SnakeSegmentationFilter.tcc"

#endif //SNAKE_SEGMENTATION_FILTER_H
