#ifndef SNAKE_SEGMENTATION_FILTER_H
#define SNAKE_SEGMENTATION_FILTER_H

#include "Imaging.h"

namespace M4D
{

namespace Imaging
{

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
	};

	~SnakeSegmentationFilter() {}

	SnakeSegmentationFilter();
	SnakeSegmentationFilter( Properties * prop );

	GET_SET_PROPERTY_METHOD_MACRO( Coordinates, FirstPoint, firstPoint );
	GET_SET_PROPERTY_METHOD_MACRO( int32, FirstSlice, firstSlice );
	GET_SET_PROPERTY_METHOD_MACRO( Coordinates, SecondPoint, secondPoint );
	GET_SET_PROPERTY_METHOD_MACRO( int32, SecondSlice, secondSlice );
protected:
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
			//const RegionType &region, 
			CurveType &initialization, 
			typename OutputDatasetType::ObjectsInSlice &slice 
			);

	const InputImageType	*in[ InCount ];
	OutputDatasetType	*out;
	int32 _minSlice;
	int32 _maxSlice;
	ReaderBBoxInterface::Ptr readerBBox[ InCount ];

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "SnakeSegmentationFilter.tcc"

#endif //SNAKE_SEGMENTATION_FILTER_H
