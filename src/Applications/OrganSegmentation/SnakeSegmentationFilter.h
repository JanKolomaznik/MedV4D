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

	typedef SlicedGeometry< float32, Imaging::Geometry::BSpline > OutputDataset;

	typedef typename ImageTraits< InputImageType >::InputPort
		InputPortType;
	typedef OutputPortTyped< OutputDataset > OutputPortType;

	typedef Image< ElementType, 3 >		InputImageType;

	typedef ImageRegion< ElementType, 2 >	RegionType;

	static const unsigned InCount = 2;

	struct Properties : public PredecessorType::Properties
	{

	};

	~SnakeSegmentationFilter() {}

	SnakeSegmentationFilter( Properties * prop );

protected:
	const InputImageType&
	GetInputImage( uint32 idx )const;

	void
	ReleaseInputImage( uint32 idx )const;

	OutputDataset&
	GetOutputGDataset()const;

	void
	ReleaseOutputGDataset()const;

	void
	ExecutionThreadMethod();
	
	void
	PrepareOutputDatasets();
	
	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );
	
	void
	MarkChanges( AbstractPipeFilter::UPDATE_TYPE &utype );

	void
	AfterComputation( bool successful );

	void
	ProcessSlice( const RegionType &region, CurveType &initialization, typename OutputDataset::ObjectsInSlice &slice );

	const InputImageType	*in[ InCount ];
	OutputDataset		*out;
	int32 _minSlice;
	int32 _maxSlice;
	ReaderBBoxInterface::Ptr readerBBox[ InCount ];

};

//include implementation
#include "Imaging/filters/ConvolutionFilter.tcc"

#endif //SNAKE_SEGMENTATION_FILTER_H
