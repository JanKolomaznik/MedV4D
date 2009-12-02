#ifndef SLICE_SPLINE_FILL_H
#define SLICE_SPLINE_FILL_H

#include "Imaging/Imaging.h"
#include <cmath>

namespace M4D
{

namespace Imaging
{

template < typename CoordType >
class SliceSplineFill: public APipeFilter
{
public:
	typedef APipeFilter	PredecessorType;
	typedef	Imaging::Geometry::BSpline< CoordType, 2 >		CurveType;
	typedef SlicedGeometry< CurveType >				InputDatasetType;
	typedef Mask3D							OutputDatasetType;
	typedef InputPortTyped< InputDatasetType >			InputPortType;
	typedef OutputPortTyped< Mask3D > 				OutputPortType;
	//typedef Mask3D ImageRegion< uint8, 2 >					RegionType;
	typedef typename InputDatasetType::ObjectsInSlice		ObjectsInSlice;

	typedef Vector< CoordType, 2 >					Coordinates;
	typedef Vector< int32, 3 >					RasterPos;
	typedef	Vector< float32, 3 >					ElementExtentsType;


	static const uint8 InMaskVal = 255;
	static const uint8 OutMaskVal = 0;

	struct Properties : public PredecessorType::Properties
	{
		Properties() : 	elementExtents( 1.0f ) {}
		RasterPos minimum;
		RasterPos maximum;
		ElementExtentsType elementExtents;
	};

	~SliceSplineFill() {}

	SliceSplineFill();
	SliceSplineFill( Properties * prop );

	GET_SET_PROPERTY_METHOD_MACRO( RasterPos, Minimum, minimum );
	GET_SET_PROPERTY_METHOD_MACRO( RasterPos, Maximum, maximum );
	GET_SET_PROPERTY_METHOD_MACRO( ElementExtentsType, ElementExtents, elementExtents );
protected:
	bool
	ExecutionThreadMethod( APipeFilter::UPDATE_TYPE utype );
	
	void
	PrepareOutputDatasets();
	
	void
	BeforeComputation( APipeFilter::UPDATE_TYPE &utype );
	
	void
	MarkChanges( APipeFilter::UPDATE_TYPE utype );

	void
	AfterComputation( bool successful );

	void
	ProcessSlice( 
			ImageRegion< uint8, 2 >	slice,
			const ObjectsInSlice	&objects 
			);

	void
	ProcessBlankSlice( 
			ImageRegion< uint8, 2 >	slice
			);

	void
	ProcessSlice( 
			//const RegionType &region, 
			CurveType &initialization, 
			ObjectsInSlice &slice 
			);

	Mask3D			*out;
	const InputDatasetType	*in;

	ReaderBBoxInterface::Ptr	readerBBox;
	WriterBBoxInterface		*writerBBox;
};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "SliceSplineFill.tcc"

#endif //SLICE_SPLINE_FILL_H
