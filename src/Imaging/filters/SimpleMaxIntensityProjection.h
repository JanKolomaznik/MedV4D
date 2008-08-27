/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file SimpleMaxIntensityProjection.h 
 * @{ 
 **/

#ifndef _SIMPLE_MAX_INTENSITY_PROJECTION_H
#define _SIMPLE_MAX_INTENSITY_PROJECTION_H

#include "Common.h"
#include "Imaging/AbstractImageFilterWholeAtOnce.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

enum CartesianPlanes{
	XY_PLANE,
	XZ_PLANE,
	YZ_PLANE
};	

template< typename ImageType >
class SimpleMaxIntensityProjection;

template< typename ElementType >
class SimpleMaxIntensityProjection< Image< ElementType, 3 > >
	: public AbstractImageFilterWholeAtOnce< Image< ElementType, 3 >, Image< ElementType, 2 > >
{
public:	
	typedef Image< ElementType, 3 >		InputImageType;
	typedef Image< ElementType, 2 >		OutputImageType;
	typedef AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType > PredecessorType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): plane( XY_PLANE ) {}

		CartesianPlanes	plane;
	};

	SimpleMaxIntensityProjection( Properties * prop );
	SimpleMaxIntensityProjection();

	GET_SET_PROPERTY_METHOD_MACRO( CartesianPlanes, Plane, plane );
protected:

	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	bool
	ProcessImage(
			const InputImageType 	&in,
			OutputImageType		&out
		    );

	void
	PrepareOutputDatasets();

	bool
	DoProjection(
			ElementType	*inPointer,
			ElementType	*outPointer,
			int32		ixStride,
			int32		iyStride,
			int32		izStride,
			int32		oxStride,
			int32		oyStride,
			uint32		width,
			uint32		height,
			uint32		depth
		    );


private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

//******************************************************************************
//******************************************************************************


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/SimpleMaxIntensityProjection.tcc"

#endif /*_SIMPLE_MAX_INTENSITY_PROJECTION_H*/

/** @} */

