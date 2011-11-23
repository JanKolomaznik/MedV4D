#ifndef _SPHERE_SELECTION_H
#define _SPHERE_SELECTION_H

#include "common/Common.h"
#include "Imaging/AImageFilterWholeAtOnce.h"

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Milan Lepik
 * @file SphereSelection.h 
 * @{ 
 **/

namespace Imaging
{

template< typename ImageType >
class SphereSelection;

template< typename ElementType >
class SphereSelection< Image< ElementType, 3 > >
	: public AImageFilterWholeAtOnce< Image< ElementType, 3 >, Image< ElementType, 3 > >
{
public:	
	typedef Image< ElementType, 3 >		InputImageType;
	typedef Image< ElementType, 3 >		OutputImageType;
	typedef AImageFilterWholeAtOnce< InputImageType, OutputImageType > PredecessorType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): radius( 60 ), iCenter( 256 ), jCenter( 256 ), kCenter( 100 ) {}

		uint16	radius;
		uint16	iCenter;
		uint16	jCenter;
		uint16	kCenter;
	};

	SphereSelection( Properties * prop );
	SphereSelection();

	GET_SET_PROPERTY_METHOD_MACRO( uint16, Radius, radius );
	GET_SET_PROPERTY_METHOD_MACRO( uint16, ColumnCenter, iCenter );
	GET_SET_PROPERTY_METHOD_MACRO( uint16, RowCenter, jCenter );
	GET_SET_PROPERTY_METHOD_MACRO( uint16, SliceCenter, kCenter );
protected:

	void
	BeforeComputation( APipeFilter::UPDATE_TYPE &utype );

	bool
	ProcessImage(
			const InputImageType 	&in,
			OutputImageType		&out
		    );

	template< typename OperatorType >
	bool
	ProcessImageHelper(
			const InputImageType 	&in,
			OutputImageType		&out
		    );

	void
	PrepareOutputDatasets();

	template< typename OperatorType >
	bool
	DoProjection(
			ElementType	*inPointer,
			ElementType	*outPointer,
			int32		ixStride,
			int32		iyStride,
			int32		izStride,
			int32		oxStride,
			int32		oyStride,
			int32		ozStride,
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
/** @} */

} /*namespace M4D*/


//include implementation
#include "Imaging/filters/SphereSelection.tcc"

#endif /*_SPHERE_SELECTION_H*/


