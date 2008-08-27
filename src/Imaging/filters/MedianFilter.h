#ifndef _MEDIAN_FILTER_H
#define _MEDIAN_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImage2DFilter.h"
#include <boost/shared_array.hpp>
#include <map>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< typename InputImageType >
class MedianFilter2D
	: public AbstractImage2DFilter< InputImageType, InputImageType >
{
public:	
	typedef AbstractImage2DFilter< InputImageType, InputImageType > PredecessorType;
	typedef typename ImageTraits< InputImageType >::ElementType InputElementType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): radius( 0 ) {}

		uint32	radius;
	};

	MedianFilter2D( Properties * prop );
	MedianFilter2D();

	GET_SET_PROPERTY_METHOD_MACRO( uint32, Radius, radius );
protected:

	void
	BeforeComputation( AbstractPipeFilter::UPDATE_TYPE &utype );

	bool
	Process2D(
			InputElementType	*inPointer,
			int32			i_xStride,
			int32			i_yStride,
			InputElementType	*outPointer,
			int32			o_xStride,
			int32			o_yStride,
			uint32			width,
			uint32			height
		 );
private:
	typedef typename std::map< InputElementType, int >	Histogram;

	GET_PROPERTIES_DEFINITION_MACRO;

	inline InputElementType
	GetElementInOrder(
		Histogram				&histogram,
		uint32					order
	      );

};

//******************************************************************************
//******************************************************************************

/*

template< typename InputImageType, typename MatrixElement >
class ConvolutionFilter3D;

template< typename InputElementType, typename MatrixElement >
class ConvolutionFilter3D< Image< InputElementType, 2 >, MatrixElement >
{
	//TODO
};

template< typename InputElementType, typename MatrixElement >
class ConvolutionFilter3D< Image< InputElementType, 3 >, MatrixElement > 
	: public AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< InputElementType, 3 > >
{
public:
	struct Properties : public PredecessorType::Properties
	{
		typedef boost::shared_array<MatrixElement> MatrixPtr;
		Properties();
		
		void
		CheckProperties() 
			{ _sliceComputationNeighbourCount = depth / 2; }

		MatrixPtr	matrix; //length = width*height*depth

		uint32	width;
		uint32	height;
		uint32	depth;
	
	};

	ConvolutionFilter3D();
protected:
	typedef typename  Imaging::AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< InputElementType, 3 > > PredecessorType;

	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< InputElementType, 3 >		&out,
			int32			x1,	
			int32			y1,	
			int32			x2,	
			int32			y2,	
			int32			slice
		    );
private:
	GET_PROPERTIES_DEFINITION_MACRO;

};
*/

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/MedianFilter.tcc"

#endif /*_MEDIAN_FILTER_H*/
