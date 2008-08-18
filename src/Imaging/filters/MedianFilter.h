#ifndef _MEDIAN_FILTER_H
#define _MEDIAN_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImageSliceFilter.h"
#include <boost/shared_array.hpp>

namespace M4D
{

namespace Imaging
{

template< typename InputImageType >
class MedianFilter2D;

template< typename InputElementType >
class MedianFilter2D< Image< InputElementType, 2 >
{
	//TODO
};

template< typename InputElementType, typename MatrixElement >
class MedianFilter2D< Image< InputElementType, 3 >, MatrixElement > 
	: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > >
{
public:	
	struct Properties : public PredecessorType::Properties
	{
		Properties(): PredecessorType::Properties( 0, 10 ), radius( 0 ) {}

		uint32	radius;
	};

	MedianFilter2D( Properties * prop );
	MedianFilter2D();
protected:
	typedef typename  Imaging::IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > > PredecessorType;

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
	: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > >
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
	typedef typename  Imaging::IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > > PredecessorType;

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

//include implementation
#include "Imaging/filters/MedianFilter.tcc"

#endif /*_MEDIAN_FILTER_H*/
