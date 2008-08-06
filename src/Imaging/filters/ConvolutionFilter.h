#ifndef _THRESHOLDING_FILTER_H
#define _THRESHOLDING_FILTER_H

#include "Common.h"


namespace M4D
{

namespace Imaging
{

template< typename MatrixElement >
struct ConvolutionFilter2DOptions
{
	MatrixElement *matrix; //length = width*height

	size_t	width;
	size_t	height;
};

template< typename InputImageType, typename MatrixElement >
class ConvolutionFilter2D;

template< typename InputElementType, typename MatrixElement >
class ConvolutionFilter2D< Image< InputElementType, 2 >
{
	//TODO
};

template< typename InputElementType, typename MatrixElement >
class ConvolutionFilter2D< Image< InputElementType, 3 >, MatrixElement > 
	: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > >
{
public:
	typedef ThresholdingFilterOptions< MatrixElement >	Settings;
	ConvolutionFilter2D();
protected:
	typedef typename  Imaging::IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > > PredecessorType;

	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< InputElementType, 3 >		&out,
			size_t			x1,	
			size_t			y1,	
			size_t			x2,	
			size_t			y2,	
			size_t			slice
		    );

};

//******************************************************************************
//******************************************************************************

template< typename InputElementType >
struct ConvolutionFilter3DOptions
{
	InputElementType	bottom;	
	InputElementType	top;
	
	InputElementType	outValue;
};

template< typename InputImageType >
class ConvolutionFilter3D;

template< typename InputElementType >
class ConvolutionFilter3D< Image< InputElementType, 2 >
{
	//TODO
};

template< typename InputElementType >
class ConvolutionFilter3D< Image< InputElementType, 3 > 
	: public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > >
{
public:
	typedef ConvolutionFilter3D< InputElementType >	Settings;
	ConvolutionFilter3D();
protected:
	typedef typename  Imaging::IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< InputElementType, 3 > > PredecessorType;

	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< InputElementType, 3 >		&out,
			size_t			x1,	
			size_t			y1,	
			size_t			x2,	
			size_t			y2,	
			size_t			slice
		    );

};


} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/ConvolutionFilter.tcc"

#endif /*_THRESHOLDING_FILTER_H*/
