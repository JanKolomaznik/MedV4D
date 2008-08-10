#ifndef _THRESHOLDING_FILTER_H
#define _THRESHOLDING_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImageSliceFilter.h"
#include <boost/shared_array.hpp>

namespace M4D
{

namespace Imaging
{

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
	struct Properties : public PredecessorType::Properties
	{
		typedef boost::shared_array<MatrixElement> MatrixPtr;
		Properties();

		MatrixPtr matrix; //length = width*height

		size_t	width;
		size_t	height;
	};

	ConvolutionFilter2D( Properties * prop );
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
private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

//******************************************************************************
//******************************************************************************



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
		MatrixPtr	matrix; //length = width*height*depth

		size_t	width;
		size_t	height;
		size_t	depth;
	};

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
private:
	GET_PROPERTIES_DEFINITION_MACRO;

};


} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/filters/ConvolutionFilter.tcc"

#endif /*_THRESHOLDING_FILTER_H*/
