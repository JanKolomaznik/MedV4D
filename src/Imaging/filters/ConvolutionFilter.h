/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConvolutionFilter.h 
 * @{ 
 **/

#ifndef _CONVOLUTION_FILTER_H
#define _CONVOLUTION_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImage2DFilter.h"
#include <boost/shared_array.hpp>
#include "Imaging/Convolution.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{


template< typename ImageType, typename MatrixElement = float32 >
class ConvolutionFilter2D 
	: public AbstractImage2DFilter< ImageType, ImageType >
{
public:	
	static const unsigned Dimension = ImageTraits< ImageType >::Dimension;
	typedef typename ImageTraits< ImageType >::ElementType ElementType;
	typedef AbstractImage2DFilter< ImageType, ImageType > PredecessorType;
	typedef typename ConvolutionMask<2,MatrixElement>::Ptr	MaskPtr;
	typedef ImageRegion< ElementType, 2 >		Region;

	struct Properties : public PredecessorType::Properties
	{
		Properties(){}

		MaskPtr matrix; 
	};

	ConvolutionFilter2D( Properties * prop );
	ConvolutionFilter2D();
	
	GET_SET_PROPERTY_METHOD_MACRO( MaskPtr, ConvolutionMask, matrix );
protected:
	bool
	Process2D(
			const Region	&inRegion,
			Region 		&outRegion
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

};*/


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/ConvolutionFilter.tcc"

#endif /*_CONVOLUTION_FILTER_H*/

/** @} */

