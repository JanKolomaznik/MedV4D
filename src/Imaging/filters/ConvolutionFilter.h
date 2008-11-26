/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file ConvolutionFilter.h 
 * @{ 
 **/

#ifndef _CONVOLUTION_FILTER_H
#define _CONVOLUTION_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImageSliceFilter.h"
#include <boost/shared_array.hpp>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{

template< unsigned Dim, typename MatrixElement = float32 >
struct ConvolutionMask
{
	typedef boost::shared_ptr<ConvolutionMask<MatrixElement, Dim> > Ptr;

	ConvolutionMask( MatrixElement *m, int32 s[Dim] )
		: mask( m )
		{ 	length = 1;
			for( unsigned i = 0; i < Dim; ++i ) {
				length *= size[i];
				size[i] = s[i];
				center[i] = s[i]/2;
			}
		}

	~ConvolutionMask()
		{ delete [] mask; }

	MatrixElement &
	GetElement( int32 pos[Dim] )
		{
			int32 idx = 0;
			for( unsigned i=0; i<Dim; ++i ) {
				idx += pos[i]*size[i];
			}
			return *(mask+idx);
		}

	int32		size[ Dim ];
	int32		center[ Dim ];
	int32		length;
	MatrixElement	*mask;
};

template< typename InputImageType, typename MatrixElement = float32 >
class ConvolutionFilter2D;

template< typename InputElementType, typename MatrixElement  = float32 >
class ConvolutionFilter2D< Image< InputElementType, 2 >, MatrixElement >
{
	//TODO
};

template< typename InputElementType, typename MatrixElement  = float32 >
class ConvolutionFilter2D< Image< InputElementType, 3 >, MatrixElement > 
	: public AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< InputElementType, 3 > >
{
public:	
	static const unsigned Dimension = 3;
	typedef AbstractImageSliceFilterIExtents< Image< InputElementType, 3 >, Image< InputElementType, 3 > > PredecessorType;
	typedef ConvolutionMask<MatrixElement, 2>	Mask;

	struct Properties : public PredecessorType::Properties
	{
		Properties(){}

		Mask::Ptr matrix; //length = width*height
	};

	ConvolutionFilter2D( Properties * prop );
	ConvolutionFilter2D();
	
	GET_SET_PROPERTY_METHOD_MACRO( Mask::Ptr, ConvolutionMask, matrix );
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


template< typename ElementType, typenamepe  MatrixElement, unsigned Dim >
inline ElementType
ApplyConvolutionMask( 
		ElementType 	*center, 
		int 		strides[Dim], 
		const ConvolutionMask< MatrixElement, Dim > &mask 
		)
{
	ElementType result = 0;
	ElementType *pointer = center;
	for( unsigned d=0; d < Dim; ++d )
	{
		pointer -= strides[d] * mask.center[d];
	}

	int32 coord[ Dim ] = { 0 }
	for( unsigned i=0; i<mask.length; ++i ) {
		result += mask.mask[i] * (*pointer);

		for( unsigned d=0; d < Dim; ++d )
		{
			if( coord[d] == mask.size[d]-1 ) {
				coord[d] = 0;
			} else {
				++coord[d];
				break;
			}
		}
	}
	return result;
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

//include implementation
#include "Imaging/filters/ConvolutionFilter.tcc"

#endif /*_CONVOLUTION_FILTER_H*/

/** @} */

