/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Convolution.h 
 * @{ 
 **/

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "Imaging/ImageRegion.h"
#include <boost/shared_ptr.hpp>

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
	typedef boost::shared_ptr<ConvolutionMask<Dim,MatrixElement> > Ptr;

	ConvolutionMask( MatrixElement *m, Vector< uint32, Dim > s )
		: length( 1 ), mask( m )
		{ 	
			length = 1;
			for( unsigned i = 0; i < Dim; ++i ) {
				size[i] = s[i];
				center[i] = s[i]/2;
				strides[i] = length;
				length *= s[i];
			}
		}

	~ConvolutionMask()
		{ delete [] mask; }

	MatrixElement &
	Get( const Vector< uint32, Dim > &coord )
		{
			return mask[coord * strides];
		}
	MatrixElement
	Get( const Vector< uint32, Dim > &coord )const
		{
			return mask[coord * strides];
		}

	Vector< uint32, Dim >	size;
	Vector< uint32, Dim >	center;
	Vector< uint32, Dim >	strides;
	uint32		length;
	MatrixElement	*mask;
};

/*
template< typename ElementType, typename  MatrixElement >
void
Compute2DConvolution(
		const ImageRegion< ElementType, 2 > 		&inRegion,
		ImageRegion< ElementType, 2 > 			&outRegion,
		const ConvolutionMask< 2, MatrixElement > 	&mask,
		const ElementType				addition,
		const MatrixElement				multiplication
	);
*/
/*
 * struct PostProcessor {
 * void
 * operator( const ElementType &, OutElementType & );
 * };
 */
/*
template< typename ElementType, typename OutElementType, typename  MatrixElement, typename PostProcessor >
void
Compute2DConvolutionPostProcess(
		const ImageRegion< ElementType, 2 > 		&inRegion,
		ImageRegion< OutElementType, 2 > 		&outRegion,
		const ConvolutionMask< 2, MatrixElement > 	&mask,
		const ElementType				addition,
		const MatrixElement				multiplication,
		PostProcessor					postprocessor
	);

template< typename ElementType, unsigned Dim >
ElementType *
MirrorBorderAccess( 
		const uint32 					coord[ Dim ],
		const uint32 					maskcenter[ Dim ],
		ElementType 					*center, 
		const Vector< int32, Dim >			strides, 
		const uint32 					firstBorder[Dim],
		const uint32 					secondBorder[Dim] 
		)
{
	ElementType *pointer = center;
	for( unsigned d=0; d < Dim; ++d )
	{
		if( coord[ d ] < firstBorder[d] ) {
			int32 diff = 2*firstBorder[d] - maskcenter[d] -coord[d] -1;
			pointer += strides[d] * diff;
		} else if( coord[ d ] >= secondBorder[d] ) {
			int32 diff = 2*secondBorder[d] - maskcenter[d] -coord[d];
			pointer += strides[d] * diff;
		} else {
			int32 diff = coord[d] - maskcenter[d];
			pointer += strides[d] * diff;
		}
	}
	return pointer;
}
*/


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */
//#include "Imaging/Convolution.tcc"

#endif /*CONVOLUTION_H*/


/** @} */

