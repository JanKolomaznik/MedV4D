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

	ConvolutionMask( MatrixElement *m, uint32 s[Dim] )
		: length( 1 ), mask( m )
		{ 	
			length = 1;
			for( unsigned i = 0; i < Dim; ++i ) {
				size[i] = s[i];
				center[i] = s[i]/2;
				length *= s[i];
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

	uint32		size[ Dim ];
	uint32		center[ Dim ];
	uint32		length;
	MatrixElement	*mask;
};

template< typename ElementType, typename  MatrixElement, unsigned Dim >
inline ElementType
ApplyConvolutionMask( 
		ElementType 	*center, 
		const int32 	strides[Dim], 
		const ConvolutionMask< Dim, MatrixElement > &mask 
		)
{
	ElementType result = 0;
	ElementType *pointer = center;
	for( unsigned d=0; d < Dim; ++d )
	{
		pointer -= strides[d] * mask.center[d];
	}

	uint32 coord[ Dim ] = { 0 };
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

template< typename ElementType, typename  MatrixElement >
void
Compute2DConvolution(
		const ImageRegion< ElementType, 2 > 		&inRegion,
		ImageRegion< ElementType, 2 > 			&outRegion,
		const ConvolutionMask< 2, MatrixElement > 	&mask
	)
{
	uint32 width = mask.size[0];
	uint32 height = mask.size[1];
	uint32 hwidth = mask.center[0];
	uint32 hheight = mask.center[1];
	//TODO check
	
	Coordinates< int32, 2 > coords;
	for( coords[1] = hheight; static_cast<uint32>(coords[1]) < ( inRegion.GetSize(1) - height + hheight ); ++coords[1] ) {
		for( coords[0] = hwidth; static_cast<uint32>(coords[0]) < ( inRegion.GetSize(0) - width + hwidth ); ++coords[0] ) {
			outRegion.GetElement( coords ) = 
				ApplyConvolutionMask( 
						inRegion.GetPointer( coords ), 
						inRegion.GetStride(), 
						mask 
						);
		}
	}
}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*CONVOLUTION_H*/

/** @} */
