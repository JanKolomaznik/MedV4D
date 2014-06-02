/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file Convolution.h
 * @{
 **/

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "MedV4D/Imaging/ImageRegion.h"
#include <memory>

#include "MedV4D/Imaging/FilterComputation.h"

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging {

template< size_t Dim, typename MatrixElement = float32 >
struct ConvolutionMask {
        typedef std::shared_ptr<ConvolutionMask<Dim,MatrixElement> > Ptr;

        ConvolutionMask ( MatrixElement *m, Vector< uint32, Dim > s )
                        : length ( 1 ), mask ( m ) {
                length = 1;
                for ( unsigned i = 0; i < Dim; ++i ) {
                        size[i] = s[i];
                        center[i] = s[i]/2;
                        strides[i] = length;
                        length *= s[i];
                }
        }

        ~ConvolutionMask() {
                delete [] mask;
        }

        MatrixElement &
        Get ( const Vector< uint32, Dim > &coord ) {
                return mask[coord * strides];
        }
        MatrixElement
        Get ( const Vector< uint32, Dim > &coord ) const {
                return mask[coord * strides];
        }
        MatrixElement &
        Get ( uint32 i ) {
                return mask[i];
        }
        MatrixElement
        Get ( uint32 i ) const {
                return mask[i];
        }


        Vector< uint32, Dim >	size;
        Vector< uint32, Dim >	center;
        Vector< uint32, Dim >	strides;
        uint32		length;
        MatrixElement	*mask;
};

template< typename ElType >
class ConvolutionFilterFtor: public FilterFunctorBase< ElType >
{
public:
        ConvolutionFilterFtor ( const ConvolutionMask<2, float32> &mask ) : _mask ( mask ), _size ( mask.size ), _center ( mask.center ) {
                _leftCorner = -1 * Vector< int32, 2 > ( _center[0], _center[1] );
                _rightCorner = Vector< int32, 2 > ( _size[0] - _center[0] - 1, _size[1] - _center[1] - 1 );
        }

        ConvolutionFilterFtor ( const ConvolutionFilterFtor &ftor ) : _mask ( ftor._mask ), _size ( ftor._size ) {
                _leftCorner = -1 * Vector< int32, 2 > ( _center[0], _center[1] );
                _rightCorner = Vector< int32, 2 > ( _size[0] - _center[0] - 1, _size[1] - _center[1] - 1 );
        }

        ~ConvolutionFilterFtor() {}

        template< typename Accessor >
        ElType
        Apply ( const Vector< int32, 2 > &pos, Accessor &accessor ) {
                Vector< int32, 2 > idx;
                unsigned i = 0;
                typedef typename TypeTraits< ElType >::SuperiorFloatType SType;
                SType result = 0;
                const Vector< int32, 2 > lborder ( pos + _leftCorner );
                const Vector< int32, 2 > rborder ( pos + _rightCorner );

                for ( idx[1] = lborder[1]; idx[1] <= rborder[1]; ++idx[1] ) {
                        for ( idx[0] = lborder[0]; idx[0] <= rborder[0]; ++idx[0] ) {
                                result += _mask.mask[i] * static_cast< SType > ( accessor ( idx ) );
                                ++i;
                        }
                }

                return static_cast<ElType> ( result );
        }
        Vector< int32, 2 >
        GetLeftCorner() const {
                return _leftCorner;
        }

        Vector< int32, 2 >
        GetRightCorner() const {
                return _rightCorner;
        }
protected:
        const ConvolutionMask<2, float32> &_mask;
        Vector< uint32, 2 > _size;
        Vector< uint32, 2 > _center;

        Vector< int32, 2 > _leftCorner;
        Vector< int32, 2 > _rightCorner;
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

template< typename ElementType, size_t Dim >
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
//#include "MedV4D/Imaging/Convolution.tcc"

#endif /*CONVOLUTION_H*/


/** @} */

