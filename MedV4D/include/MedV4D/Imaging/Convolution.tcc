/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file Convolution.tcc
 * @{
 **/

#ifndef CONVOLUTION_H
#error File Convolution.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging {

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

template< typename ElementType, typename  MatrixElement, size_t Dim >
inline typename TypeTraits< ElementType >::SuperiorFloatType
ApplyConvolutionMask (
        ElementType 					*center,
        const Vector< int32, Dim >			strides,
        const ConvolutionMask< Dim, MatrixElement > 	&mask,
        const MatrixElement				&multiplication
)
{
        typename TypeTraits< ElementType >::SuperiorFloatType result = TypeTraits< ElementType >::Zero;
        ElementType *pointer = center;
        for ( unsigned d=0; d < Dim; ++d ) {
                pointer -= strides[d] * mask.center[d];
        }

        uint32 coord[ Dim ] = { 0 };
        for ( unsigned i=0; i<mask.length; ++i ) {
                result += mask.mask[i] * ( *pointer );

                for ( unsigned d=0; d < Dim; ++d ) {
                        if ( coord[d] == mask.size[d]-1 ) {
                                coord[d] = 0;
                                pointer -= ( mask.size[d]-1 ) *strides[d];
                        } else {
                                ++coord[d];
                                pointer += strides[d];
                                break;
                        }
                }
        }
        return multiplication * result;
}

template< typename ElementType, typename  MatrixElement, size_t Dim >
inline typename TypeTraits< ElementType >::SuperiorFloatType
ApplyConvolutionMaskMirrorBorder (
        ElementType 					*center,
        const Vector< int32, Dim >			strides,
        const uint32 					firstBorder[Dim],
        const uint32 					secondBorder[Dim],
        const ConvolutionMask< Dim, MatrixElement > 	&mask,
        const MatrixElement				&multiplication
)
{
        typename TypeTraits< ElementType >::SuperiorFloatType result = TypeTraits< ElementType >::Zero;

        uint32 coord[ Dim ] = { 0 };
        for ( unsigned i=0; i<mask.length; ++i ) {
                result += mask.mask[i] * ( *MirrorBorderAccess< ElementType, Dim > ( coord, mask.center, center, strides, firstBorder, secondBorder ) );

                for ( unsigned d=0; d < Dim; ++d ) {
                        if ( coord[d] == mask.size[d]-1 ) {
                                coord[d] = 0;
                                //pointer -= (mask.size[d]-1)*strides[d];
                        } else {
                                ++coord[d];
                                //pointer += strides[d];
                                break;
                        }
                }
        }
        return multiplication * result;
}

template< typename ElementType, typename  MatrixElement >
void
Compute2DConvolution (
        const ImageRegion< ElementType, 2 > 		&inRegion,
        ImageRegion< ElementType, 2 > 			&outRegion,
        const ConvolutionMask< 2, MatrixElement > 	&mask,
        const ElementType				addition,
        const MatrixElement				multiplication
)
{
        uint32 width = mask.size[0];
        uint32 height = mask.size[1];
        uint32 hwidth = mask.center[0];
        uint32 hheight = mask.center[1];
        //TODO check

        uint32 firstBorder[2];
        uint32 secondBorder[2];
        Vector< int32, 2 > coords;
        for ( coords[1] = 0; static_cast<uint32> ( coords[1] ) < hheight; ++coords[1] ) {
                firstBorder[1] = hheight - coords[1];
                secondBorder[1] = mask.size[1];
                for ( coords[0] = 0; static_cast<uint32> ( coords[0] ) < inRegion.GetSize ( 0 ); ++coords[0] ) {
                        firstBorder[0] = max ( static_cast<int32> ( hwidth - coords[0] ), 0 );
                        secondBorder[0] = min ( inRegion.GetSize ( 0 )-coords[0]+hwidth, mask.size[0] );
                        outRegion.GetElementRel ( coords ) = addition +
                                                             static_cast< ElementType > ( ApplyConvolutionMaskMirrorBorder (
                                                                                                  inRegion.GetPointer ( coords ),
                                                                                                  inRegion.GetStride(),
                                                                                                  firstBorder,
                                                                                                  secondBorder,
                                                                                                  mask,
                                                                                                  multiplication
                                                                                          ) );
                }
        }
        for ( coords[1] = inRegion.GetSize ( 1 )-height+hheight; static_cast<uint32> ( coords[1] ) < inRegion.GetSize ( 1 ); ++coords[1] ) {
                firstBorder[1] = 0;
                secondBorder[1] = inRegion.GetSize ( 1 ) - coords[1]+hheight;
                for ( coords[0] = 0; static_cast<uint32> ( coords[0] ) < inRegion.GetSize ( 0 ); ++coords[0] ) {
                        firstBorder[0] = max ( static_cast<int32> ( hwidth - coords[0] ), 0 );
                        secondBorder[0] = min ( inRegion.GetSize ( 0 )-coords[0]+hwidth, mask.size[0] );
                        outRegion.GetElementRel ( coords ) = addition +
                                                             static_cast< ElementType > ( ApplyConvolutionMaskMirrorBorder (
                                                                                                  inRegion.GetPointer ( coords ),
                                                                                                  inRegion.GetStride(),
                                                                                                  firstBorder,
                                                                                                  secondBorder,
                                                                                                  mask,
                                                                                                  multiplication
                                                                                          ) );
                }
        }
        for ( coords[1] = hheight; static_cast<uint32> ( coords[1] ) < ( inRegion.GetSize ( 1 ) - height + hheight ); ++coords[1] ) {
                firstBorder[1] = 0;
                secondBorder[1] = mask.size[1];
                for ( coords[0] = 0; static_cast<uint32> ( coords[0] ) < hwidth; ++coords[0] ) {
                        firstBorder[0] = hwidth - coords[0];
                        secondBorder[0] = mask.size[0];
                        outRegion.GetElementRel ( coords ) = addition +
                                                             static_cast< ElementType > ( ApplyConvolutionMaskMirrorBorder (
                                                                                                  inRegion.GetPointer ( coords ),
                                                                                                  inRegion.GetStride(),
                                                                                                  firstBorder,
                                                                                                  secondBorder,
                                                                                                  mask,
                                                                                                  multiplication
                                                                                          ) );

                }
                for ( coords[0] = inRegion.GetSize ( 0 )-width+hwidth; static_cast<uint32> ( coords[0] ) < inRegion.GetSize ( 0 ); ++coords[0] ) {
                        firstBorder[0] = 0;
                        secondBorder[0] = inRegion.GetSize ( 0 )-coords[0]+hwidth;
                        outRegion.GetElementRel ( coords ) = addition +
                                                             static_cast< ElementType > ( ApplyConvolutionMaskMirrorBorder (
                                                                                                  inRegion.GetPointer ( coords ),
                                                                                                  inRegion.GetStride(),
                                                                                                  firstBorder,
                                                                                                  secondBorder,
                                                                                                  mask,
                                                                                                  multiplication
                                                                                          ) );

                }
        }

        for ( coords[1] = hheight; static_cast<uint32> ( coords[1] ) < ( inRegion.GetSize ( 1 ) - height + hheight ); ++coords[1] ) {
                for ( coords[0] = hwidth; static_cast<uint32> ( coords[0] ) < ( inRegion.GetSize ( 0 ) - width + hwidth ); ++coords[0] ) {
                        outRegion.GetElementRel ( coords ) = addition +
                                                             static_cast< ElementType > ( ApplyConvolutionMask (
                                                                                                  inRegion.GetPointer ( coords ),
                                                                                                  inRegion.GetStride(),
                                                                                                  mask,
                                                                                                  multiplication
                                                                                          ) );

                }
        }
}
template< typename ElementType, typename OutElementType, typename  MatrixElement, typename PostProcessor >
void
Compute2DConvolutionPostProcess (
        const ImageRegion< ElementType, 2 > 		&inRegion,
        ImageRegion< OutElementType, 2 > 		&outRegion,
        const ConvolutionMask< 2, MatrixElement > 	&mask,
        const ElementType				addition,
        const MatrixElement				multiplication,
        PostProcessor					postprocessor
)
{
        uint32 width = mask.size[0];
        uint32 height = mask.size[1];
        uint32 hwidth = mask.center[0];
        uint32 hheight = mask.center[1];
        //TODO check

        uint32 firstBorder[2];
        uint32 secondBorder[2];
        Vector< int32, 2 > coords;
        typename TypeTraits< ElementType >::SuperiorFloatType result = TypeTraits< ElementType >::Zero;
        for ( coords[1] = 0; static_cast<uint32> ( coords[1] ) < hheight; ++coords[1] ) {
                firstBorder[1] = hheight - coords[1];
                secondBorder[1] = mask.size[1];
                for ( coords[0] = 0; static_cast<uint32> ( coords[0] ) < inRegion.GetSize ( 0 ); ++coords[0] ) {
                        firstBorder[0] = max ( static_cast<int32> ( hwidth - coords[0] ), 0 );
                        secondBorder[0] = min ( inRegion.GetSize ( 0 )-coords[0]+hwidth, mask.size[0] );
                        result = addition + ApplyConvolutionMaskMirrorBorder (
                                         inRegion.GetPointer ( coords ),
                                         inRegion.GetStride(),
                                         firstBorder,
                                         secondBorder,
                                         mask,
                                         multiplication
                                 );
                        postprocessor ( result, outRegion.GetElementRel ( coords ) );
                }
        }
        for ( coords[1] = inRegion.GetSize ( 1 )-height+hheight; static_cast<uint32> ( coords[1] ) < inRegion.GetSize ( 1 ); ++coords[1] ) {
                firstBorder[1] = 0;
                secondBorder[1] = inRegion.GetSize ( 1 ) - coords[1]+hheight;
                for ( coords[0] = 0; static_cast<uint32> ( coords[0] ) < inRegion.GetSize ( 0 ); ++coords[0] ) {
                        firstBorder[0] = max ( static_cast<int32> ( hwidth - coords[0] ), 0 );
                        secondBorder[0] = min ( inRegion.GetSize ( 0 )-coords[0]+hwidth, mask.size[0] );
                        result = addition + ApplyConvolutionMaskMirrorBorder (
                                         inRegion.GetPointer ( coords ),
                                         inRegion.GetStride(),
                                         firstBorder,
                                         secondBorder,
                                         mask,
                                         multiplication
                                 );
                        postprocessor ( result, outRegion.GetElementRel ( coords ) );

                }
        }
        for ( coords[1] = hheight; static_cast<uint32> ( coords[1] ) < ( inRegion.GetSize ( 1 ) - height + hheight ); ++coords[1] ) {
                firstBorder[1] = 0;
                secondBorder[1] = mask.size[1];
                for ( coords[0] = 0; static_cast<uint32> ( coords[0] ) < hwidth; ++coords[0] ) {
                        firstBorder[0] = hwidth - coords[0];
                        secondBorder[0] = mask.size[0];
                        result = addition + ApplyConvolutionMaskMirrorBorder (
                                         inRegion.GetPointer ( coords ),
                                         inRegion.GetStride(),
                                         firstBorder,
                                         secondBorder,
                                         mask,
                                         multiplication
                                 );
                        postprocessor ( result, outRegion.GetElementRel ( coords ) );
                }
                for ( coords[0] = inRegion.GetSize ( 0 )-width+hwidth; static_cast<uint32> ( coords[0] ) < inRegion.GetSize ( 0 ); ++coords[0] ) {
                        firstBorder[0] = 0;
                        secondBorder[0] = inRegion.GetSize ( 0 )-coords[0]+hwidth;
                        result = addition + ApplyConvolutionMaskMirrorBorder (
                                         inRegion.GetPointer ( coords ),
                                         inRegion.GetStride(),
                                         firstBorder,
                                         secondBorder,
                                         mask,
                                         multiplication
                                 );
                        postprocessor ( result, outRegion.GetElementRel ( coords ) );
                }
        }

        for ( coords[1] = hheight; static_cast<uint32> ( coords[1] ) < ( inRegion.GetSize ( 1 ) - height + hheight ); ++coords[1] ) {
                for ( coords[0] = hwidth; static_cast<uint32> ( coords[0] ) < ( inRegion.GetSize ( 0 ) - width + hwidth ); ++coords[0] ) {
                        result = addition + ApplyConvolutionMask (
                                         inRegion.GetPointer ( coords ),
                                         inRegion.GetStride(),
                                         mask,
                                         multiplication
                                 );
                        postprocessor ( result, outRegion.GetElementRel ( coords ) );

                }
        }
}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*CONVOLUTION_H*/

/** @} */
