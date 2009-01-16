/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file Convolution.h 
 * @{ 
 **/

#ifndef FILTER_COMPUTATION_H
#define FILTER_COMPUTATION_H

#include "Imaging/Image.h"
#include "Coordinates.h"
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
template< typename ElementType >
class ExampleFilterKernel
{
public:
	typedef double ResultType;

protected:
	ElementType a;
};

template< typename InType, typename OutType >
class ExamplePostProcessor
{
public:

protected:
	InType a;
	OutType b;
};

template< 
	typename ElementType, 
	typename OutElementType,
	unsigned Dim, 
	template< typename InElement > 			class FilterKernel, 
	template< typename Input, typename Output > 	class PostProcessor,
	typename< typename IteratorType >		class Accessor
	>
void
__ImageFilterComputationFarFromBorder(
		M4D::Imaging::ImageIterator< ElementType, Dim >	&inIterator,
		M4D::Imaging::ImageIterator< OutElementType, Dim > 	&outIterator,
		FilterKernel< ElementType >				&kernel,
		PostProcessor< typename FilterKernel< ElementType >::ResultType, OutElementType >	&postprocessor,
		Accessor< M4D::Imaging::ImageIterator< ElementType, Dim > >				&accessor
	)
{
	while( !inIterator.IsEnd ) {
		FilterKernel< ElementType >::ResultType tmpResult = kernel.Apply( accessor );
		*outIterator = postprocessor.Compute( tmpResult );

		++inIterator;
		++outIterator;
	}
}

template< 
	typename ElementType, 
	typename OutElementType,
	unsigned Dim, 
	template< typename InElement > 			class FilterKernel, 
	template< typename Input, typename Output > 	class PostProcessor 
	>
void
__ImageFilterComputationFarFromBorder(
		const M4D::Imaging::ImageRegion< ElementType, Dim >	&inRegion,
		M4D::Imaging::ImageRegion< OutElementType, Dim > 	&outRegion,
		FilterKernel< ElementType >			&kernel,
		PostProcessor< typename FilterKernel< ElementType >::ResultType, OutElementType >	&postprocessor
	)
{
	M4D::Imaging::ImageIterator< ElementType, Dim > inIterator = ...
	M4D::Imaging::ImageIterator< OutElementType, Dim > outIterator = ...
	

	/*Coordinates< uint32, Dim > firstCorner = kernel.CenterCoordinates<Dim>();
	Coordinates< uint32, Dim > secondCorner = inRegion.GetSize() - (kernel.Size<Dim>() - kernel.CenterCoordinates<Dim>());

	bool finished = false;
	Coordinates< uint32, Dim > actual( firstCorner );

	NormalAccessor< M4D::Imaging::ImageRegion< ElementType, Dim > accessor( inRegion );

	while( !finished ) {
	Process:
		accessor.SetCenter( actual )
		FilterKernel< ElementType >::ResultType tmpResult = kernel.Apply( accessor );
		postprocessor.Compute( tmpResult, outRegion, actual );
		for( unsigned d = 0; d < Dim; ++d ) {
			if( actual[d] >= secondCorner[d] ) {
				actual[d] = firstCorner[i];
			} else { 
				++(actual[d]);
				goto Process;
			}
		}
		break;
	}*/
}

template< 
	typename ElementType, 
	typename OutElementType,
	unsigned Dim, 
	template< typename InElement > 			class FilterKernel, 
	template< typename Input, typename Output > 	class PostProcessor 
	>
void
__ImageFilterComputationBorderCase(
		const M4D::Imaging::ImageRegion< ElementType, Dim > 		&inRegion,
		M4D::Imaging::ImageRegion< OutElementType, Dim > 		&outRegion,
		const FilterKernel< ElementType >		&kernel,
		const PostProcessor< typename FilterKernel< ElementType >::ResultType, OutElementType >	&postprocessor
	)
{

}


template< 
	typename ElementType, 
	typename OutElementType,
	unsigned Dim, 
	template< typename InElement > 			class FilterKernel, 
	template< typename Input, typename Output > 	class PostProcessor 
	>
void
ImageFilterComputation(
		const M4D::Imaging::ImageRegion< ElementType, Dim > 		&inRegion,
		M4D::Imaging::ImageRegion< OutElementType, Dim > 		&outRegion,
		const FilterKernel< ElementType >		&kernel,
		const PostProcessor< typename FilterKernel< ElementType >::ResultType, OutElementType >	&postprocessor
	)
{
	__ImageFilterComputationFarFromBorder( inRegion, outRegion, kernel, postprocessor );

	__ImageFilterComputationBorderCase( inRegion, outRegion, kernel, postprocessor );
}


} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */
#endif /*FILTER_COMPUTATION_H*/

/** @} */

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
	)
{
	uint32 width = mask.size[0];
	uint32 height = mask.size[1];
	uint32 hwidth = mask.center[0];
	uint32 hheight = mask.center[1];
	//TODO check
	
	uint32 firstBorder[2];
	uint32 secondBorder[2];
	Coordinates< int32, 2 > coords;
	typename TypeTraits< ElementType >::SuperiorFloatType result = TypeTraits< ElementType >::Zero;
	for( coords[1] = 0; static_cast<uint32>(coords[1]) < hheight; ++coords[1] ) {
		firstBorder[1] = hheight - coords[1];
		secondBorder[1] = mask.size[1];
		for( coords[0] = 0; static_cast<uint32>(coords[0]) < inRegion.GetSize(0); ++coords[0] ) {
			firstBorder[0] = Max( static_cast<int32>(hwidth - coords[0]), 0 );
			secondBorder[0] = Min( inRegion.GetSize(0)-coords[0]+hwidth, mask.size[0] );
			result = addition + ApplyConvolutionMaskMirrorBorder( 
						inRegion.GetPointer( coords ), 
						inRegion.GetStride(), 
						firstBorder,
						secondBorder,
						mask,
						multiplication
						);
			postprocessor( result, outRegion.GetElement( coords ) ); 
		}
	}
	for( coords[1] = inRegion.GetSize(1)-height+hheight; static_cast<uint32>(coords[1]) < inRegion.GetSize(1); ++coords[1] ) {
		firstBorder[1] = 0;
		secondBorder[1] = inRegion.GetSize(1) - coords[1]+hheight;
		for( coords[0] = 0; static_cast<uint32>(coords[0]) < inRegion.GetSize(0); ++coords[0] ) {
			firstBorder[0] = Max( static_cast<int32>(hwidth - coords[0]), 0 );
			secondBorder[0] = Min( inRegion.GetSize(0)-coords[0]+hwidth, mask.size[0] );
			result = addition + ApplyConvolutionMaskMirrorBorder( 
						inRegion.GetPointer( coords ), 
						inRegion.GetStride(), 
						firstBorder,
						secondBorder,
						mask,
						multiplication
						);
			postprocessor( result, outRegion.GetElement( coords ) ); 

		}
	}
	for( coords[1] = hheight; static_cast<uint32>(coords[1]) < ( inRegion.GetSize(1) - height + hheight ); ++coords[1] ) {
		firstBorder[1] = 0;
		secondBorder[1] = mask.size[1];
		for( coords[0] = 0; static_cast<uint32>(coords[0]) < hwidth; ++coords[0] ) {
			firstBorder[0] = hwidth - coords[0];
			secondBorder[0] = mask.size[0];
			result = addition + ApplyConvolutionMaskMirrorBorder( 
						inRegion.GetPointer( coords ), 
						inRegion.GetStride(), 
						firstBorder,
						secondBorder,
						mask,
						multiplication
						);
			postprocessor( result, outRegion.GetElement( coords ) ); 
		}
		for( coords[0] = inRegion.GetSize(0)-width+hwidth; static_cast<uint32>(coords[0]) < inRegion.GetSize(0); ++coords[0] ) {
			firstBorder[0] = 0;
			secondBorder[0] = inRegion.GetSize(0)-coords[0]+hwidth;
			result = addition + ApplyConvolutionMaskMirrorBorder( 
						inRegion.GetPointer( coords ), 
						inRegion.GetStride(), 
						firstBorder,
						secondBorder,
						mask,
						multiplication
						);
			postprocessor( result, outRegion.GetElement( coords ) ); 
		}
	}

	for( coords[1] = hheight; static_cast<uint32>(coords[1]) < ( inRegion.GetSize(1) - height + hheight ); ++coords[1] ) {
		for( coords[0] = hwidth; static_cast<uint32>(coords[0]) < ( inRegion.GetSize(0) - width + hwidth ); ++coords[0] ) {
			result = addition + ApplyConvolutionMask( 
						inRegion.GetPointer( coords ), 
						inRegion.GetStride(), 
						mask,
						multiplication
						);
			postprocessor( result, outRegion.GetElement( coords ) ); 
			
		}
	}
}*/
