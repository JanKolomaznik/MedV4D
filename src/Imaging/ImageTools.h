#ifndef _IMAGE_TOOLS_H
#define _IMAGE_TOOLS_H

#include "common/Common.h"
#include "Imaging/ImageRegion.h"
#include "Imaging/Histogram.h"

namespace M4D {
namespace Imaging {

template< typename TImageRegion, typename THistogram >
void
AddRegionToHistogram( THistogram &aHistogram, const TImageRegion &aRegion )
{
	//TODO - test TImageRegion type
	int32 min = aHistogram.GetMin() - 1;
	int32 max = aHistogram.GetMax();
	typename TImageRegion::Iterator it = aRegion.GetIterator();
	while ( !it.IsEnd() ) {
		aHistogram.FastIncCell( clampToInterval( min, max, (int32)*it ) );
		++it;
	}
}

template< typename THistogram, typename TElementType >
void
AddArrayToHistogram( THistogram &aHistogram, const TElementType *aArray, size_t aSize )
{
	//TODO - test TImageRegion type
	int32 min = aHistogram.GetMin() - 1;
	int32 max = aHistogram.GetMax();
	const TElementType *end = aArray + aSize;
	while ( aArray < end ) {
		aHistogram.FastIncCell( clampToInterval( min, max, (int32)*aArray ) );
		++aArray;
	}
}

template< typename THistogram, typename TElementType >
typename THistogram::Ptr
PrepareHistogramForArray( const TElementType *aArray, size_t aSize )
{
	//TODO type test, parallelization
	const TElementType *end = aArray + aSize;
	const TElementType *ptr = aArray;
	TElementType minimum = TypeTraits< TElementType >::Max;
	TElementType maximum = TypeTraits< TElementType >::Min;
	while ( ptr < end ) {
		minimum = min( minimum, *ptr );
		maximum = max( maximum, *ptr );
		++ptr;
	}

	return THistogram::Create( int32(minimum), int32(maximum), true );
}

template< typename THistogram, typename TRegion >
typename THistogram::Ptr
CreateHistogramForImageRegion( const TRegion &aRegion )
{
	const typename TRegion::Element *array = aRegion.GetPointer();
	typename THistogram::Ptr histogram = PrepareHistogramForArray< THistogram, typename TRegion::Element >( array, VectorCoordinateProduct( aRegion.GetSize() ) );

	AddArrayToHistogram< THistogram, typename TRegion::Element >( *histogram, array, VectorCoordinateProduct( aRegion.GetSize() ) );

	return histogram;
}


} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE_TOOLS_H*/
