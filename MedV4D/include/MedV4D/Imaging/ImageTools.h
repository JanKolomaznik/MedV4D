#ifndef _IMAGE_TOOLS_H
#define _IMAGE_TOOLS_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/ImageRegion.h"
#include "MedV4D/Imaging/Histogram.h"

#ifdef USE_TBB
#include <tbb/tbb.h>
//#include <tbb/parallel_reduce.h>
//#include <tbb/task_scheduler_init.h>
//#include <tbb/blocked_range.h>
#endif

namespace M4D
{
namespace Imaging {

template< typename TImageRegion, typename THistogram >
void
addRegionToHistogram ( THistogram &aHistogram, const TImageRegion &aRegion )
{
	//TODO - test TImageRegion type
	int32 min = aHistogram.GetMin() - 1;
	int32 max = aHistogram.GetMax();
	typename TImageRegion::Iterator it = aRegion.GetIterator();
	while ( !it.IsEnd() ) {
		aHistogram.FastIncCell ( clamp( min, max, ( int32 ) *it ) );
		++it;
	}
}


#ifdef USE_TBB

template< typename THistogram, typename TElementType >
struct HistogramFromArrayFtor {
	HistogramFromArrayFtor ( const TElementType *a, int32 aMinimum, int32 aMaximum ) : mArray ( a ), mHistogram ( aMinimum, aMaximum, true ), mMinimum ( aMinimum-1 ), mMaximum ( aMaximum ) { }
	HistogramFromArrayFtor ( HistogramFromArrayFtor &x, tbb::split ) : mArray ( x.mArray ), mHistogram ( x.mHistogram.GetMin(), x.mHistogram.GetMax(), true ) {
	}

	void
	join ( const HistogramFromArrayFtor &x ) {
		mHistogram += x.mHistogram;
	}
	void
	operator() ( const tbb::blocked_range<size_t> &r ) {
		const TElementType *a = mArray;
		size_t end = r.end();
		for ( size_t i = r.begin(); i!=end; ++i ) {
			mHistogram.FastIncCell ( clamp( mMinimum, mMaximum, static_cast<int32> ( a[i] ) ) );
		}
	}


	const TElementType *mArray;
	THistogram mHistogram;
	int32 mMinimum;
	int32 mMaximum;
};

template< typename THistogram, typename TElementType >
void
addArrayToHistogram ( THistogram &aHistogram, const TElementType *aArray, size_t aSize )
{
	//TODO - test TImageRegion type
	/*int32 min = aHistogram.GetMin() - 1;
	int32 max = aHistogram.GetMax();
	const TElementType *end = aArray + aSize;
	while ( aArray < end ) {
		aHistogram.FastIncCell( clamp( min, max, (int32)*aArray ) );
		++aArray;
	}*/

	HistogramFromArrayFtor< THistogram, TElementType > ftor ( aArray, aHistogram.GetMin(), aHistogram.GetMax() );
	tbb::parallel_reduce ( tbb::blocked_range<size_t> ( 0, aSize ), ftor );

	swap ( aHistogram, ftor.mHistogram );
}


template< typename TElementType >
struct MinMaxForArrayFtor {
	MinMaxForArrayFtor ( const TElementType *a ) : mArray ( a ), mMin ( TypeTraits< TElementType >::Max ), mMax ( TypeTraits< TElementType >::Min ) { }

	MinMaxForArrayFtor ( MinMaxForArrayFtor &x, tbb::split ) : mArray ( x.mArray ), mMin ( x.mMin ), mMax ( x.mMax ) { }

	void
	join ( const MinMaxForArrayFtor &x ) {
		mMin = M4D::min ( mMin, x.mMin );
		mMax = M4D::max ( mMax, x.mMax );
	}
	void
	operator() ( const tbb::blocked_range<size_t> &r ) {
		const TElementType *a = mArray;
		TElementType pMin = mMin;
		TElementType pMax = mMax;
		size_t end = r.end();
		for ( size_t i = r.begin(); i!=end; ++i ) {
			pMin = M4D::min ( pMin, a[i] );
			pMax = M4D::max ( pMax, a[i] );
		}
		mMin = pMin;
		mMax = pMax;
	}


	const TElementType *mArray;
	TElementType mMin;
	TElementType mMax;
};

template< typename THistogram, typename TElementType >
typename THistogram::Ptr
prepareHistogramForArray ( const TElementType *aArray, size_t aSize )
{
	ASSERT ( aArray );
	MinMaxForArrayFtor< TElementType > ftor ( aArray );

	tbb::parallel_reduce ( tbb::blocked_range<size_t> ( 0,aSize ), ftor );
	//TODO type test, parallelization
	//const TElementType *end = aArray + aSize;
	//const TElementType *ptr = aArray;
	TElementType minimum = ftor.mMin;
	TElementType maximum = ftor.mMax;
	/*while ( ptr < end ) {
		minimum = min( minimum, *ptr );
		maximum = max( maximum, *ptr );
		++ptr;
	}*/
	D_PRINT ( "prepareHistogramForArray() minimum = " << minimum );
	D_PRINT ( "prepareHistogramForArray() maximum = " << maximum );

	return THistogram::Create ( int32 ( minimum ), int32 ( maximum ), true );
}

#else
template< typename THistogram, typename TElementType >
void
addArrayToHistogram ( THistogram &aHistogram, const TElementType *aArray, size_t aSize )
{
	//TODO - test TImageRegion type
	int32 min = aHistogram.getMin() - 1;
	int32 max = aHistogram.getMax();
	const TElementType *end = aArray + aSize;
	while ( aArray < end ) {
		aHistogram.fastIncCell ( clamp( min, max, ( int32 ) *aArray ) );
		++aArray;
	}
}

template< typename THistogram, typename TElementType >
typename THistogram::Ptr
prepareHistogramForArray ( const TElementType *aArray, size_t aSize )
{
	//TODO type test, parallelization
	const TElementType *end = aArray + aSize;
	const TElementType *ptr = aArray;
	TElementType minimum = TypeTraits< TElementType >::Max;
	TElementType maximum = TypeTraits< TElementType >::Min;
	while ( ptr < end ) {
		minimum = min ( minimum, *ptr );
		maximum = max ( maximum, *ptr );
		++ptr;
	}

	return THistogram::create ( int32 ( minimum ), int32 ( maximum ), true );
}
#endif


template< typename THistogram, typename TRegion >
typename THistogram::Ptr
createHistogramForImageRegion ( const TRegion &aRegion )
{
	const typename TRegion::Element *array = aRegion.GetPointer();
	typename THistogram::Ptr histogram = prepareHistogramForArray< THistogram, typename TRegion::Element > ( array, VectorCoordinateProduct ( aRegion.GetSize() ) );

	addArrayToHistogram< THistogram, typename TRegion::Element > ( *histogram, array, VectorCoordinateProduct ( aRegion.GetSize() ) );

	return histogram;
}

template< typename THistogram, typename TRegion >
THistogram
createHistogramForImageRegion2(const TRegion &aRegion)
{
	auto it = aRegion.GetIterator();

	double minimum = *it;
	double maximum = *it;

	while ( !it.IsEnd() ) {
		minimum = std::min<double>(minimum, *it);
		maximum = std::max<double>(maximum, *it);
		++it;
	}

	THistogram histogram(minimum, maximum, 500);
	it = aRegion.GetIterator();
	while ( !it.IsEnd() ) {
		histogram.put(*it);
		++it;
	}
	return histogram;
}

template< typename TScatterPlot, typename TRegion >
TScatterPlot
createGradientScatterPlotForImageRegion(const TRegion &aRegion)
{
	typedef typename TScatterPlot::Value Value;
	typedef typename TScatterPlot::CellCoordinates CellCoordinates;
	TScatterPlot scatterPlot(Value(0,0), Value(2000,500), CellCoordinates(2000, 500));

	auto from = aRegion.GetMinimum();
	auto to = aRegion.GetMaximum();
	auto element = aRegion.GetElementExtents();

	auto index = from;

	for (; index[2] < to[2]-1; ++index[2]) {
		for (index[1] = from[1]; index[1] < to[1]-1; ++index[1]) {
			for (index[0] = from[0]; index[0] < to[0]-1; ++index[0]) {
				auto value = aRegion.GetElement(index);
				decltype(value) gradient = 0;
				for (int i = 0; i < 3; ++i) {
					auto index2 = index;
					++index2[i];
					auto d = (value - aRegion.GetElementFast(index2)) / element[i];
					gradient += d * d;
				}
				scatterPlot.put(value, std::sqrt(gradient));
			}
		}
	}
	return scatterPlot;
}

template< typename TImage, typename TContainer >
void
fillDataAlongRay ( float aSampleDistance, Vector<float,TImage::Dimension> aDirection, Vector<float,TImage::Dimension> aStartPosition, TContainer &aContainer )
{

}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE_TOOLS_H*/
