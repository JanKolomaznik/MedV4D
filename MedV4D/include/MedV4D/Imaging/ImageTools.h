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
AddRegionToHistogram ( THistogram &aHistogram, const TImageRegion &aRegion )
{
        //TODO - test TImageRegion type
        int32 min = aHistogram.GetMin() - 1;
        int32 max = aHistogram.GetMax();
        typename TImageRegion::Iterator it = aRegion.GetIterator();
        while ( !it.IsEnd() ) {
                aHistogram.FastIncCell ( clampToInterval ( min, max, ( int32 ) *it ) );
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
                        mHistogram.FastIncCell ( clampToInterval ( mMinimum, mMaximum, static_cast<int32> ( a[i] ) ) );
                }
        }


        const TElementType *mArray;
        THistogram mHistogram;
        int32 mMinimum;
        int32 mMaximum;
};

template< typename THistogram, typename TElementType >
void
AddArrayToHistogram ( THistogram &aHistogram, const TElementType *aArray, size_t aSize )
{
        //TODO - test TImageRegion type
        /*int32 min = aHistogram.GetMin() - 1;
        int32 max = aHistogram.GetMax();
        const TElementType *end = aArray + aSize;
        while ( aArray < end ) {
        	aHistogram.FastIncCell( clampToInterval( min, max, (int32)*aArray ) );
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
PrepareHistogramForArray ( const TElementType *aArray, size_t aSize )
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
        D_PRINT ( "PrepareHistogramForArray() minimum = " << minimum );
        D_PRINT ( "PrepareHistogramForArray() maximum = " << maximum );

        return THistogram::Create ( int32 ( minimum ), int32 ( maximum ), true );
}

#else
template< typename THistogram, typename TElementType >
void
AddArrayToHistogram ( THistogram &aHistogram, const TElementType *aArray, size_t aSize )
{
        //TODO - test TImageRegion type
        int32 min = aHistogram.GetMin() - 1;
        int32 max = aHistogram.GetMax();
        const TElementType *end = aArray + aSize;
        while ( aArray < end ) {
                aHistogram.FastIncCell ( clampToInterval ( min, max, ( int32 ) *aArray ) );
                ++aArray;
        }
}

template< typename THistogram, typename TElementType >
typename THistogram::Ptr
PrepareHistogramForArray ( const TElementType *aArray, size_t aSize )
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

        return THistogram::Create ( int32 ( minimum ), int32 ( maximum ), true );
}
#endif


template< typename THistogram, typename TRegion >
typename THistogram::Ptr
CreateHistogramForImageRegion ( const TRegion &aRegion )
{
        const typename TRegion::Element *array = aRegion.GetPointer();
        typename THistogram::Ptr histogram = PrepareHistogramForArray< THistogram, typename TRegion::Element > ( array, VectorCoordinateProduct ( aRegion.GetSize() ) );

        AddArrayToHistogram< THistogram, typename TRegion::Element > ( *histogram, array, VectorCoordinateProduct ( aRegion.GetSize() ) );

        return histogram;
}

template< typename TImage, typename TContainer >
void
fillDataAlongRay ( float aSampleDistance, Vector<float,TImage::Dimension> aDirection, Vector<float,TImage::Dimension> aStartPosition, TContainer &aContainer )
{

}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE_TOOLS_H*/
