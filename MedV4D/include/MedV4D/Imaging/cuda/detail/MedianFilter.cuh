#ifndef MEDIAN_FILTER_CUH
#define MEDIAN_FILTER_CUH

#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"

template< typename TElement >
struct MedianFilter3DFtor
{
	MedianFilter3DFtor()
	{}

	__device__ TElement
	operator()( TElement data[], uint idx, uint syStride, uint szStride, uint gIdx )
	{
		TElement tmp[9];
		int ii = 0;
		for ( int i = idx-1; i <= idx+1; ++i ) {
			for ( int j = i-syStride; j <= i+syStride; j+=syStride ) {
				for ( int k = j-szStride; k <= j+szStride; k+=szStride ) {
					tmp[ii++] = data[k];
				}
			}
		}
		for ( int i = 0; i < 8; ++i ) {
			int valIdx = i;
			TElement value = tmp[valIdx];
			for ( int j = i+1; j < 9; ++j ) {
				if( value > tmp[j] ) {
					valIdx = j;
					value = tmp[valIdx];
				}
			}
			cudaSwap( tmp[i], tmp[valIdx] );
		}
		return tmp[4];	
	}

	//TElement threshold;
	//int3 radius;
};

#endif //MEDIAN_FILTER_CUH
