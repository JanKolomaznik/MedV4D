#ifndef LOCAL_MINIMA_DETECTION_CUH
#define LOCAL_MINIMA_DETECTION_CUH

#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"

template< typename TElement >
struct LocalMinima3DFtor
{
	LocalMinima3DFtor( TElement aThreshold ): radius( make_int3( 1, 1, 1 ) ), mThreshold( aThreshold )
	{}

	__device__ uint8
	operator()( TElement data[], uint idx, uint syStride, uint szStride, uint gIdx )
	{
		TElement value = data[idx];
		for ( int i = idx-1; i <= idx+1; ++i ) {
			for ( int j = i-syStride; j <= i+syStride; j+=syStride ) {
				for ( int k = j-szStride; k <= j+szStride; k+=szStride ) {
					value = min( data[k], value );
				}
			}
		}
		return ( value >= data[idx] && data[idx] < mThreshold ) ? 255 : 0;
	}
	int3 radius;
	TElement mThreshold;
};

template< typename TElement >
struct LocalMinimaRegions3DFtor
{
	LocalMinimaRegions3DFtor( TElement aThreshold ): radius( make_int3( 1, 1, 1 ) ), mThreshold( aThreshold )
	{}

	__device__ uint32
	operator()( TElement data[], uint idx, uint syStride, uint szStride, uint gIdx )
	{
		TElement value = data[idx];
		for ( int i = idx-1; i <= idx+1; ++i ) {
			for ( int j = i-syStride; j <= i+syStride; j+=syStride ) {
				for ( int k = j-szStride; k <= j+szStride; k+=szStride ) {
					value = min( data[k], value );
				}
			}
		}
		/*uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint32 tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
		uint32 ridx = blockId * blockDim.x * blockDim.y + tid;*/

		return ( value >= data[idx] && data[idx] < mThreshold ) ? gIdx + 1: 0;
	}
	int3 radius;
	TElement mThreshold;
};

template< typename TElement >
__global__ void 
ConsolidationScanImage( Buffer3D< TElement > inBuffer, Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut, int3 blockResolution )
{
	__shared__ TElement inData[MAX_SHARED_MEMORY];
	__shared__ uint32 labelData[MAX_SHARED_MEMORY];
	
	const int cBlockDim = 8;
	const int cRadius = 1;
	const uint syStride = cBlockDim+2*cRadius;
	const uint szStride = (cBlockDim+2*cRadius) * (cBlockDim+2*cRadius);

	uint sidx = (threadIdx.y+cRadius) * syStride + (threadIdx.z+cRadius) * szStride + threadIdx.x + cRadius;
	int3 size = toInt3( inBuffer.mSize );
	int3 blockCoordinates = GetBlockCoordinates( blockResolution, __mul24(blockIdx.y, gridDim.x) + blockIdx.x );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	
	
	int3 coordinates = blockOrigin + threadIdx;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), size );
	int idx1 = IdxFromCoordStrides( coordinates, inBuffer.mStrides );
	int idx2 = IdxFromCoordStrides( coordinates, buffer.mStrides );

	
	FillSharedMemory3D_8x8x8< TElement, cRadius, syStride, szStride >( inData, sidx, inBuffer.mData, inBuffer.mStrides, size, blockOrigin, coordinates, idx1 );
	FillSharedMemory3D_8x8x8< uint32, cRadius, syStride, szStride >( labelData, sidx, buffer.mData, buffer.mStrides, size, blockOrigin, coordinates, idx2 );
	
	__syncthreads();

	if( !projected && labelData[sidx] > 0 ) {
		for ( int i = sidx-1; i <= sidx+1; ++i ) {
			for ( int j = i-syStride; j <= i+syStride; j+=syStride ) {
				for ( int k = j-szStride; k <= j+szStride; k+=szStride ) {
					if( labelData[k] == 0 && inData[k] <= inData[sidx] ) {
						lut.mData[ labelData[sidx] - 1 ] = 0;
						return;
					}
				}
			}
		}
	}
}

__global__ void 
MarkUsedIds( Buffer3D< uint32 > outBuffer, Buffer1D< uint32 > lut );

__global__ void 
UpdateLabelsFromScan( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut );


#endif //LOCAL_MINIMA_DETECTION_CUH