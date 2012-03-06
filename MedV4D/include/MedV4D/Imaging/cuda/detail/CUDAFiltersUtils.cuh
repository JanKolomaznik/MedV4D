#ifndef CUDA_FILTERS_UTILS_CUH
#define CUDA_FILTERS_UTILS_CUH

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>


template< typename TElement >
struct Buffer1D
{
	Buffer1D( size_t aLength, TElement *aData ): mData( aData ), mLength( aLength )
	{ /*empty*/ }

	size_t		mLength;
	TElement	*mData;
};

template< typename TElement >
struct Buffer3D
{
	Buffer3D( uint3 aSize, int3 aStrides, TElement *aData ): mSize( aSize ), mStrides( aStrides ), mData( aData ), mLength( aSize.x*aSize.y*aSize.z )
	{ /*empty*/ }
	Buffer3D( uint3 aSize, int3 aStrides, size_t aLength, TElement *aData ): mSize( aSize ), mStrides( aStrides ), mData( aData ), mLength( aLength )
	{ /*empty*/ }

	uint3		mSize;
	int3		mStrides;
	size_t		mLength;
	TElement	*mData;
};



template< typename TElement, unsigned tRadius, unsigned syStride, unsigned szStride >
__device__ inline void
FillSharedMemory3D_8x8x8( TElement data[], uint sidx, TElement *buffer, int3 strides, int3 size, int3 blockOrigin, int3 coordinates, int idx )
{
	const int cBlockDim = 8;

	data[sidx] = buffer[ idx ];
	
	uint3 sIdx;
	int3 mCoordinates = blockOrigin;
	switch( threadIdx.z ) {
	case 0:
		sIdx.x = threadIdx.x + tRadius;
		sIdx.y = threadIdx.y + tRadius;
		sIdx.z = 0;
		break;
	case 1:
		sIdx.x = threadIdx.x + tRadius;
		sIdx.y = threadIdx.y + tRadius;
		sIdx.z = cBlockDim + tRadius;
		break;
	case 2:
		sIdx.x = threadIdx.x + tRadius;
		sIdx.y = 0;
		sIdx.z = threadIdx.y + tRadius;
		break;
	case 3:
		sIdx.x = threadIdx.x + tRadius;
		sIdx.y = cBlockDim + tRadius;
		sIdx.z = threadIdx.y + tRadius;
		break;
	case 4:
		sIdx.x = 0;
		sIdx.y = threadIdx.y + tRadius;
		sIdx.z = threadIdx.x + tRadius;
		break;
	case 5:
		sIdx.x = cBlockDim + tRadius;
		sIdx.y = threadIdx.y + tRadius;
		sIdx.z = threadIdx.x + tRadius;
		break;
	case 6:
		if ( threadIdx.y < 4 ) {
			sIdx.x = threadIdx.x + tRadius;
			sIdx.y = (threadIdx.y & 1)*(cBlockDim + tRadius);
			sIdx.z = (threadIdx.y >> 1)*(cBlockDim + tRadius);
		} else {
			sIdx.x = ((threadIdx.y-4) >> 1)*(cBlockDim + tRadius);
			sIdx.y = threadIdx.x + tRadius;
			sIdx.z = (threadIdx.y & 1)*(cBlockDim + tRadius);
		}
		break;
	case 7:
		if ( threadIdx.y < 4 ) {
			sIdx.x = (threadIdx.y & 1)*(cBlockDim + tRadius);
			sIdx.y = ((threadIdx.y) >> 1)*(cBlockDim + tRadius);
			sIdx.z = threadIdx.x + tRadius;
		} else {	
			sIdx.x = threadIdx.x < 4 ? 0 : (cBlockDim + tRadius);
			sIdx.y = (threadIdx.x >> 1) & 1 ? 0 : (cBlockDim + tRadius);
			sIdx.z = threadIdx.x & 1 ? 0 : (cBlockDim + tRadius);
		}
		break;
	default:
		break;
	}
	mCoordinates.x += sIdx.x - tRadius;
	mCoordinates.y += sIdx.y - tRadius;
	mCoordinates.z += sIdx.z - tRadius;
	ProjectionToInterval( mCoordinates, make_int3(0,0,0), size );
	data[sIdx.y*syStride + sIdx.z*szStride + sIdx.x] = buffer[ IdxFromCoordStrides( mCoordinates, strides ) ];
}

template< typename TInElement, typename TOutElement, typename TFilter >
__global__ void 
FilterKernel3D( Buffer3D< TInElement > inBuffer, Buffer3D< TOutElement > outBuffer, int3 blockResolution, TFilter filter )
{ 
	__shared__ TInElement data[MAX_SHARED_MEMORY];
	
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
	int idx = IdxFromCoordStrides( coordinates, inBuffer.mStrides );

	FillSharedMemory3D_8x8x8< TInElement, cRadius, syStride, szStride >( data, sidx, inBuffer.mData, inBuffer.mStrides, size, blockOrigin, coordinates, idx );
	
	__syncthreads();

	if( !projected ) {
		outBuffer.mData[idx] = filter( data, sidx, syStride, szStride, idx );
	}
}

#endif //CUDA_FILTERS_UTILS_CUH