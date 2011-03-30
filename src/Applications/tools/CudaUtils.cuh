#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda.h>
#include "common/Common.h"
#include "Imaging/ImageRegion.h"


#define MAX_BLOCK_SIZE 10
#define MAX_SHARED_MEMORY 1024

inline int3
Vector3iToInt3( const Vector3i &v )
{
	return make_int3( v[0], v[1], v[2] );
}

inline uint3
Vector3uToUint3( const Vector3u &v )
{
	return make_uint3( v[0], v[1], v[2] );
}

__device__ inline int
IdxFromCoordStrides( int3 coords, int3 strides )
{
	return coords.x * strides.x + coords.y * strides.y + coords.z * strides.z;
}

template < typename TType3 >
__device__ __host__ inline int3
toInt3( const TType3 &arg )
{
	return make_int3( arg.x, arg.y, arg.z );
}

template < typename TType3 >
__device__ __host__ inline uint3
toUint3( const TType3 &arg )
{
	return make_uint3( arg.x, arg.y, arg.z );
}

__device__ __host__ inline int3 
operator+( const int3 &a, const int3 & b )
{
	return make_int3( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ __host__ inline uint3 
operator+( const uint3 &a, const uint3 & b )
{
	return make_uint3( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ __host__ inline int3 
operator+( const int3 &a, const uint3 & b )
{
	return make_int3( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ __host__ inline int3 
operator+( const uint3 &a, const int3 & b )
{
	return make_int3( a.x + b.x, a.y + b.y, a.z + b.z );
}

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

template< typename TElement >
Buffer1D< TElement >
CudaAllocateBuffer( size_t aLength )
{
	TElement *pointer;
	cudaMalloc( &pointer, aLength * sizeof(TElement) );
	D_PRINT( "CUDA allocated \t" << aLength * sizeof(TElement) << " bytes\nelement size\t" << sizeof(TElement) << " bytes" );
	return Buffer1D< TElement >( aLength, pointer );
}


template< typename TElement >
Buffer3D< TElement >
CudaPrepareBuffer( Vector3u aSize )
{
	uint3 size = Vector3uToUint3( aSize );
	int3 strides = make_int3( 1, size.x, size.x * size.y );
	size_t length = size.x*size.y*size.z;
	TElement * dataPointer;
	cudaMalloc( &dataPointer, length * sizeof(TElement) );
	return Buffer3D< TElement >( size, strides, length, dataPointer );
}

template< typename TElement >
Buffer3D< TElement >
CudaBuffer3DFromImageRegion( const M4D::Imaging::ImageRegion< TElement, 3 > &region )
{
	return CudaPrepareBuffer<TElement>( region.GetSize() );
}

template< typename TElement >
Buffer3D< TElement >
CudaBuffer3DFromImageRegionCopy( const M4D::Imaging::ImageRegion< TElement, 3 > &region )
{
	Buffer3D< TElement > buffer = CudaBuffer3DFromImageRegion( region );
	cudaMemcpy( buffer.mData, region.GetPointer(), buffer.mLength * sizeof(TElement), cudaMemcpyHostToDevice );
	return buffer;
}

/*inline void
CheckCudaErrorState( std::string aErrorMessage )
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		_THROW_ M4D::ErrorHandling::ExceptionBase( TO_STRING( aErrorMessage << " : " << cudaGetErrorString( err) ) );
	}
}
*/


#define CheckCudaErrorState( aErrorMessage ) \
{\
	cudaError_t err = cudaGetLastError();\
	D_PRINT( aErrorMessage ); \
	if( cudaSuccess != err) {\
		_THROW_ M4D::ErrorHandling::ExceptionBase( TO_STRING( aErrorMessage << " : " << cudaGetErrorString( err) ) );\
	}\
}\

__device__ int3
GetBlockCoordinates( int3 blockResolution, uint blockId )
{
	int3 result;
	result.z = blockId / (blockResolution.x * blockResolution.y);
	blockId = blockId % (blockResolution.x * blockResolution.y);
	result.y = blockId / blockResolution.x;
	blockId = blockId % blockResolution.x;
	result.x = blockId;
	return result;
}

__device__ int3
GetBlockOrigin( dim3 blockSize, int3 blockCoords )
{
	return make_int3( 
			blockSize.x * blockCoords.x, 
			blockSize.y * blockCoords.y,
			blockSize.z * blockCoords.z
		   );
}

__device__ bool
ProjectionToInterval( int3 &v, const int3 &min, const int3 &max )
{
	bool result = false;
	if ( v.x < min.x ) {
		result |= true;
		v.x = min.x;
	}
	if ( v.x >= max.x ) {
		result |= true;
		v.x = max.x - 1;
	}
	if ( v.y < min.y ) {
		result |= true;
		v.y = min.y;
	}
	if ( v.y >= max.y ) {
		result |= true;
		v.y = max.y - 1;
	}
	if ( v.z < min.z ) {
		result |= true;
		v.z = min.z;
	}
	if ( v.z >= max.z ) {
		result |= true;
		v.z = max.z - 1;
	}
	return result;
}

int3
GetBlockResolution( uint3 volumeSize, dim3 blockSize, int3 radius )
{
	return make_int3(
		       ( volumeSize.x + blockSize.x -1 - 2*radius.x ) / (blockSize.x - 2*radius.x),
		       ( volumeSize.y + blockSize.y -1 - 2*radius.y ) / (blockSize.y - 2*radius.y),
		       ( volumeSize.z + blockSize.z -1 - 2*radius.z ) / (blockSize.z - 2*radius.z)
			);
}

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
	
	/*uint sidx = (threadIdx.y+radius.y) * syStride + (threadIdx.z+radius.z) * szStride + threadIdx.x + radius.x;
	data[sidx] = inBuffer.mData[ idx ];
	
	uint3 sIdx;
	int3 mCoordinates = blockOrigin;
	switch( threadIdx.z ) {
	case 0:
		sIdx.x = threadIdx.x + radius.x;
		sIdx.y = threadIdx.y + radius.y;
		sIdx.z = 0;
		break;
	case 1:
		sIdx.x = threadIdx.x + radius.x;
		sIdx.y = threadIdx.y + radius.y;
		sIdx.z = blockDim.z + radius.z;
		break;
	case 2:
		sIdx.x = threadIdx.x + radius.x;
		sIdx.y = 0;
		sIdx.z = threadIdx.y + radius.z;
		break;
	case 3:
		sIdx.x = threadIdx.x + radius.x;
		sIdx.y = blockDim.y + radius.y;
		sIdx.z = threadIdx.y + radius.z;
		break;
	case 4:
		sIdx.x = 0;
		sIdx.y = threadIdx.y + radius.y;
		sIdx.z = threadIdx.x + radius.z;
		break;
	case 5:
		sIdx.x = blockDim.x + radius.x;
		sIdx.y = threadIdx.y + radius.y;
		sIdx.z = threadIdx.x + radius.z;
		break;
	case 6:
		if ( threadIdx.y < 4 ) {
			sIdx.x = threadIdx.x + radius.x;
			sIdx.y = (threadIdx.y & 1)*(blockDim.y + radius.y);
			sIdx.z = (threadIdx.y >> 1)*(blockDim.z + radius.z);
		} else {
			sIdx.x = ((threadIdx.y-4) >> 1)*(blockDim.x + radius.x);
			sIdx.y = threadIdx.x + radius.x;
			sIdx.z = (threadIdx.y & 1)*(blockDim.z + radius.z);
		}
		break;
	case 7:
		if ( threadIdx.y < 4 ) {
			sIdx.x = (threadIdx.y & 1)*(blockDim.x + radius.x);
			sIdx.y = ((threadIdx.y) >> 1)*(blockDim.y + radius.y);
			sIdx.z = threadIdx.x + radius.z;
		} else {	
			sIdx.x = threadIdx.x < 4 ? 0 : (blockDim.x + radius.x);
			sIdx.y = (threadIdx.x >> 1) & 1 ? 0 : (blockDim.y + radius.y);
			sIdx.z = threadIdx.x & 1 ? 0 : (blockDim.z + radius.z);
		}
		break;
	default:
		break;
	}
	mCoordinates.x += sIdx.x - radius.x;
	mCoordinates.y += sIdx.y - radius.y;
	mCoordinates.z += sIdx.z - radius.z;
	ProjectionToInterval( mCoordinates, make_int3(0,0,0), make_int3( size.x, size.y, size.z ) );
	data[sIdx.y*syStride + sIdx.z*szStride + sIdx.x] = inBuffer.mData[ mCoordinates.x * strides.x + mCoordinates.y * strides.y + mCoordinates.z * strides.z ];
	*/
	__syncthreads();

	if( !projected ) {
		outBuffer.mData[idx] = filter( data, sidx, syStride, szStride );
	}
}

#endif /*CUDA_UTILS_CUH*/
