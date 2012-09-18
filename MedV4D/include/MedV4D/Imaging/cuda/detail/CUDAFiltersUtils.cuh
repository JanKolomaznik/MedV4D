#ifndef CUDA_FILTERS_UTILS_CUH
#define CUDA_FILTERS_UTILS_CUH

#include <cuda.h>
#include <thrust/device_vector.h>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/ImageRegion.h"
#include <string>

//#define GLM_FORCE_CUDA
//#include <glm/glm.hpp>
#define MAX_BLOCK_SIZE 10
#define MAX_SHARED_MEMORY 1024


__device__ inline float 
atomicFloatCAS(float *address, float old, float val)
{
	int i_val = __float_as_int(val);
	int tmp0 = __float_as_int(old);

	return __int_as_float( atomicCAS((int *)address, tmp0, i_val) );
}

template< typename TType >
__device__ inline void
cudaSwap( TType &a, TType &b )
{
	TType tmp = a;
	a = b;
	b = tmp;
}

inline std::string
cudaMemoryInfoText()
{
	size_t free;
	size_t total;
	cudaError_t err = cudaMemGetInfo( &free, &total);
	if( cudaSuccess != err) {
		_THROW_ M4D::ErrorHandling::ExceptionBase( TO_STRING( "Failed to get CUDA memory info : " << cudaGetErrorString( err) ) );
	}
	return TO_STRING( "Free memory: " << bytesToHumanReadableFormat( free ) << "; Total memory " <<  bytesToHumanReadableFormat( total ) << "; Occupied: " << 100.0f * float(total - free)/total << "%");
}

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

__device__ __host__ inline bool
operator==( const int3 &a, const int3 & b )
{
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__device__ __host__ inline bool
operator==( const uint3 &a, const uint3 & b )
{
	return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__device__ __host__ inline int3 
operator+( const int3 &a, const int3 & b )
{
	return make_int3( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ __host__ inline int3 
operator-( const int3 &a, const int3 & b )
{
	return make_int3( a.x - b.x, a.y - b.y, a.z - b.z );
}

__device__ __host__ inline uint3 
operator+( const uint3 &a, const uint3 & b )
{
	return make_uint3( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ __host__ inline uint3 
operator-( const uint3 &a, const uint3 & b )
{
	return make_uint3( a.x - b.x, a.y - b.y, a.z - b.z );
}

__device__ __host__ inline int3 
operator+( const int3 &a, const uint3 & b )
{
	return make_int3( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ __host__ inline int3 
operator-( const int3 &a, const uint3 & b )
{
	return make_int3( a.x - b.x, a.y - b.y, a.z - b.z );
}

__device__ __host__ inline int3 
operator+( const uint3 &a, const int3 & b )
{
	return make_int3( a.x + b.x, a.y + b.y, a.z + b.z );
}

__device__ __host__ inline int3 
operator-( const uint3 &a, const int3 & b )
{
	return make_int3( a.x - b.x, a.y - b.y, a.z - b.z );
}

__host__ inline std::ostream &
operator<<( std::ostream &stream, const dim3 &v )
{
	stream << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
	return stream;
}

inline dim3
splitCountTo2dGrid( size_t aCount )
{
	uint threshold = 1<<15;
	if( aCount > threshold ) {
		return dim3( threshold, (aCount + threshold -1)/ threshold, 1 );
	}
	return dim3( aCount, 1, 1 );
}

struct ReferenceCounter {
	typedef size_t size_type;

	explicit ReferenceCounter(): mReferenceCount( 1 ) {
		// Nothing to do.
	}

	size_type mReferenceCount;
};

template< typename TElement >
struct Buffer1D
{
	Buffer1D( size_t aLength, TElement *aData, bool aDealloc ): mDealloc( false/*aDealloc*/ ), mData( aData ), mLength( aLength )
	{ /*empty*/ }

	__device__ __host__ int
	size() const
	{ return mLength; }
	
	__host__
	~Buffer1D()
	{
		if( mDealloc ) {
			D_PRINT( "1D buffer Deallocation :" << std::hex << size_t( mData ) );
			cudaFree( mData );
			mData = NULL;
			mLength = 0;
		}
	}

	bool mDealloc;
	int mCount;
	size_t		mLength;
	TElement	*mData;
};

template< typename TElement >
struct Buffer3D
{
	__host__
	Buffer3D( uint3 aSize, int3 aStrides, TElement *aData ): mSize( aSize ), mStrides( aStrides ), mData( aData ), mLength( aSize.x*aSize.y*aSize.z ), mReferenceCount( new ReferenceCounter )
	{ /*empty*/ }

	__host__
	Buffer3D( uint3 aSize, int3 aStrides, size_t aLength, TElement *aData ): mSize( aSize ), mStrides( aStrides ), mData( aData ), mLength( aLength ), mReferenceCount( new ReferenceCounter )
	{ /*empty*/ }

	__host__
	Buffer3D(const Buffer3D &aOther): mSize( aOther.mSize ), mStrides( aOther.mStrides ), mData( aOther.mData ), mLength( aOther.mLength ), mReferenceCount( aOther.mReferenceCount )
	{
		retain();
	}

	__host__
	Buffer3D( uint3 aSize ): mSize( aSize ), mReferenceCount( new ReferenceCounter )
	{ 
		allocate();
	}

	__host__
	Buffer3D( const Vector3u &aSize ): mSize( Vector3uToUint3(aSize) ), mReferenceCount( new ReferenceCounter )
	{
		try {
			allocate();
		} catch( ... ) {
			delete mReferenceCount;
			throw;
		}
	}

	

	__host__
	~Buffer3D()
	{
		release();
	}

	__host__ Buffer3D &
	operator=( Buffer3D other )
	{
		swap( other );
		return *this;
	}

	

	__host__ void 
	swap( Buffer3D &aOther ) {
		std::swap( mSize, aOther.mSize );
		std::swap( mStrides, aOther.mStrides );
		std::swap( mLength, aOther.mLength );
		std::swap( mData, aOther.mData );
		std::swap( mReferenceCount, aOther.mReferenceCount );
	}

	uint3		mSize;
	int3		mStrides;
	size_t		mLength;
	TElement		*mData;
	ReferenceCounter *mReferenceCount;


private:
	
	__host__ void 
	retain() 
	{
		++( mReferenceCount->mReferenceCount );
	}

	__host__ void 
	release() 
	{
		ASSERT( 0 < mReferenceCount->mReferenceCount );

		--( mReferenceCount->mReferenceCount );
		if ( 0 == mReferenceCount->mReferenceCount ) {
			D_PRINT( 
				boost::str( 
					boost::format( "CUDA Deallocating \t%1% bytes\nelement size\t %2% bytes\nfrom address %3% to %4%" ) 
					% (mLength * sizeof(TElement)) 
					% sizeof(TElement) 
					% boost::io::group(std::hex, std::showbase, size_t(mData) ) 
					% boost::io::group(std::hex, std::showbase, size_t(mData+mLength) ) ) 
				);


			cudaFree( mData );
			mData = NULL;
			mLength = 0;
		}
	}

	__host__ void
	allocate()
	{
		/*cudaPitchedPtr pitchedDevPtr;
		cudaExtent extent;
		extent.width = mSize.x;
		extent.height = mSize.y;
		extent.depth = mSize.z;

		cudaError_t errCode = cudaMalloc3D( &pitchedDevPtr, extent );
		if ( errCode != cudaSuccess ) {
			_THROW_ EAllocationFailed( "CUDA Buffer3D allocation failed" );
		}

		mStrides.x = 1;
		mStrides.y = pitchedDevPtr.pitch / sizeof(TElement);
		mStrides.z = mStrides.y * mSize.y;
		mLength = mStrides.z * mSize.z;
		mData = pitchedDevPtr.ptr;

		ASSERT( pitchedDevPtr.pitch == mStrides.y * sizeof(TElement) );*/

		mStrides.x = 1;
		mStrides.y = mSize.x;
		mStrides.z = mStrides.y * mSize.y;
		mLength = mSize.x*mSize.y*mSize.z;
		
		cudaError_t errCode = cudaMalloc( &mData, mLength * sizeof(TElement) );
		if ( errCode != cudaSuccess ) {
			_THROW_ EAllocationFailed( "CUDA Buffer3D allocation failed" );
		}

		D_PRINT( 
			boost::str( 
			boost::format( "CUDA allocated \t%1% bytes\nelement size\t %2% bytes\nfrom address %3% to %4%" ) 
			% (mLength * sizeof(TElement)) 
			% sizeof(TElement) 
			% boost::io::group(std::hex, std::showbase, size_t(mData) ) 
			% boost::io::group(std::hex, std::showbase, size_t(mData+mLength) ) ) 
			);
	}
};

__device__ inline int3
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

__device__ inline int3
GetBlockOrigin( dim3 blockSize, int3 blockCoords )
{
	return make_int3( 
			blockSize.x * blockCoords.x, 
			blockSize.y * blockCoords.y,
			blockSize.z * blockCoords.z
		   );
}

__device__ inline bool
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

inline int3
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
	
	__syncthreads();

	if( !projected ) {
		outBuffer.mData[idx] = filter( data, sidx, syStride, szStride, idx );
	}
}

template< typename TInElement, typename TOutElement, typename TFilter >
__global__ void 
FilterSimple3D( Buffer3D< TInElement > inBuffer, Buffer3D< TOutElement > outBuffer, int3 blockResolution, TFilter filter )
{
	int3 blockCoordinates = GetBlockCoordinates( blockResolution, __mul24(blockIdx.y, gridDim.x) + blockIdx.x );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	int3 coordinates = blockOrigin + threadIdx;
	int3 size = toInt3( inBuffer.mSize );

	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), size );
	int idx1 = IdxFromCoordStrides( coordinates, inBuffer.mStrides );
	int idx2 = IdxFromCoordStrides( coordinates, outBuffer.mStrides );

	if ( !projected ) {
		filter( inBuffer.mData[ idx1 ], outBuffer.mData[ idx2 ] );
	}
}


template< typename TElement >
Buffer1D< TElement >
CudaAllocateBuffer( size_t aLength )
{
	TElement *pointer;
	cudaError_t eCode = cudaMalloc( &pointer, aLength * sizeof(TElement) );
	if ( eCode != cudaSuccess ) {
		_THROW_ EAllocationFailed( "CUDA allocation failed" );
	}
	D_PRINT( "CUDA allocated \t" << aLength * sizeof(TElement) << " bytes\nelement size\t" 
			<< sizeof(TElement) << " bytes\nfrom address 0x" << std::hex << size_t(pointer) << " to 0x" << size_t(pointer+aLength) << std::dec );
	return Buffer1D< TElement >( aLength, pointer, true );
}

template< typename TElement >
Buffer1D< TElement >
cudaBufferFromThrustDeviceVector( thrust::device_vector< TElement > & aVector )
{
	return Buffer1D< TElement >( aVector.size(), thrust::raw_pointer_cast(&aVector[0]), false );
}

template< typename TElement >
Buffer3D< TElement >
CudaPrepareBuffer( Vector3u aSize )
{
	/*uint3 size = Vector3uToUint3( aSize );
	int3 strides = make_int3( 1, size.x, size.x * size.y );
	size_t length = size.x*size.y*size.z;
	TElement * dataPointer;
	cudaError_t eCode = cudaMalloc( &dataPointer, length * sizeof(TElement) );
	if ( eCode != cudaSuccess ) {
		_THROW_ EAllocationFailed( "CUDA allocation failed" );
	}
	D_PRINT( "CUDA allocated \t" << length * sizeof(TElement) << " bytes\nelement size\t" 
			<< sizeof(TElement) << " bytes\nfrom address 0x" << std::hex << size_t(dataPointer) << " to 0x" << size_t(dataPointer+length) << std::dec );
	return Buffer3D< TElement >( size, strides, length, dataPointer );*/
	return Buffer3D< TElement >( aSize );
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

#define CUDA_CHECK_RESULT_MSG( aErrorMessage, ... ) \
{\
	cudaError_t err = __VA_ARGS__ ;\
	if( cudaSuccess != err ) {\
		D_PRINT( aErrorMessage ); \
		_THROW_ M4D::ErrorHandling::ExceptionBase( TO_STRING( aErrorMessage << " : " << cudaGetErrorString( err) ) );\
	}\
}

#define CUDA_CHECK_RESULT( ... ) \
	CUDA_CHECK_RESULT_MSG( #__VA_ARGS__, __VA_ARGS__ )

#define CheckCudaErrorState( aErrorMessage ) \
	CUDA_CHECK_RESULT_MSG( aErrorMessage, cudaGetLastError() );



	


#endif //CUDA_FILTERS_UTILS_CUH
