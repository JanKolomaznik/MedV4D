#include "CudaUtils.cuh"
#include "Imaging/ImageRegion.h"
#include "Imaging/Image.h"
#include "Imaging/ImageFactory.h"


template< typename TElement >
struct SobelFilter3DFtor
{
	typedef typename TypeTraits< TElement >::SignedClosestType SignedElement;

	SobelFilter3DFtor( TElement aThreshold = 0 ): threshold( aThreshold ), radius( make_int3( 1, 1, 1 ) )
	{}

	__device__ TElement
	operator()( TElement data[], uint idx, uint syStride, uint szStride )
	{
		TElement val1 = abs( (2*static_cast<SignedElement>(data[idx+1]) - 2*static_cast<SignedElement>(data[idx-1])) 
			+ (static_cast<SignedElement>(data[idx+1 + syStride]) - static_cast<SignedElement>(data[idx-1 + syStride]))
			+ (static_cast<SignedElement>(data[idx+1 - syStride]) - static_cast<SignedElement>(data[idx-1 - syStride])) );
		TElement val2 = abs( (2*static_cast<SignedElement>(data[idx+syStride]) - 2*static_cast<SignedElement>(data[idx-syStride])) 
			+ (static_cast<SignedElement>(data[idx+1 + syStride]) - static_cast<SignedElement>(data[idx+1 - syStride]))
			+ (static_cast<SignedElement>(data[idx-1 + syStride]) - static_cast<SignedElement>(data[idx-1 - syStride])) );
		TElement val3 = abs( (2*static_cast<SignedElement>(data[idx+szStride]) - 2*static_cast<SignedElement>(data[idx-szStride])) 
			+ (static_cast<SignedElement>(data[idx + syStride + szStride]) - static_cast<SignedElement>(data[idx + syStride - szStride]))
			+ (static_cast<SignedElement>(data[idx - syStride + szStride]) - static_cast<SignedElement>(data[idx - syStride - szStride])) );
		TElement val4 = abs( (2*static_cast<SignedElement>(data[idx + syStride]) - 2*static_cast<SignedElement>(data[idx-syStride])) 
			+ (static_cast<SignedElement>(data[idx + syStride + szStride]) - static_cast<SignedElement>(data[idx - syStride + szStride]))
			+ (static_cast<SignedElement>(data[idx + syStride - szStride]) - static_cast<SignedElement>(data[idx - syStride - szStride])) );
		TElement val5 = abs( (2*static_cast<SignedElement>(data[idx+1]) - 2*static_cast<SignedElement>(data[idx-1])) 
			+ (static_cast<SignedElement>(data[idx + 1 + szStride]) - static_cast<SignedElement>(data[idx-1 + szStride]))
			+ (static_cast<SignedElement>(data[idx + 1 - szStride]) - static_cast<SignedElement>(data[idx-1 - szStride])) );
		TElement val6 = abs( (2*static_cast<SignedElement>(data[idx+szStride]) - 2*static_cast<SignedElement>(data[idx-szStride])) 
			+ (static_cast<SignedElement>(data[idx + 1 + szStride]) - static_cast<SignedElement>(data[idx +1 - szStride]))
			+ (static_cast<SignedElement>(data[idx - 1 + szStride]) - static_cast<SignedElement>(data[idx -1 - szStride])) );
		TElement result = val1 + val2 + val3 + val4 + val5 +val6;
		return result > threshold ? result : 0;
	}

	TElement threshold;
	int3 radius;
};

template< typename TElement >
struct LocalMinima3DFtor
{
	LocalMinima3DFtor( TElement aThreshold ): radius( make_int3( 1, 1, 1 ) ), mThreshold( aThreshold )
	{}

	__device__ uint8
	operator()( TElement data[], uint idx, uint syStride, uint szStride )
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
	operator()( TElement data[], uint idx, uint syStride, uint szStride )
	{
		TElement value = data[idx];
		for ( int i = idx-1; i <= idx+1; ++i ) {
			for ( int j = i-syStride; j <= i+syStride; j+=syStride ) {
				for ( int k = j-szStride; k <= j+szStride; k+=szStride ) {
					value = min( data[k], value );
				}
			}
		}
		uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		uint32 tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
		uint32 ridx = blockId * blockDim.x * blockDim.y + tid;

		return ( value >= data[idx] && data[idx] < mThreshold ) ? ridx + 1: 0;
	}
	int3 radius;
	TElement mThreshold;
};

template< typename TElement >
struct RegionBorderDetection3DFtor
{
	typedef typename TypeTraits< TElement >::SignedClosestType SignedElement;

	RegionBorderDetection3DFtor(): radius( make_int3( 1, 1, 1 ) )
	{}

	__device__ uint8
	operator()( TElement data[], uint idx, uint syStride, uint szStride )
	{
		TElement val = data[idx];
		if ( val != data[idx-1] || val != data[idx-syStride] || val != data[idx-szStride] ) {
			return 255;
		} else {
			return 0;
		}
	}

	int3 radius;
};


__device__ int lutUpdated;

#define min_valid(a, b) (a < b ? a == 0 ? b : a : b == 0 ? a : b)
__device__ uint32
ValidMin( uint32 data[], uint idx, uint syStride, uint szStride )
{
	uint32 value1 = min_valid( data[idx-1], data[idx+1] );
	uint32 value2 = min_valid( data[idx-syStride], data[idx+syStride] );
	uint32 value3 = min_valid( data[idx-szStride], data[idx+szStride] );
	uint32 value = min_valid( value1, value2 );
	return min_valid( value, value3 );
}

__global__ void 
CopyMask( Buffer3D< uint8 > inBuffer, Buffer3D< uint32 > outBuffer )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < inBuffer.mLength ) {
		outBuffer.mData[idx] = inBuffer.mData[idx]!=0 ? idx+1 : 0;
	}
}

__global__ void 
InitLut( Buffer3D< uint32 > outBuffer, Buffer1D< uint32 > lut )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < outBuffer.mLength ) {
		lut.mData[idx] = outBuffer.mData[idx];// = outBuffer.mData[idx] != 0 ? idx+1 : 0;
	}
}

//Group equivalence classes
__global__ void 
UpdateLut( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;
	uint32 label, ref;
	if ( idx < buffer.mLength ) {
		label = buffer.mData[idx];

		if (label == idx+1) {		
			ref = label-1;
			label = lut.mData[idx];
			while (ref != label-1) {
				ref = label-1;
				label = lut.mData[ref];
			}
			lut.mData[idx] = label;
		}
	}
}

__global__ void 
UpdateLabels( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < buffer.mLength ) {
		uint label = buffer.mData[idx];
		if ( label > 0 ) {
			buffer.mData[idx] = lut.mData[label-1];
		}
	}
}

__global__ void 
ScanImage( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut, int3 blockResolution )
{
	__shared__ uint32 data[MAX_SHARED_MEMORY];
	
	int3 radius = make_int3(1,1,1);
	uint syStride = blockDim.x+2*radius.x;
	uint szStride = (blockDim.x+2*radius.x) * (blockDim.y+2*radius.y);

	uint3 size = buffer.mSize;
	int3 strides = buffer.mStrides;
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int3 blockCoordinates = GetBlockCoordinates ( blockResolution, blockId );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	int3 coordinates = blockOrigin;
	//uint tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
	coordinates.x += threadIdx.x;
	coordinates.y += threadIdx.y;
	coordinates.z += threadIdx.z;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), make_int3( size.x, size.y, size.z ) );
	
	int idx = coordinates.x * strides.x + coordinates.y * strides.y + coordinates.z * strides.z;
	uint sidx = (threadIdx.y+radius.y) * syStride + (threadIdx.z+radius.z) * szStride + threadIdx.x + radius.x;
	data[sidx] = buffer.mData[ idx ];
	
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
	data[sIdx.y*syStride + sIdx.z*szStride + sIdx.x] = buffer.mData[ mCoordinates.x * strides.x + mCoordinates.y * strides.y + mCoordinates.z * strides.z ];

	__syncthreads();

	if( !projected ) {
		uint32 current = data[sidx];
		if ( current != 0 ) {
			uint32 minLabel = ValidMin( data, sidx, syStride, szStride );
			if ( minLabel < current && minLabel != 0) {
				lut.mData[current-1] = minLabel < lut.mData[current-1] ? minLabel : lut.mData[current-1];
				lutUpdated = 1;
			}
		}
	}
}


void
ConnectedComponentLabeling3DNoAllocation( Buffer3D< uint32 > outBuffer, Buffer1D< uint32 > lut )
{
	LOG_CONT( "CCL started ... " );
	int lutUpdated = 0;
	int3 radius = make_int3( 1, 1, 1 );


	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (outBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	dim3 blockSize3D( 8, 8, 8 );
	int3 blockResolution3D = GetBlockResolution( outBuffer.mSize, blockSize3D, make_int3(0,0,0) );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

	CheckCudaErrorState( "Before InitLut()" );
	InitLut<<< gridSize1D, blockSize1D >>>( outBuffer, lut );
	CheckCudaErrorState( "After InitLut()" );
	cudaThreadSynchronize();
	ScanImage<<< gridSize3D, blockSize3D >>>( 
					outBuffer, 
					lut,
					blockResolution3D
					);
	cudaThreadSynchronize();

	CheckCudaErrorState( "Before iterations" );
	cudaMemcpyFromSymbol( &lutUpdated, "lutUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
	while (lutUpdated != 0) {
		LOG( ">" );
		CheckCudaErrorState( "Begin iteration" );
                cudaMemcpyToSymbol( "lutUpdated", &(lutUpdated = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

		UpdateLut<<< gridSize1D, blockSize1D >>>( outBuffer, lut );
		UpdateLabels<<< gridSize1D, blockSize1D >>>( outBuffer, lut );

		cudaThreadSynchronize();
		ScanImage<<< gridSize3D, blockSize3D >>>( 
					outBuffer, 
					lut,
					blockResolution3D
					);
		CheckCudaErrorState( "After ScanImage" );
		cudaThreadSynchronize();
		CheckCudaErrorState( "cudaMemcpyFromSymbol" );
		cudaMemcpyFromSymbol( &lutUpdated, "lutUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
		cudaThreadSynchronize();
		CheckCudaErrorState( "End of iteration" );
	}
	cudaThreadSynchronize();
	CheckCudaErrorState( "Synchronization after CCL" );
	LOG( "Done" );
}


void
ConnectedComponentLabeling3D( M4D::Imaging::MaskRegion3D input, M4D::Imaging::ImageRegion< uint32, 3 > output )
{
	Buffer3D< uint8 > inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	Buffer3D< uint32 > outBuffer = CudaBuffer3DFromImageRegion( output );
	int3 radius = make_int3( 1, 1, 1 );


	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (inBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	dim3 blockSize3D( 8, 8, 8 );
	int3 blockResolution3D = GetBlockResolution( inBuffer.mSize, blockSize3D, make_int3(0,0,0) );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

	M4D::Common::Clock clock;

	CheckCudaErrorState( "Before execution" );
	CopyMask<<< gridSize1D, blockSize1D >>>( inBuffer, outBuffer );
	CheckCudaErrorState( "After CopyMask()" );
	cudaFree( inBuffer.mData );

	Buffer1D< uint32 > lut = CudaAllocateBuffer<uint32>( outBuffer.mLength ); 

	ConnectedComponentLabeling3DNoAllocation( outBuffer, lut );

	/*InitLut<<< gridSize1D, blockSize1D >>>( outBuffer, lut );
	CheckCudaErrorState( "After InitLut()" );
	ScanImage<<< gridSize3D, blockSize3D >>>( 
					outBuffer, 
					lut,
					blockResolution3D
					);

	CheckCudaErrorState( "Before iterations" );
	cudaMemcpyFromSymbol( &lutUpdated, "lutUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
	while (lutUpdated != 0) {
		LOG( ">" );
                cudaMemcpyToSymbol( "lutUpdated", &(lutUpdated = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

		UpdateLut<<< gridSize1D, blockSize1D >>>( outBuffer, lut );
		UpdateLabels<<< gridSize1D, blockSize1D >>>( outBuffer, lut );

		ScanImage<<< gridSize3D, blockSize3D >>>( 
					outBuffer, 
					lut,
					blockResolution3D
					);
		cudaMemcpyFromSymbol( &lutUpdated, "lutUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
		CheckCudaErrorState( "End of iteration" );
	}
	cudaThreadSynchronize();*/
	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint32), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	cudaFree( outBuffer.mData );
	cudaFree( lut.mData );
}




__device__ int wshedUpdated;

template< typename TEType >
__global__ void 
InitWatershedBuffers( Buffer3D< uint32 > labeledRegionsBuffer, Buffer3D< TEType > tmpBuffer, TEType infinity )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < tmpBuffer.mLength ) {
		tmpBuffer.mData[idx] = labeledRegionsBuffer.mData[idx] == 0 ? infinity : 0;
	}
}

template< typename TInEType, typename TTmpEType >
__global__ void 
WShedEvolution( Buffer3D< uint32 > labeledRegionsBuffer, Buffer3D< TInEType > inputBuffer, Buffer3D< TTmpEType > tmpBuffer, int3 blockResolution, TTmpEType infinity )
{
	__shared__ uint32 labels[MAX_SHARED_MEMORY];
	__shared__ TTmpEType tmpValues[MAX_SHARED_MEMORY];
	
	int3 radius = make_int3(1,1,1);
	uint syStride = blockDim.x+2*radius.x;
	uint szStride = (blockDim.x+2*radius.x) * (blockDim.y+2*radius.y);

	uint3 size = inputBuffer.mSize;
	int3 strides = inputBuffer.mStrides;
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int3 blockCoordinates = GetBlockCoordinates ( blockResolution, blockId );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	int3 coordinates = blockOrigin;
	//uint tid = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
	coordinates.x += threadIdx.x;
	coordinates.y += threadIdx.y;
	coordinates.z += threadIdx.z;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), make_int3( size.x, size.y, size.z ) );
	
	int idx = coordinates.x * strides.x + coordinates.y * strides.y + coordinates.z * strides.z;
	uint sidx = (threadIdx.y+radius.y) * syStride + (threadIdx.z+radius.z) * szStride + threadIdx.x + radius.x;
	labels[sidx] = labeledRegionsBuffer.mData[ idx ];
	tmpValues[sidx] = tmpBuffer.mData[ idx ];
	
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

	labels[sIdx.y*syStride + sIdx.z*szStride + sIdx.x] = labeledRegionsBuffer.mData[ mCoordinates.x * strides.x + mCoordinates.y * strides.y + mCoordinates.z * strides.z ];
	tmpValues[sIdx.y*syStride + sIdx.z*szStride + sIdx.x] = tmpBuffer.mData[ mCoordinates.x * strides.x + mCoordinates.y * strides.y + mCoordinates.z * strides.z ];

	__syncthreads();

	if( !projected ) {
		int minIdx = -1;
		TInEType value = inputBuffer.mData[ idx ];
		TTmpEType minVal = max( tmpValues[ sidx ] - value,TTmpEType(0) );
		for ( int i = sidx-1; i <= sidx+1; ++i ) {
			for ( int j = i-syStride; j <= i+syStride; j+=syStride ) {
				for ( int k = j-szStride; k <= j+szStride; k+=szStride ) {
					if( tmpValues[ k ] < minVal ) {
						minVal = tmpValues[ k ];
						minIdx = k;
					}
				}
			}
		}
		if( minIdx != -1 ) {
			labeledRegionsBuffer.mData[ idx ] = labels[ minIdx ];
			tmpBuffer.mData[ idx ] = tmpValues[minIdx] + value;
			wshedUpdated = 1;
		}
		/*
		for( unsigned it = 0; it < 2; ++it ) {
			TTmpEType minVal = max( tmpValues[ sidx ] - value,TTmpEType(0) );
			for ( int i = sidx-1; i <= sidx+1; ++i ) {
				for ( int j = i-syStride; j <= i+syStride; j+=syStride ) {
					for ( int k = j-szStride; k <= j+szStride; k+=szStride ) {
						if( tmpValues[ k ] < minVal ) {
							minVal = tmpValues[ k ];
							minIdx = k;
						}
					}
				}
			}
			if( minIdx != -1 ) {
				labels[ sidx ] = labels[ minIdx ];
				tmpValues[ sidx ] = tmpValues[minIdx] + value;
				wshedUpdated = 1;
			}
		}
		labeledRegionsBuffer.mData[ idx ] = labels[ sidx ];
		tmpBuffer.mData[ idx ] = tmpValues[sidx];
		*/		
	}
}

/*#include "Imaging/Image.h"
#include "Imaging/ImageFactory.h"
*/

template< typename TEType >
void
WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput )
{
	typedef typename TypeTraits< TEType >::SignedClosestType SignedElement;
	int wshedUpdated = 1;
	Buffer3D< uint32 > labeledRegionsBuffer = CudaBuffer3DFromImageRegionCopy( aLabeledMarkerRegions );
	Buffer3D< TEType > inputBuffer = CudaBuffer3DFromImageRegionCopy( aInput );
	Buffer3D< SignedElement > tmpBuffer = CudaPrepareBuffer<SignedElement>( aInput.GetSize() );
	int3 radius = make_int3( 1, 1, 1 );


	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (inputBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	dim3 blockSize3D( 8, 8, 8 );
	int3 blockResolution3D = GetBlockResolution( inputBuffer.mSize, blockSize3D, make_int3(0,0,0) );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

	M4D::Common::Clock clock;
	D_PRINT( "InitWatershedBuffers()" );
	InitWatershedBuffers<<< gridSize1D, blockSize1D >>>( labeledRegionsBuffer, tmpBuffer, TypeTraits<SignedElement>::Max );

	unsigned i = 0;
	while (wshedUpdated != 0 && i < 51) {
		cudaMemcpyToSymbol( "wshedUpdated", &(wshedUpdated = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

		//D_PRINT( "WShedEvolution()" );
		WShedEvolution<<< gridSize3D, blockSize3D >>>( 
					labeledRegionsBuffer,
				       	inputBuffer,	
					tmpBuffer,
					blockResolution3D, 
					TypeTraits<SignedElement>::Max
					);

		cudaMemcpyFromSymbol( &wshedUpdated, "wshedUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
		++i;
	}

	cudaThreadSynchronize();
	D_PRINT( "Computations took " << clock.SecondsPassed() << " and " << i << " iterations" )

	cudaMemcpy(aOutput.GetPointer(), labeledRegionsBuffer.mData, labeledRegionsBuffer.mLength * sizeof(uint32), cudaMemcpyDeviceToHost );
	cudaFree( labeledRegionsBuffer.mData );
	cudaFree( inputBuffer.mData );


	/*typename M4D::Imaging::Image< SignedElement, 3 >::Ptr tmpDebugImage = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< SignedElement, 3 >( aLabeledMarkerRegions.GetMinimum(), aLabeledMarkerRegions.GetMaximum(), aLabeledMarkerRegions.GetElementExtents() );
	cudaMemcpy(tmpDebugImage->GetRegion().GetPointer(), tmpBuffer.mData, labeledRegionsBuffer.mLength * sizeof(SignedElement), cudaMemcpyDeviceToHost );
	M4D::Imaging::ImageFactory::DumpImage( "Intermediate.dump", *tmpDebugImage );
*/
	cudaFree( tmpBuffer.mData );
}

template< typename RegionType >
void
Sobel3D( RegionType input, RegionType output, typename RegionType::ElementType threshold )
{
	typedef typename RegionType::ElementType TElement;
	typedef Buffer3D< TElement > Buffer;

	Buffer inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	Buffer outBuffer = CudaBuffer3DFromImageRegion( output );

	SobelFilter3DFtor< TElement > filter( threshold );
	//int3 radius = filter.radius;

	dim3 blockSize( 8, 8, 8 );
	int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, make_int3(0,0,0) );
	dim3 gridSize( blockResolution.x * blockResolution.y, blockResolution.z, 1 );

	M4D::Common::Clock clock;
	CheckCudaErrorState( "Before kernel execution" );
	FilterKernel3D< TElement, TElement, SobelFilter3DFtor< TElement > >
		<<< gridSize, blockSize >>>( 
					inBuffer, 
					outBuffer, 
					blockResolution,
					filter
					);
	cudaThreadSynchronize();
	CheckCudaErrorState( "After kernel execution" );
	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(TElement), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	cudaFree( inBuffer.mData );
	cudaFree( outBuffer.mData );
	CheckCudaErrorState( "Free memory" );
}

template< typename RegionType >
void
RegionBorderDetection3D( RegionType input, M4D::Imaging::MaskRegion3D output )
{
	typedef typename RegionType::ElementType TElement;
	typedef Buffer3D< TElement > Buffer;
	typedef Buffer3D< uint8 > MaskBuffer;

	Buffer inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	MaskBuffer outBuffer = CudaBuffer3DFromImageRegion( output );

	RegionBorderDetection3DFtor< TElement > filter;
	//int3 radius = filter.radius;

	dim3 blockSize( 8, 8, 8 );
	int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, make_int3(0,0,0) );
	dim3 gridSize( blockResolution.x * blockResolution.y, blockResolution.z, 1 );

	M4D::Common::Clock clock;
	CheckCudaErrorState( "Before kernel execution" );
	FilterKernel3D< TElement, uint8, RegionBorderDetection3DFtor< TElement > >
		<<< gridSize, blockSize >>>( 
					inBuffer, 
					outBuffer, 
					blockResolution,
					filter
					);
	cudaThreadSynchronize();
	CheckCudaErrorState( "After kernel execution" );
	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint8), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	cudaFree( inBuffer.mData );
	cudaFree( outBuffer.mData );
	CheckCudaErrorState( "Free memory" );
}

template< typename RegionType >
void
LocalMinima3D( RegionType input, M4D::Imaging::MaskRegion3D output, typename RegionType::ElementType aThreshold )
{
	typedef typename RegionType::ElementType TElement;
	
	Buffer3D< TElement > inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	Buffer3D< uint8 > outBuffer = CudaBuffer3DFromImageRegion( output );

	LocalMinima3DFtor< TElement > filter( aThreshold );
	//int3 radius = filter.radius;

	dim3 blockSize( 8, 8, 8 );
	int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, make_int3(0,0,0) );
	dim3 gridSize( blockResolution.x * blockResolution.y, blockResolution.z, 1 );

	M4D::Common::Clock clock;
	CheckCudaErrorState( "Before kernel execution" );
	FilterKernel3D< TElement, uint8, LocalMinima3DFtor< TElement > >
		<<< gridSize, blockSize >>>( 
					inBuffer, 
					outBuffer, 
					blockResolution,
					filter
					);
	cudaThreadSynchronize();
	CheckCudaErrorState( "After kernel execution" );
	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint8), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	cudaFree( inBuffer.mData );
	cudaFree( outBuffer.mData );
	CheckCudaErrorState( "Free memory" );
}

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

template< typename RegionType >
void
LocalMinimaRegions3D( RegionType input, M4D::Imaging::ImageRegion< uint32, 3 > output, typename RegionType::ElementType aThreshold )
{
	M4D::Imaging::Mask3D::Ptr maskImage = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< uint8, 3 >( input.GetMinimum(), input.GetMaximum(), input.GetElementExtents() );
	
	M4D::Imaging::MaskRegion3D mask = maskImage->GetRegion();
	
	ASSERT( mask.GetSize() == input.GetSize() );
	ASSERT( output.GetSize() == input.GetSize() );
	//LocalMinima3D( input, maskImage->GetRegion(), aThreshold );

	typedef typename RegionType::ElementType TElement;
		
	Buffer3D< TElement > inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	Buffer3D< uint32 > outBuffer = CudaBuffer3DFromImageRegion( output );
	/*{
		
		Buffer3D< uint8 > outBuffer = CudaBuffer3DFromImageRegion( mask );

		LocalMinima3DFtor< TElement > filter( aThreshold );
		//int3 radius = filter.radius;

		dim3 blockSize( 8, 8, 8 );
		int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, make_int3(0,0,0) );
		dim3 gridSize( blockResolution.x * blockResolution.y, blockResolution.z, 1 );

		M4D::Common::Clock clock;
		CheckCudaErrorState( "Before kernel execution" );
		FilterKernel3D< TElement, uint8, LocalMinima3DFtor< TElement > >
			<<< gridSize, blockSize >>>( 
						inBuffer, 
						outBuffer, 
						blockResolution,
						filter
						);
		cudaThreadSynchronize();
		CheckCudaErrorState( "After kernel execution" );
		D_PRINT( "Computations took " << clock.SecondsPassed() )

		cudaMemcpy(mask.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint8), cudaMemcpyDeviceToHost );
		CheckCudaErrorState( "Copy back" );
		cudaFree( outBuffer.mData );
		CheckCudaErrorState( "Free memory" );
	}*/


	{
		//typedef typename RegionType::ElementType TElement;
		
		//Buffer3D< TElement > inBuffer = CudaBuffer3DFromImageRegionCopy( input );
		//Buffer3D< uint32 > outBuffer = CudaBuffer3DFromImageRegion( output );
		LocalMinimaRegions3DFtor< TElement > filter( aThreshold );

		dim3 blockSize( 8, 8, 8 );
		int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, make_int3(0,0,0) );
		dim3 gridSize( blockResolution.x * blockResolution.y, blockResolution.z, 1 );
		dim3 blockSize1D( 512 );
		dim3 gridSize1D( (inBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

		M4D::Common::Clock clock;
		CheckCudaErrorState( "Before 'LocalMinimaRegions3D' kernel execution" );
		FilterKernel3D< TElement, uint32, LocalMinimaRegions3DFtor< TElement > >
			<<< gridSize, blockSize >>>( 
						inBuffer, 
						outBuffer, 
						blockResolution,
						filter
						);
		cudaThreadSynchronize();
		CheckCudaErrorState( "After 'LocalMinimaRegions3D' kernel execution" );
		//cudaFree( inBuffer.mData );
		//cudaFree( outBuffer.mData );
	}
			
	//ConnectedComponentLabeling3D( maskImage->GetRegion(), output );
	{
		D_PRINT( "ALLOCATE mask buffer" );
		//Buffer3D< uint8 > maskBuffer = CudaBuffer3DFromImageRegionCopy( mask );
		D_PRINT( "ALLOCATE output buffer" );
		//Buffer3D< uint32 > outBuffer = CudaBuffer3DFromImageRegion( output );

		dim3 blockSize1D( 512 );
		dim3 gridSize1D( (inBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
		
		dim3 blockSize3D( 8, 8, 8 );
		int3 blockResolution3D = GetBlockResolution( inBuffer.mSize, blockSize3D, make_int3(0,0,0) );
		dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

		M4D::Common::Clock clock;

		/*CheckCudaErrorState( "Before execution" );
		CopyMask<<< gridSize1D, blockSize1D >>>( maskBuffer, outBuffer );
		CheckCudaErrorState( "After CopyMask()" );

		cudaFree( maskBuffer.mData );*/

		D_PRINT( "ALLOCATE LUT buffer" );
		Buffer1D< uint32 > lut = CudaAllocateBuffer<uint32>( outBuffer.mLength ); 

		ConnectedComponentLabeling3DNoAllocation( outBuffer, lut );

		CheckCudaErrorState( "Before ConsolidationScanImage()" );
		ConsolidationScanImage<<< gridSize3D, blockSize3D >>>( inBuffer, outBuffer, lut, blockResolution3D );

		cudaThreadSynchronize();
		CheckCudaErrorState( "After ConsolidationScanImage()" );
		D_PRINT( "Computations took " << clock.SecondsPassed() )

		cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint32), cudaMemcpyDeviceToHost );
		CheckCudaErrorState( "Copy back" );
		cudaFree( outBuffer.mData );
		cudaFree( lut.mData );

	}
	cudaFree( inBuffer.mData );

}

/*template< typename RegionType >
void
LocalMinimaRegions3D( RegionType input, M4D::Imaging::ImageRegion< uint32, 3 > output, typename RegionType::ElementType aThreshold )
{
	typedef typename RegionType::ElementType TElement;
	
	Buffer3D< TElement > inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	Buffer3D< uint32 > outBuffer = CudaBuffer3DFromImageRegion( output );

	LocalMinimaRegions3DFtor< TElement > filter( aThreshold );
	//int3 radius = filter.radius;

	dim3 blockSize( 8, 8, 8 );
	int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, make_int3(0,0,0) );
	dim3 gridSize( blockResolution.x * blockResolution.y, blockResolution.z, 1 );
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (inBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	M4D::Common::Clock clock;
	CheckCudaErrorState( "Before 'LocalMinimaRegions3D' kernel execution" );
	FilterKernel3D< TElement, uint32, LocalMinimaRegions3DFtor< TElement > >
		<<< gridSize, blockSize >>>( 
					inBuffer, 
					outBuffer, 
					blockResolution,
					filter
					);
	cudaThreadSynchronize();
	CheckCudaErrorState( "After 'LocalMinimaRegions3D' kernel execution" );


	Buffer1D< uint32 > lut = CudaAllocateBuffer<uint32>( outBuffer.mLength ); 
	ConnectedComponentLabeling3DNoAllocation( outBuffer, lut );
	CheckCudaErrorState( "After CCL kernel execution" );
	//ConsolidationScanImage<<< gridSize, blockSize >>>( inBuffer, outBuffer, lut, blockResolution );

	//UpdateLabels<<< gridSize1D, blockSize1D >>>( outBuffer, lut );


	cudaThreadSynchronize();

	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint8), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	cudaFree( inBuffer.mData );
	cudaFree( outBuffer.mData );
	cudaFree( lut.mData );
	CheckCudaErrorState( "Free memory" );
}*/




template void Sobel3D( M4D::Imaging::ImageRegion< int8, 3 > input, M4D::Imaging::ImageRegion< int8, 3 > output, int8 threshold );
template void Sobel3D( M4D::Imaging::ImageRegion< uint8, 3 > input, M4D::Imaging::ImageRegion< uint8, 3 > output, uint8 threshold );
template void Sobel3D( M4D::Imaging::ImageRegion< int16, 3 > input, M4D::Imaging::ImageRegion< int16, 3 > output, int16 threshold );
template void Sobel3D( M4D::Imaging::ImageRegion< uint16, 3 > input, M4D::Imaging::ImageRegion< uint16, 3 > output, uint16 threshold );
template void Sobel3D( M4D::Imaging::ImageRegion< int32, 3 > input, M4D::Imaging::ImageRegion< int32, 3 > output, int32 threshold );
template void Sobel3D( M4D::Imaging::ImageRegion< uint32, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, uint32 threshold );
template void Sobel3D( M4D::Imaging::ImageRegion< int64, 3 > input, M4D::Imaging::ImageRegion< int64, 3 > output, int64 threshold );
template void Sobel3D( M4D::Imaging::ImageRegion< uint64, 3 > input, M4D::Imaging::ImageRegion< uint64, 3 > output, uint64 threshold );
template void Sobel3D( M4D::Imaging::ImageRegion< float, 3 > input, M4D::Imaging::ImageRegion< float, 3 > output, float threshold );
template void Sobel3D( M4D::Imaging::ImageRegion< double, 3 > input, M4D::Imaging::ImageRegion< double, 3 > output, double threshold );

template void LocalMinima3D( M4D::Imaging::ImageRegion< int8, 3 > input, M4D::Imaging::MaskRegion3D output, int8 aThreshold );
template void LocalMinima3D( M4D::Imaging::ImageRegion< uint8, 3 > input, M4D::Imaging::MaskRegion3D output, uint8 aThreshold );
template void LocalMinima3D( M4D::Imaging::ImageRegion< int16, 3 > input, M4D::Imaging::MaskRegion3D output, int16 aThreshold );
template void LocalMinima3D( M4D::Imaging::ImageRegion< uint16, 3 > input, M4D::Imaging::MaskRegion3D output, uint16 aThreshold );
template void LocalMinima3D( M4D::Imaging::ImageRegion< int32, 3 > input, M4D::Imaging::MaskRegion3D output, int32 aThreshold );
template void LocalMinima3D( M4D::Imaging::ImageRegion< uint32, 3 > input, M4D::Imaging::MaskRegion3D output, uint32 aThreshold );
template void LocalMinima3D( M4D::Imaging::ImageRegion< int64, 3 > input, M4D::Imaging::MaskRegion3D output, int64 aThreshold );
template void LocalMinima3D( M4D::Imaging::ImageRegion< uint64, 3 > input, M4D::Imaging::MaskRegion3D output, uint64 aThreshold );
template void LocalMinima3D( M4D::Imaging::ImageRegion< float, 3 > input, M4D::Imaging::MaskRegion3D output, float aThreshold );
template void LocalMinima3D( M4D::Imaging::ImageRegion< double, 3 > input, M4D::Imaging::MaskRegion3D output, double aThreshold );

template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< int8, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, int8 aThreshold );
template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< uint8, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, uint8 aThreshold );
template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< int16, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, int16 aThreshold );
template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< uint16, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, uint16 aThreshold );
template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< int32, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, int32 aThreshold );
template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< uint32, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, uint32 aThreshold );
template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< int64, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, int64 aThreshold );
template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< uint64, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, uint64 aThreshold );
template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< float, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, float aThreshold );
template void LocalMinimaRegions3D( M4D::Imaging::ImageRegion< double, 3 > input, M4D::Imaging::ImageRegion< uint32, 3 > output, double aThreshold );

template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< int8, 3 > input, M4D::Imaging::MaskRegion3D output );
template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< uint8, 3 > input, M4D::Imaging::MaskRegion3D output );
template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< int16, 3 > input, M4D::Imaging::MaskRegion3D output );
template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< uint16, 3 > input, M4D::Imaging::MaskRegion3D output );
template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< int32, 3 > input, M4D::Imaging::MaskRegion3D output );
template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< uint32, 3 > input, M4D::Imaging::MaskRegion3D output );
template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< int64, 3 > input, M4D::Imaging::MaskRegion3D output );
template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< uint64, 3 > input, M4D::Imaging::MaskRegion3D output );
template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< float, 3 > input, M4D::Imaging::MaskRegion3D output );
template void RegionBorderDetection3D( M4D::Imaging::ImageRegion< double, 3 > input, M4D::Imaging::MaskRegion3D output );

template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int8, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint8, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int16, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint16, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int32, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint32, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int64, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint64, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< float, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< double, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
