#include "CudaUtils.cuh"
#include "Imaging/ImageRegion.h"


template< typename TElement >
struct SobelFilter3DFtor
{
	typedef typename TypeTraits< TElement >::SignedClosestType SignedElement;

	SobelFilter3DFtor( TElement aThreshold = 0 ): threshold( aThreshold ), radius( make_int3( 1, 1, 1 ) )
	{}

	__device__ TElement
	operator()( TElement data[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE][MAX_BLOCK_SIZE], uint3 idx )
	{
		TElement val1 = abs( (2*static_cast<SignedElement>(data[idx.x+1][idx.y][idx.z]) - 2*static_cast<SignedElement>(data[idx.x-1][idx.y][idx.z])) 
			+ (static_cast<SignedElement>(data[idx.x+1][idx.y+1][idx.z]) - static_cast<SignedElement>(data[idx.x-1][idx.y+1][idx.z]))
			+ (static_cast<SignedElement>(data[idx.x+1][idx.y-1][idx.z]) - static_cast<SignedElement>(data[idx.x-1][idx.y-1][idx.z])) );
		TElement val2 = abs( (2*static_cast<SignedElement>(data[idx.x][idx.y+1][idx.z]) - 2*static_cast<SignedElement>(data[idx.x][idx.y-1][idx.z])) 
			+ (static_cast<SignedElement>(data[idx.x+1][idx.y+1][idx.z]) - static_cast<SignedElement>(data[idx.x+1][idx.y-1][idx.z]))
			+ (static_cast<SignedElement>(data[idx.x-1][idx.y+1][idx.z]) - static_cast<SignedElement>(data[idx.x-1][idx.y-1][idx.z])) );
		TElement val3 = abs( (2*static_cast<SignedElement>(data[idx.x][idx.y][idx.z+1]) - 2*static_cast<SignedElement>(data[idx.x][idx.y][idx.z-1])) 
			+ (static_cast<SignedElement>(data[idx.x][idx.y+1][idx.z+1]) - static_cast<SignedElement>(data[idx.x][idx.y+1][idx.z-1]))
			+ (static_cast<SignedElement>(data[idx.x][idx.y-1][idx.z+1]) - static_cast<SignedElement>(data[idx.x][idx.y-1][idx.z-1])) );
		TElement val4 = abs( (2*static_cast<SignedElement>(data[idx.x][idx.y+1][idx.z]) - 2*static_cast<SignedElement>(data[idx.x][idx.y-1][idx.z])) 
			+ (static_cast<SignedElement>(data[idx.x][idx.y+1][idx.z+1]) - static_cast<SignedElement>(data[idx.x][idx.y-1][idx.z+1]))
			+ (static_cast<SignedElement>(data[idx.x][idx.y+1][idx.z-1]) - static_cast<SignedElement>(data[idx.x][idx.y-1][idx.z-1])) );
		TElement val5 = abs( (2*static_cast<SignedElement>(data[idx.x+1][idx.y][idx.z]) - 2*static_cast<SignedElement>(data[idx.x-1][idx.y][idx.z])) 
			+ (static_cast<SignedElement>(data[idx.x+1][idx.y][idx.z+1]) - static_cast<SignedElement>(data[idx.x-1][idx.y][idx.z+1]))
			+ (static_cast<SignedElement>(data[idx.x+1][idx.y][idx.z-1]) - static_cast<SignedElement>(data[idx.x-1][idx.y][idx.z-1])) );
		TElement val6 = abs( (2*static_cast<SignedElement>(data[idx.x][idx.y][idx.z+1]) - 2*static_cast<SignedElement>(data[idx.x][idx.y][idx.z-1])) 
			+ (static_cast<SignedElement>(data[idx.x+1][idx.y][idx.z+1]) - static_cast<SignedElement>(data[idx.x+1][idx.y][idx.z-1]))
			+ (static_cast<SignedElement>(data[idx.x-1][idx.y][idx.z+1]) - static_cast<SignedElement>(data[idx.x-1][idx.y][idx.z-1])) );
		TElement result = val1 + val2 + val3 + val4 + val5 +val6;
		return result > threshold ? result : 0;
	}

	TElement threshold;
	int3 radius;
};

template< typename TElement >
struct LocalMinima3DFtor
{
	LocalMinima3DFtor(): radius( make_int3( 1, 1, 1 ) )
	{}

	__device__ uint8
	operator()( TElement data[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE][MAX_BLOCK_SIZE], uint3 idx )
	{
		bool res = true;
		for ( int i = idx.x-1; i <= idx.x+1; ++i ) {
			for ( int j = idx.y-1; j <= idx.y+1; ++j ) {
				for ( int k = idx.z-1; k <= idx.z+1; ++k ) {
					res = res && data[i][j][k] >= data[idx.x][idx.y][idx.z];
				}
			}
		}
		return res ? 255 : 0;
	}
	int3 radius;
};


__device__ int lutUpdated;

#define min_valid(a, b) (a < b ? a == 0 ? b : a : b == 0 ? a : b)
__device__ uint32
ValidMin( uint32 data[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE][MAX_BLOCK_SIZE], uint3 idx )
{
	uint32 value1 = min_valid( data[idx.x-1][idx.y][idx.z], data[idx.x-1][idx.y][idx.z] );
	uint32 value2 = min_valid( data[idx.x][idx.y-1][idx.z], data[idx.x][idx.y+1][idx.z] );
	uint32 value3 = min_valid( data[idx.x][idx.y][idx.z-1], data[idx.x][idx.y][idx.z+1] );
	uint32 value = min_valid( value1, value2 );
	return min_valid( value, value3 );
}

__global__ void 
CopyMask( Buffer3D< uint8 > inBuffer, Buffer3D< uint32 > outBuffer )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < inBuffer.mLength ) {
		outBuffer.mData[idx] = inBuffer.mData[idx]>0?1000:0;
	}
}

__global__ void 
InitLut( Buffer3D< uint32 > outBuffer, Buffer1D< uint32 > lut )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < outBuffer.mLength ) {
		lut.mData[idx] = outBuffer.mData[idx] = outBuffer.mData[idx] != 0 ? idx+1 : 0;
	}
}

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
	__shared__ uint32 data[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];

	uint3 size = buffer.mSize;
	int3 strides = buffer.mStrides;
	int3 radius = make_int3( 1, 1, 1 );
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int3 blockSize = make_int3( blockDim.x - 2*radius.x, blockDim.y - 2*radius.y, blockDim.z - 2*radius.z );
	int3 blockCoordinates = GetBlockCoordinates ( blockResolution, blockId );
	int3 coordinates = GetBlockOrigin( blockSize, blockCoordinates );
	coordinates.x += threadIdx.x - radius.x;
	coordinates.y += threadIdx.y - radius.y;
	coordinates.z += threadIdx.z - radius.z;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), make_int3( size.x, size.y, size.z ) );
	
	int idx = coordinates.x * strides.x + coordinates.y * strides.y + coordinates.z * strides.z;
	data[threadIdx.x][threadIdx.y][threadIdx.z] = buffer.mData[ idx ];
	__syncthreads();

	if( !projected &&
		threadIdx.x != 0 && threadIdx.y != 0 && threadIdx.z != 0 &&
		threadIdx.x != blockDim.x - 1 && threadIdx.y != blockDim.y - 1 && threadIdx.z != blockDim.z - 1 
	  ) {
		uint32 current = buffer.mData[idx];
		if ( current != 0 ) {
			uint32 minLabel = ValidMin( data, threadIdx );
			if ( minLabel < current && minLabel != 0) {
				lut.mData[current-1] = minLabel < lut.mData[current-1] ? minLabel : lut.mData[current-1];
				lutUpdated = 1;
			}
		}
	}
}

void
ConnectedComponentLabeling3D( M4D::Imaging::MaskRegion3D input, M4D::Imaging::ImageRegion< uint32, 3 > output )
{
	int lutUpdated = 0;
	Buffer3D< uint8 > inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	Buffer3D< uint32 > outBuffer = CudaBuffer3DFromImageRegion( output );
	int3 radius = make_int3( 1, 1, 1 );


	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (inBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	dim3 blockSize3D( 10, 10, 10 );
	int3 blockResolution3D = GetBlockResolution( inBuffer.mSize, blockSize3D, radius );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

	M4D::Common::Clock clock;

	CheckCudaErrorState( "Before execution" );
	CopyMask<<< gridSize1D, blockSize1D >>>( inBuffer, outBuffer );
	cudaFree( inBuffer.mData );

	Buffer1D< uint32 > lut = CudaAllocateBuffer<uint32>( outBuffer.mLength ); 

	InitLut<<< gridSize1D, blockSize1D >>>( outBuffer, lut );
	ScanImage<<< gridSize3D, blockSize3D >>>( 
					outBuffer, 
					lut,
					blockResolution3D
					);

	CheckCudaErrorState( "Before iterations" );
	cudaMemcpyFromSymbol( &lutUpdated, "lutUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
	while (lutUpdated != 0) {
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
	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint32), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	cudaFree( outBuffer.mData );
	cudaFree( lut.mData );
}

template< typename TEType >
__global__ void 
InitWatershedBuffers( Buffer3D< uint32 > labeledRegionsBuffer, Buffer3D< TEType > tmpBuffer, TEType infinity )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < tmpBuffer.mLength ) {
		tmpBuffer.mData[idx] = labeledRegionsBuffer.mData[idx] != 0 ? infinity : 0;
	}
}

template< typename TInEType, typename TTmpEType >
__global__ void 
WshedEvolution( Buffer3D< uint32 > labeledRegionsBuffer, Buffer3D< TInEType > inputBuffer, Buffer3D< TTmpEType > tmpBuffer, int3 blockResolution )
{
	__shared__ uint32 data[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];

	uint3 size = labeledRegionsBuffer.mSize;
	int3 strides = labeledRegionsBuffer.mStrides;
	int3 radius = make_int3( 1, 1, 1 );
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int3 blockSize = make_int3( blockDim.x - 2*radius.x, blockDim.y - 2*radius.y, blockDim.z - 2*radius.z );
	int3 blockCoordinates = GetBlockCoordinates ( blockResolution, blockId );
	int3 coordinates = GetBlockOrigin( blockSize, blockCoordinates );
	coordinates.x += threadIdx.x - radius.x;
	coordinates.y += threadIdx.y - radius.y;
	coordinates.z += threadIdx.z - radius.z;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), make_int3( size.x, size.y, size.z ) );
	
	int idx = coordinates.x * strides.x + coordinates.y * strides.y + coordinates.z * strides.z;
	data[threadIdx.x][threadIdx.y][threadIdx.z] = labeledRegionsBuffer.mData[ idx ];
	__syncthreads();

	if( !projected &&
		threadIdx.x != 0 && threadIdx.y != 0 && threadIdx.z != 0 &&
		threadIdx.x != blockDim.x - 1 && threadIdx.y != blockDim.y - 1 && threadIdx.z != blockDim.z - 1 
	  ) {
		/*uint32 current = buffer.mData[idx];
		if ( current != 0 ) {
			uint32 minLabel = ValidMin( data, threadIdx );
			if ( minLabel < current && minLabel != 0) {
				lut.mData[current-1] = minLabel < lut.mData[current-1] ? minLabel : lut.mData[current-1];
				lutUpdated = 1;
			}
		}*/
	}
}

template< typename TEType >
void
WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput )
{
	int wshedUpdated = 1;
	Buffer3D< uint32 > labeledRegionsBuffer = CudaBuffer3DFromImageRegionCopy( aLabeledMarkerRegions );
	Buffer3D< TEType > inputBuffer = CudaBuffer3DFromImageRegionCopy( aInput );
	Buffer3D< TEType > tmpBuffer = CudaBuffer3DFromImageRegion( aInput );
	int3 radius = make_int3( 1, 1, 1 );


	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (inputBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	dim3 blockSize3D( 10, 10, 10 );
	int3 blockResolution3D = GetBlockResolution( inputBuffer.mSize, blockSize3D, radius );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

	M4D::Common::Clock clock;
	InitWatershedBuffers<<< gridSize1D, blockSize1D >>>( labeledRegionsBuffer, tmpBuffer, 100000 );

	while (wshedUpdated != 0) {
		cudaMemcpyToSymbol( "wshedUpdated", &(wshedUpdated = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

		WshedEvolution<<< gridSize3D, blockSize3D >>>( 
					labeledRegionsBuffer,
				       	inputBuffer,	
					tmpBuffer,
					blockResolution3D
					);

		cudaMemcpyFromSymbol( &wshedUpdated, "wshedUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
	}


	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaFree( labeledRegionsBuffer.mData );
	cudaFree( inputBuffer.mData );
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
	int3 radius = filter.radius;

	dim3 blockSize( 10, 10, 10 );
	int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, radius );
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
LocalMinima3D( RegionType input, M4D::Imaging::MaskRegion3D output )
{
	typedef typename RegionType::ElementType TElement;
	
	Buffer3D< TElement > inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	Buffer3D< uint8 > outBuffer = CudaBuffer3DFromImageRegion( output );

	LocalMinima3DFtor< TElement > filter;
	int3 radius = filter.radius;

	dim3 blockSize( 10, 10, 10 );
	int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, radius );
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

template void LocalMinima3D( M4D::Imaging::ImageRegion< int8, 3 > input, M4D::Imaging::MaskRegion3D output );
template void LocalMinima3D( M4D::Imaging::ImageRegion< uint8, 3 > input, M4D::Imaging::MaskRegion3D output );
template void LocalMinima3D( M4D::Imaging::ImageRegion< int16, 3 > input, M4D::Imaging::MaskRegion3D output );
template void LocalMinima3D( M4D::Imaging::ImageRegion< uint16, 3 > input, M4D::Imaging::MaskRegion3D output );
template void LocalMinima3D( M4D::Imaging::ImageRegion< int32, 3 > input, M4D::Imaging::MaskRegion3D output );
template void LocalMinima3D( M4D::Imaging::ImageRegion< uint32, 3 > input, M4D::Imaging::MaskRegion3D output );
template void LocalMinima3D( M4D::Imaging::ImageRegion< int64, 3 > input, M4D::Imaging::MaskRegion3D output );
template void LocalMinima3D( M4D::Imaging::ImageRegion< uint64, 3 > input, M4D::Imaging::MaskRegion3D output );
template void LocalMinima3D( M4D::Imaging::ImageRegion< float, 3 > input, M4D::Imaging::MaskRegion3D output );
template void LocalMinima3D( M4D::Imaging::ImageRegion< double, 3 > input, M4D::Imaging::MaskRegion3D output );
