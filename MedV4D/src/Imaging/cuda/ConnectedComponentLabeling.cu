#include "MedV4D/Imaging/cuda/detail/ConnectedComponentLabeling.cuh"

__device__ int lutUpdated;

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

	if ( idx < buffer.mLength )	{
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
	
	const int cBlockDim = 8;
	const int cRadius = 1;
	const uint syStride = cBlockDim+2*cRadius;
	const uint szStride = (cBlockDim+2*cRadius) * (cBlockDim+2*cRadius);

	uint sidx = (threadIdx.y+cRadius) * syStride + (threadIdx.z+cRadius) * szStride + threadIdx.x + cRadius;
	int3 size = toInt3( buffer.mSize );
	int3 blockCoordinates = GetBlockCoordinates( blockResolution, __mul24(blockIdx.y, gridDim.x) + blockIdx.x );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	
	
	int3 coordinates = blockOrigin + threadIdx;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), size );
	int idx = IdxFromCoordStrides( coordinates, buffer.mStrides );

	FillSharedMemory3D_8x8x8< uint32, cRadius, syStride, szStride >( data, sidx, buffer.mData, buffer.mStrides, size, blockOrigin, coordinates, idx );

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
ConnectedComponentLabeling3DNoAllocation( Buffer3D< uint32 > &outBuffer, Buffer1D< uint32 > &lut )
{
	LOG_CONT( "CCL started ... " );
	int lutUpdated = 0;
	int3 radius = make_int3( 1, 1, 1 );


	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (outBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	dim3 blockSize3D( 8, 8, 8 );
	int3 blockResolution3D = GetBlockResolution( outBuffer.mSize, blockSize3D, make_int3(0,0,0) );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

        cudaMemcpyToSymbol( "lutUpdated", &(lutUpdated = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

	CheckCudaErrorState( "Before InitLut()" );
	InitLut<<< gridSize1D, blockSize1D >>>( outBuffer, lut );
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
		CheckCudaErrorState( "Begin iteration" );
                cudaMemcpyToSymbol( "lutUpdated", &(lutUpdated = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

		UpdateLut<<< gridSize1D, blockSize1D >>>( outBuffer, lut );
		UpdateLabels<<< gridSize1D, blockSize1D >>>( outBuffer, lut );

		CheckCudaErrorState( "After LUT and labels update" );
		ScanImage<<< gridSize3D, blockSize3D >>>( 
					outBuffer, 
					lut,
					blockResolution3D
					);
		CheckCudaErrorState( "After ScanImage" );
		cudaMemcpyFromSymbol( &lutUpdated, "lutUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
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
	LOG( "ConnectedComponentLabeling3D computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint32), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	cudaFree( outBuffer.mData );
	cudaFree( lut.mData );
}