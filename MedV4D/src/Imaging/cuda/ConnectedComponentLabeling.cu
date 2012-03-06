#include "MedV4D/Imaging/cuda/detail/ConnectedComponentLabeling.cuh"

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
	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint32), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	cudaFree( outBuffer.mData );
	cudaFree( lut.mData );
}