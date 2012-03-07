#include "MedV4D/Imaging/cuda/detail/LocalMinimaDetection.cuh"
#include "MedV4D/Imaging/cuda/detail/ConnectedComponentLabeling.cuh"
#include "MedV4D/Imaging/Image.h"
#include "MedV4D/Imaging/ImageRegion.h"
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>

__global__ void 
MarkUsedIds( Buffer3D< uint32 > outBuffer, Buffer1D< uint32 > lut )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < outBuffer.mLength ) {
		lut.mData[ outBuffer.mData[idx] ] = 1;
	}
}

__global__ void 
UpdateLabelsFromScan( Buffer3D< uint32 > buffer, Buffer1D< uint32 > lut )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < buffer.mLength ) {
		uint label = buffer.mData[idx];
		buffer.mData[idx] = lut.mData[label];
	}
}



//TODO - handle in a better way
void
ConnectedComponentLabeling3DNoAllocation( Buffer3D< uint32 > outBuffer, Buffer1D< uint32 > lut );

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


template< typename RegionType >
void
LocalMinimaRegions3D( RegionType input, M4D::Imaging::ImageRegion< uint32, 3 > output, typename RegionType::ElementType aThreshold )
{
	typedef typename RegionType::ElementType TElement;
	
	ASSERT( output.GetSize() == input.GetSize() );
	//LocalMinima3D( input, maskImage->GetRegion(), aThreshold );

	D_PRINT( "ALLOCATE input buffer" );
	Buffer3D< TElement > inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	D_PRINT( "ALLOCATE output buffer" );
	Buffer3D< uint32 > outBuffer = CudaBuffer3DFromImageRegion( output );
	D_PRINT( "ALLOCATE LUT buffer" );
	//Buffer1D< uint32 > lut = CudaAllocateBuffer<uint32>( outBuffer.mLength +1 ); //+1 is due to shift of labels after parallelScan
	thrust::device_vector<uint32> lutVector(outBuffer.mLength +1);
	Buffer1D< uint32 > lut = cudaBufferFromThrustDeviceVector( lutVector );

	LocalMinimaRegions3DFtor< TElement > filter( aThreshold );

	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (inBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	
	dim3 blockSize3D( 8, 8, 8 );
	int3 blockResolution3D = GetBlockResolution( inBuffer.mSize, blockSize3D, make_int3(0,0,0) );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

	M4D::Common::Clock clock;
	CheckCudaErrorState( "Before 'LocalMinimaRegions3D' kernel execution" );
	FilterKernel3D< TElement, uint32, LocalMinimaRegions3DFtor< TElement > >
		<<< gridSize3D, blockSize3D >>>( 
					inBuffer, 
					outBuffer, 
					blockResolution3D,
					filter
					);
	cudaThreadSynchronize();
	CheckCudaErrorState( "After 'LocalMinimaRegions3D' kernel execution" );




	cudaThreadSynchronize();

	ConnectedComponentLabeling3DNoAllocation( outBuffer, lut );

	CheckCudaErrorState( "Before ConsolidationScanImage()" );
	ConsolidationScanImage<<< gridSize3D, blockSize3D >>>( inBuffer, outBuffer, lut, blockResolution3D );

	cudaThreadSynchronize();
	CheckCudaErrorState( "After ConsolidationScanImage()" );

	//cudaMemset( lut.mData, 0, sizeof( uint32 ) * lut.mLength );
	thrust::fill( lutVector.begin(), lutVector.end(), 0 );

	MarkUsedIds<<< gridSize1D, blockSize1D >>>( outBuffer, lut );
	CheckCudaErrorState( "After MarkUsedIds()" );
	//parallelScan< uint32, Sum< uint32 >, 512 >( lut, lut, Sum< uint32 >() );
	thrust::exclusive_scan( lutVector.begin(), lutVector.end(), lutVector.begin() );
	CheckCudaErrorState( "After parallelScan()" );
	UpdateLabelsFromScan<<< gridSize1D, blockSize1D >>>( outBuffer, lut );
	CheckCudaErrorState( "After UpdateLabelsFromScan()" );
	cudaThreadSynchronize();
	CheckCudaErrorState( "After relabeling" );

	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint32), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );

	cudaFree( outBuffer.mData );
	//cudaFree( lut.mData );
	cudaFree( inBuffer.mData );

}

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