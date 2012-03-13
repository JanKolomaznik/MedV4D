#include "MedV4D/Imaging/cuda/detail/EdgeDetection.cuh"


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
	LOG( "Sobel3D computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(TElement), cudaMemcpyDeviceToHost );
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