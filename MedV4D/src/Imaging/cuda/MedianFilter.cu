#include "MedV4D/Imaging/cuda/detail/MedianFilter.cuh"


template< typename RegionType >
void
median3D( RegionType input, RegionType output, size_t aRadius )
{
	typedef typename RegionType::ElementType TElement;
	typedef Buffer3D< TElement > Buffer;

	size_t iterations = 3;

	Buffer inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	Buffer outBuffer = CudaBuffer3DFromImageRegion( output );

	MedianFilter3DFtor< TElement > filter;

	dim3 blockSize( 8, 8, 8 );
	int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, make_int3(0,0,0) );
	dim3 gridSize( blockResolution.x * blockResolution.y, blockResolution.z, 1 );

	M4D::Common::Clock clock;
	for( size_t i = 0; i < iterations; ++i ) {
		CheckCudaErrorState( "Before kernel execution" );
		FilterKernel3D< TElement, TElement, MedianFilter3DFtor< TElement > >
			<<< gridSize, blockSize >>>( 
						inBuffer, 
						outBuffer, 
						blockResolution,
						filter
						);
		cudaThreadSynchronize();
		CheckCudaErrorState( "After kernel execution" );
		inBuffer.swap( outBuffer );
	}
	LOG( "median3D computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), inBuffer.mData, inBuffer.mLength * sizeof(TElement), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	//cudaFree( inBuffer.mData );
	//cudaFree( outBuffer.mData );
	//CheckCudaErrorState( "Free memory" );
}


#define DECLARE_TEMPLATE_INSTANCE template void median3D( M4D::Imaging::ImageRegion< TTYPE, 3 > aInput, M4D::Imaging::ImageRegion< TTYPE, 3 > aOutput, size_t aRadius );
#include "MedV4D/Common/DeclareTemplateNumericInstances.h"


