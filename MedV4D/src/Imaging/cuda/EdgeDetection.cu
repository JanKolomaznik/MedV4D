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
	//cudaFree( inBuffer.mData );
	//cudaFree( outBuffer.mData );
	CheckCudaErrorState( "Free memory" );
}

#define DECLARE_TEMPLATE_INSTANCE template void Sobel3D( M4D::Imaging::ImageRegion< TTYPE, 3 > input, M4D::Imaging::ImageRegion< TTYPE, 3 > output, TTYPE threshold );
#include "MedV4D/Common/DeclareTemplateNumericInstances.h"

