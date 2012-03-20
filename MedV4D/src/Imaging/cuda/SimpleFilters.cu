#include "MedV4D/Imaging/cuda/SimpleFilters.h"
#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>


template< typename TElementType >
struct ThresholdingFtor
{
	TElementType mThreshold;
	TElementType mBelowThreshold;

	__device__ __host__ 
	ThresholdingFtor( TElementType aThreshold, TElementType aBelowThreshold ): mThreshold( aThreshold ), mBelowThreshold( aBelowThreshold )
	{ }

	__device__ __host__ void
	operator()( const TElementType &aIn, uint8 &aOut )const
	{
		if( mThreshold <= mBelowThreshold ) {
			aOut = ( aIn >= mThreshold && aIn <= mBelowThreshold ) ? 255 : 0;
			return;
		}
		
		aOut = ( aIn >= mThreshold || aIn <= mBelowThreshold ) ? 255 : 0;
	}
};

template< typename TRegionType >
void
thresholding3D( TRegionType input,  M4D::Imaging::MaskRegion3D output, typename TRegionType::ElementType aThreshold, typename TRegionType::ElementType aBelowThreshold )
{
	typedef typename TRegionType::ElementType TInElement;
	typedef M4D::Imaging::MaskRegion3D::ElementType TOutElement;

	CheckCudaErrorState( "Before thresholding3D()" );

	Buffer3D< TInElement > inBuffer = CudaBuffer3DFromImageRegionCopy( input );
	Buffer3D< TOutElement > outBuffer = CudaBuffer3DFromImageRegion( output );

	dim3 blockSize( 8, 8, 8 );
	int3 blockResolution = GetBlockResolution( inBuffer.mSize, blockSize, make_int3(0,0,0) );
	dim3 gridSize( blockResolution.x * blockResolution.y, blockResolution.z, 1 );

	M4D::Common::Clock clock;
	FilterSimple3D< TInElement, TOutElement, ThresholdingFtor< TInElement > >
			<<< gridSize, blockSize >>>( inBuffer, outBuffer, blockResolution, ThresholdingFtor< TInElement >( aThreshold, aBelowThreshold ) );

	cudaThreadSynchronize();

	CheckCudaErrorState( "After thresholding3D()" );

	D_PRINT( "Computations took " << clock.SecondsPassed() )

	cudaMemcpy(output.GetPointer(), outBuffer.mData, outBuffer.mLength * sizeof(uint8), cudaMemcpyDeviceToHost );
	CheckCudaErrorState( "Copy back" );
	cudaFree( inBuffer.mData );
	cudaFree( outBuffer.mData );
	CheckCudaErrorState( "Free memory" );
}

template void thresholding3D( M4D::Imaging::ImageRegion< int8, 3 > input, M4D::Imaging::MaskRegion3D output, int8 aThreshold, int8 aBelowThreshold );
template void thresholding3D( M4D::Imaging::ImageRegion< uint8, 3 > input, M4D::Imaging::MaskRegion3D output, uint8 aThreshold, uint8 aBelowThreshold );
template void thresholding3D( M4D::Imaging::ImageRegion< int16, 3 > input, M4D::Imaging::MaskRegion3D output, int16 aThreshold, int16 aBelowThreshold );
template void thresholding3D( M4D::Imaging::ImageRegion< uint16, 3 > input, M4D::Imaging::MaskRegion3D output, uint16 aThreshold, uint16 aBelowThreshold );
template void thresholding3D( M4D::Imaging::ImageRegion< int32, 3 > input, M4D::Imaging::MaskRegion3D output, int32 aThreshold, int32 aBelowThreshold );
template void thresholding3D( M4D::Imaging::ImageRegion< uint32, 3 > input, M4D::Imaging::MaskRegion3D output, uint32 aThreshold, uint32 aBelowThreshold );
template void thresholding3D( M4D::Imaging::ImageRegion< int64, 3 > input, M4D::Imaging::MaskRegion3D output, int64 aThreshold, int64 aBelowThreshold );
template void thresholding3D( M4D::Imaging::ImageRegion< uint64, 3 > input, M4D::Imaging::MaskRegion3D output, uint64 aThreshold, uint64 aBelowThreshold );
template void thresholding3D( M4D::Imaging::ImageRegion< float, 3 > input, M4D::Imaging::MaskRegion3D output, float aThreshold, float aBelowThreshold );
template void thresholding3D( M4D::Imaging::ImageRegion< double, 3 > input, M4D::Imaging::MaskRegion3D output, double aThreshold, double aBelowThreshold );