#include "MedV4D/Imaging/cuda/detail/WatershedTransformation.cuh"
#include "MedV4D/Imaging/ImageRegion.h"

__device__ uint64 foundZero;

__global__ void 
isNonzeroKernel( Buffer3D< uint32 > buffer, int3 blockResolution )
{
	int3 blockCoordinates = GetBlockCoordinates( blockResolution, __mul24(blockIdx.y, gridDim.x) + blockIdx.x );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	int3 coordinates = blockOrigin + threadIdx;
	int3 size = toInt3( buffer.mSize );

	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), size );
	int idx = IdxFromCoordStrides( coordinates, buffer.mStrides );

	if ( !projected ) {
		if( buffer.mData[idx] == 0 ) {
			atomicAdd( &foundZero, 1 );
		}
	}
}


uint64
isNonzero( Buffer3D< uint32 > buffer )
{
	CheckCudaErrorState( "Before isNonzero()" );
	uint64 foundZero = 0;
	cudaMemcpyToSymbol( "foundZero", &(foundZero = 0), sizeof(uint64), 0, cudaMemcpyHostToDevice );

	dim3 blockSize( 8, 8, 8 );
	int3 blockResolution = GetBlockResolution( buffer.mSize, blockSize, make_int3(0,0,0) );
	dim3 gridSize( blockResolution.x * blockResolution.y, blockResolution.z, 1 );

	isNonzeroKernel<<< gridSize, blockSize >>>( buffer, blockResolution );
	cudaThreadSynchronize();
	cudaMemcpyFromSymbol( &foundZero, "foundZero", sizeof(uint64), 0, cudaMemcpyDeviceToHost );

	CheckCudaErrorState( "After isNonzero()" );
	return foundZero;
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
	//cudaFree( inBuffer.mData );
	//cudaFree( outBuffer.mData );
	CheckCudaErrorState( "Free memory" );
}

template< typename TEType >
void
watershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput )
{
	D_PRINT( "Before " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
	typedef typename TypeTraits< TEType >::SuperiorSignedType SignedElement;
	//typedef typename TypeTraits< TEType >::SignedClosestType SignedElement;
	//typedef typename TypeTraits< TEType >::SuperiorFloatType SignedElement;
	int wshedUpdated = 1;
	Buffer3D< uint32 > labeledRegionsBuffer = CudaBuffer3DFromImageRegionCopy( aLabeledMarkerRegions );
	Buffer3D< TEType > inputBuffer = CudaBuffer3DFromImageRegionCopy( aInput );
	Buffer3D< SignedElement > tmpBuffer = CudaPrepareBuffer<SignedElement>( aInput.GetSize() );

	Buffer3D< uint32 > labeledRegionsBuffer2 = CudaBuffer3DFromImageRegion( aLabeledMarkerRegions );
	Buffer3D< SignedElement > tmpBuffer2 = CudaPrepareBuffer<SignedElement>( aInput.GetSize() );
	ASSERT( labeledRegionsBuffer.mStrides == labeledRegionsBuffer2.mStrides );
	ASSERT( tmpBuffer.mStrides == tmpBuffer2.mStrides );

	//int3 radius = make_int3( 1, 1, 1 );
	D_PRINT( "After allocation in " << __FUNCTION__ << ": " << cudaMemoryInfoText() );

	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (inputBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	dim3 blockSize3D( 8, 8, 8 );
	int3 blockResolution3D = GetBlockResolution( inputBuffer.mSize, blockSize3D, make_int3(0,0,0) );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

	M4D::Common::Clock clock;
	D_PRINT( "InitWatershedBuffers()" );
	D_PRINT( "TypeTraits<SignedElement>::Max = " << TypeTraits<SignedElement>::Max );
	InitWatershedBuffers<<< gridSize1D, blockSize1D >>>( labeledRegionsBuffer, tmpBuffer, TypeTraits<SignedElement>::Max );

	unsigned i = 0;
	while (wshedUpdated != 0 && i < 1000) {
		cudaMemcpyToSymbol( "wshedUpdated", &(wshedUpdated = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

		/*WShedEvolution<<< gridSize3D, blockSize3D >>>( 
					inputBuffer,
					labeledRegionsBuffer,	
					tmpBuffer,
					blockResolution3D, 
					TypeTraits<SignedElement>::Max
					);*/

			WShedEvolution<<< gridSize3D, blockSize3D >>>( 
					inputBuffer,
					labeledRegionsBuffer,	
					tmpBuffer,
					labeledRegionsBuffer2,	
					tmpBuffer2,
					blockResolution3D, 
					TypeTraits<SignedElement>::Max
					);
		using std::swap;
		swap( labeledRegionsBuffer, labeledRegionsBuffer2 );
		swap( tmpBuffer, tmpBuffer2 );
		
		cudaThreadSynchronize();
		cudaMemcpyFromSymbol( &wshedUpdated, "wshedUpdated", sizeof(int), 0, cudaMemcpyDeviceToHost );
		++i;
	}
	D_PRINT( "wshedUpdated = " << wshedUpdated );
	
	LOG( "WatershedTransformation3D computations took " << clock.SecondsPassed() << " and " << i << " iterations" )

	LOG( "number of zero voxels = " << isNonzero( labeledRegionsBuffer ) );
	
	cudaMemcpy(aOutput.GetPointer(), labeledRegionsBuffer.mData, labeledRegionsBuffer.mLength * sizeof(uint32), cudaMemcpyDeviceToHost );
	
	//cudaFree( labeledRegionsBuffer.mData );
	//cudaFree( inputBuffer.mData );

	//cudaFree( labeledRegionsBuffer2.mData );
	//cudaFree( tmpBuffer2.mData );

	/*typename M4D::Imaging::Image< SignedElement, 3 >::Ptr tmpDebugImage = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< SignedElement, 3 >( aLabeledMarkerRegions.GetMinimum(), aLabeledMarkerRegions.GetMaximum(), aLabeledMarkerRegions.GetElementExtents() );
	cudaMemcpy(tmpDebugImage->GetRegion().GetPointer(), tmpBuffer.mData, labeledRegionsBuffer.mLength * sizeof(SignedElement), cudaMemcpyDeviceToHost );
	M4D::Imaging::ImageFactory::DumpImage( "Intermediate.dump", *tmpDebugImage );
*/
	//cudaFree( tmpBuffer.mData );

	D_PRINT( "After " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
}

#define DECLARE_TEMPLATE_INSTANCE template void watershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TTYPE, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
#include "MedV4D/Common/DeclareTemplateNumericInstances.h"

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

/*template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int8, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint8, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int16, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint16, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int32, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint32, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int64, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint64, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< float, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
template void WatershedTransformation3D( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< double, 3 > aInput, M4D::Imaging::ImageRegion< uint32, 3 > aOutput );
*/

