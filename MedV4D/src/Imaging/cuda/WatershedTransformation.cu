#include "MedV4D/Imaging/cuda/detail/WatershedTransformation.cuh"
#include "MedV4D/Imaging/ImageRegion.h"


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
	LOG( "WatershedTransformation3D computations took " << clock.SecondsPassed() << " and " << i << " iterations" )

	cudaMemcpy(aOutput.GetPointer(), labeledRegionsBuffer.mData, labeledRegionsBuffer.mLength * sizeof(uint32), cudaMemcpyDeviceToHost );
	cudaFree( labeledRegionsBuffer.mData );
	cudaFree( inputBuffer.mData );


	/*typename M4D::Imaging::Image< SignedElement, 3 >::Ptr tmpDebugImage = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< SignedElement, 3 >( aLabeledMarkerRegions.GetMinimum(), aLabeledMarkerRegions.GetMaximum(), aLabeledMarkerRegions.GetElementExtents() );
	cudaMemcpy(tmpDebugImage->GetRegion().GetPointer(), tmpBuffer.mData, labeledRegionsBuffer.mLength * sizeof(SignedElement), cudaMemcpyDeviceToHost );
	M4D::Imaging::ImageFactory::DumpImage( "Intermediate.dump", *tmpDebugImage );
*/
	cudaFree( tmpBuffer.mData );
}

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


