#include "MedV4D/Imaging/cuda/GraphOperations.h"
#include "MedV4D/Imaging/cuda/GraphDefinitions.h"

#include "MedV4D/Imaging/cuda/detail/GraphMinCut.cuh"
#include "MedV4D/Imaging/cuda/detail/RegionAdjacencyGraph.cuh"
#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"



template< typename TEType >
void
pushRelabelMaxFlow( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput )
{
	

	CheckCudaErrorState( "Before pushRelabelMaxFlow() toplevel code" );
	M4D::Common::Clock clock;
	D_PRINT( "Before " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
	Buffer3D< uint32 > labeledRegionsBuffer = CudaBuffer3DFromImageRegionCopy( aLabeledMarkerRegions );
	
	thrust::device_ptr<uint32 > res = thrust::max_element( 
						thrust::device_pointer_cast( labeledRegionsBuffer.mData ), 
						thrust::device_pointer_cast( labeledRegionsBuffer.mData+labeledRegionsBuffer.mLength ) 
						);
	size_t regionCount = *res;
	D_PRINT( "Region count " << regionCount );


	Buffer3D< TEType > inputBuffer = CudaBuffer3DFromImageRegionCopy( aInput );

	thrust::device_vector< EdgeRecord > edges( regionCount*25 );
	thrust::device_vector< float > edgeWeights( edges.size() );

	size_t edgeCount = 0;
	D_PRINT( "After allocation in " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
	fillEdgeList( labeledRegionsBuffer, inputBuffer, edges, edgeWeights, edgeCount );
	D_PRINT( "After fillEdgeList() : " << cudaMemoryInfoText() );
	//thrust::copy( edgeWeights.begin(), edgeWeights.begin() + edgeCount, std::ostream_iterator<float>(std::cout, "\n"));

	pushRelabelMaxFlow( edgeCount, regionCount, edges, edgeWeights, 1, regionCount - 1 ); //TODO - sink source

	cudaFree( labeledRegionsBuffer.mData );
	cudaFree( inputBuffer.mData );
}

template void pushRelabelMaxFlow<signed char>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<signed char, 3u>);
template void pushRelabelMaxFlow<unsigned char>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<unsigned char, 3u>);
template void pushRelabelMaxFlow<short>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<short, 3u>);
template void pushRelabelMaxFlow<unsigned short>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<unsigned short, 3u>);
template void pushRelabelMaxFlow<int>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<int, 3u>);
template void pushRelabelMaxFlow<unsigned int>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<unsigned int, 3u>);
template void pushRelabelMaxFlow<long long>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<long long, 3u>);
template void pushRelabelMaxFlow<unsigned long long>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<unsigned long long, 3u>);
template void pushRelabelMaxFlow<float>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<float, 3u>);
template void pushRelabelMaxFlow<double>(M4D::Imaging::ImageRegion<unsigned int, 3u>, M4D::Imaging::ImageRegion<double, 3u>);

