#include "MedV4D/Imaging/cuda/GraphOperations.h"
#include "MedV4D/Imaging/cuda/GraphDefinitions.h"

#include "MedV4D/Imaging/cuda/detail/GraphMinCut.cuh"
#include "MedV4D/Imaging/cuda/detail/RegionAdjacencyGraph.cuh"
#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"


__global__ void 
getMarkedRegionsIDsKernel( Buffer3D< uint32 > aLabeledRegions, uint32 aRegionCount, Buffer3D< uint8 > aMarkers, bool *aMarkedRegions1, bool *aMarkedRegions2 )
{ 
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < aMarkers.mLength ) {
		uint8 val = aMarkers.mData[idx];
		if ( val > 0 ) {
			uint32 tmp = aLabeledRegions.mData[idx];
			
			if( val > 240 ) {
				aMarkedRegions2[tmp] = true;
			} else {
				aMarkedRegions1[tmp] = true;
			}
		}
	}
}

void
getMarkedRegionsIDs( const Buffer3D< uint32 > &aLabeledRegions, uint32 aRegionCount, const Buffer3D< uint8 > &aMarkers )
{
	thrust::device_vector< bool > markedRegions1( aRegionCount+1, false );
	thrust::device_vector< bool > markedRegions2( aRegionCount+1, false );
}


template< typename TEType >
void
pushRelabelMaxFlow( const Buffer3D< uint32 > &aLabeledRegions, const Buffer3D< TEType > &aInput )
{
	CheckCudaErrorState( "Before pushRelabelMaxFlow() toplevel code" );
	M4D::Common::Clock clock;
	
	thrust::device_ptr<uint32 > res = thrust::max_element( 
						thrust::device_pointer_cast( aLabeledRegions.mData ), 
						thrust::device_pointer_cast( aLabeledRegions.mData+aLabeledRegions.mLength ) 
						);
	size_t regionCount = *res;
	D_PRINT( "Region count " << regionCount );

	thrust::device_vector< EdgeRecord > edges( regionCount*25 );
	thrust::device_vector< float > edgeWeights( edges.size() );

	size_t edgeCount = 0;
	D_PRINT( "After allocation in " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
	fillEdgeList( aLabeledRegions, aInput, edges, edgeWeights, edgeCount );
	D_PRINT( "After fillEdgeList() : " << cudaMemoryInfoText() );
	//thrust::copy( edgeWeights.begin(), edgeWeights.begin() + edgeCount, std::ostream_iterator<float>(std::cout, "\n"));

	pushRelabelMaxFlow( edgeCount, regionCount, edges, edgeWeights, 1, regionCount - 1 ); //TODO - sink source
}

template< typename TEType >
void
pushRelabelMaxFlow( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput )
{
	CheckCudaErrorState( "Before pushRelabelMaxFlow() toplevel code" );
	M4D::Common::Clock clock;
	D_PRINT( "Before " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
	Buffer3D< uint32 > labeledRegionsBuffer = CudaBuffer3DFromImageRegionCopy( aLabeledRegions );
	Buffer3D< TEType > inputBuffer = CudaBuffer3DFromImageRegionCopy( aInput );

	pushRelabelMaxFlow( labeledRegionsBuffer, inputBuffer );

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

