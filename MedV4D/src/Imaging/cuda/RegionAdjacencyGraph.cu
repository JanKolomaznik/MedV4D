#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/ImageRegion.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/extrema.h>

struct VertexRecord
{
	size_t edgeStart;
};

struct EdgeRecord
{
	__device__ __host__ 
	EdgeRecord( uint32 aFirst, uint32 aSecond, float aWeight ): weight(aWeight), count( 1 )
	{
		if( aFirst < aSecond ) {
			edgeCombIdx = uint64(aFirst) << 32 + aSecond;
		} else {
			edgeCombIdx = uint64(aSecond) << 32 + aFirst;
		}
	}
	__device__ __host__ 
	EdgeRecord(): edgeCombIdx(0), weight(0.0f), count(0)
	{ }

	uint64 edgeCombIdx;
	//uint32 first;
	//uint32 second;
	float weight;
	uint32 count;
};

struct GetSymmetricEdge : public thrust::unary_function< EdgeRecord, EdgeRecord >
{
	__host__ __device__ EdgeRecord operator()(const EdgeRecord &aEdge) const 
	{
		EdgeRecord rec = aEdge;
		rec.edgeCombIdx = ((aEdge.edgeCombIdx & 0xFFFFFFFF) << 32) + (aEdge.edgeCombIdx >> 32);
		return rec;
	}
};

struct CompareEdge : public thrust::binary_function< EdgeRecord, EdgeRecord, bool>
{
	__host__ __device__ bool operator()( const EdgeRecord &aEdgeA, const EdgeRecord &aEdgeB ) const 
	{
		if ( aEdgeA.edgeCombIdx == 0 ) return false;
		if ( aEdgeB.edgeCombIdx == 0 ) return true;
		
		return aEdgeA.edgeCombIdx < aEdgeB.edgeCombIdx;
	}
};

struct IsValidEdge : public thrust::unary_function< EdgeRecord, EdgeRecord >
{
	__host__ __device__ bool operator()(const EdgeRecord &aEdge) const 
	{
		return aEdge.edgeCombIdx != 0;
	}
};

class EdgeHashTable: public Buffer1D<EdgeRecord>
{
public:
	EdgeHashTable( thrust::device_vector< EdgeRecord > &aVect ): Buffer1D<EdgeRecord>( aVect.size(), thrust::raw_pointer_cast(&aVect[0]) )
	{ }

	EdgeHashTable( size_t aLength, EdgeRecord *aData ): Buffer1D<EdgeRecord>( aLength, aData )
	{ }

	__device__ void 
	insertEdge( const EdgeRecord & aEdge )
	{
		size_t idx = aEdge.edgeCombIdx % mLength;
		while ( true ) {
			if ( mData[idx].edgeCombIdx == 0 ) {
				if ( atomicCAS( &(mData[idx].edgeCombIdx), uint64(0), aEdge.edgeCombIdx ) == 0 ) {
					atomicAdd( &(mData[idx].weight), aEdge.weight );
					return;
				}
			} else {
				if ( mData[idx].edgeCombIdx == aEdge.edgeCombIdx ) {
					atomicAdd( &(mData[idx].count), aEdge.count );
					atomicAdd( &(mData[idx].weight), aEdge.weight );
					return;
				}
			}
			idx = (idx+1) % mLength;
		}
	}
};

template< typename TEType >
__global__ void 
preallocationOfAdjacencyGraph( Buffer3D< uint32 > aRegionBuffer, Buffer3D< TEType > aGradientBuffer, EdgeHashTable aEdgeHashMap, int3 blockResolution )
{
	__shared__ uint32 inData[MAX_SHARED_MEMORY];
	
	const int cBlockDim = 8;
	const int cRadius = 1;
	const uint syStride = cBlockDim+2*cRadius;
	const uint szStride = (cBlockDim+2*cRadius) * (cBlockDim+2*cRadius);

	uint sidx = (threadIdx.y+cRadius) * syStride + (threadIdx.z+cRadius) * szStride + threadIdx.x + cRadius;
	int3 size = toInt3( aRegionBuffer.mSize );
	int3 blockCoordinates = GetBlockCoordinates( blockResolution, __mul24(blockIdx.y, gridDim.x) + blockIdx.x );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	
	
	int3 coordinates = blockOrigin + threadIdx;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), size );
	int idx = IdxFromCoordStrides( coordinates, aRegionBuffer.mStrides );
	int idx2 = IdxFromCoordStrides( coordinates, aGradientBuffer.mStrides );

	
	FillSharedMemory3D_8x8x8< uint32, cRadius, syStride, szStride >( inData, sidx, aRegionBuffer.mData, aRegionBuffer.mStrides, size, blockOrigin, coordinates, idx );
	
	__syncthreads();

	uint32 current = inData[sidx];
	uint32 second = inData[sidx+1];
	TEType val = aGradientBuffer.mData[idx2];
	if ( current != second ) {
		float weight = max( val, aGradientBuffer.mData[idx2+1] );
		aEdgeHashMap.insertEdge( EdgeRecord( current, second, weight ) );
	}
	second = inData[sidx+syStride];
	if ( current != second ) {
		float weight = max( val, aGradientBuffer.mData[idx2+aGradientBuffer.mStrides.y] );
		aEdgeHashMap.insertEdge( EdgeRecord( current, second, weight ) );
	}
	second = inData[sidx+szStride];
	if ( current != second ) {
		float weight = max( val, aGradientBuffer.mData[idx2+aGradientBuffer.mStrides.z] );
		aEdgeHashMap.insertEdge( EdgeRecord( current, second, weight ) );
	}
}

template< typename TEType >
void
computeRegionAdjacencyGraph( Buffer3D< uint32 > &aRegionBuffer, Buffer3D< TEType > &aGradientBuffer, thrust::device_vector< EdgeRecord > &aEdges, thrust::device_vector< VertexRecord > &aVertices )
{
	int3 radius = make_int3( 1, 1, 1 );

	EdgeHashTable edgeHashMap( aEdges.size()/2, thrust::raw_pointer_cast(&aEdges[0]) );

	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (aEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	dim3 blockSize3D( 8, 8, 8 );
	int3 blockResolution3D = GetBlockResolution( aRegionBuffer.mSize, blockSize3D, make_int3(0,0,0) );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

	M4D::Common::Clock clock;
	CheckCudaErrorState( "Before preallocationOfAdjacencyGraph()" );
	preallocationOfAdjacencyGraph< TEType >
		<<< gridSize3D, blockSize3D >>>( aRegionBuffer, aGradientBuffer, edgeHashMap, blockResolution3D );
	cudaThreadSynchronize();
	CheckCudaErrorState( "After preallocationOfAdjacencyGraph()" );

	thrust::transform( aEdges.begin(), aEdges.begin() + aEdges.size()/2, aEdges.begin() + aEdges.size()/2, GetSymmetricEdge() );
	thrust::sort( aEdges.begin(), aEdges.end(), CompareEdge() );
	size_t edgeCount = thrust::count_if( aEdges.begin(), aEdges.end(), IsValidEdge() );
	LOG( "computeRegionAdjacencyGraph computations took " << clock.SecondsPassed() );
	D_PRINT( "created " << edgeCount << " edges" );
}

template< typename TEType >
void
createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput, size_t aRegionCount )
{
	Buffer3D< uint32 > labeledRegionsBuffer = CudaBuffer3DFromImageRegionCopy( aLabeledMarkerRegions );
	Buffer3D< TEType > inputBuffer = CudaBuffer3DFromImageRegionCopy( aInput );

	aRegionCount = *(thrust::max_element( thrust::device_pointer_cast( labeledRegionsBuffer.mData ), thrust::device_pointer_cast( labeledRegionsBuffer.mData+labeledRegionsBuffer.mLength ) ));
	D_PRINT( "Region count " << aRegionCount );

	thrust::device_vector< VertexRecord > vertices( aRegionCount );
	thrust::device_vector< EdgeRecord > edges( aRegionCount*30 );

	computeRegionAdjacencyGraph( labeledRegionsBuffer, inputBuffer, edges, vertices );
}

template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int8, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint8, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int16, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint16, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int32, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint32, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int64, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint64, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< float, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< double, 3 > aInput, size_t aRegionCount );

