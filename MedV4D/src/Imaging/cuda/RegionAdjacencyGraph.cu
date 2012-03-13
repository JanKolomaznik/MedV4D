#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/ImageRegion.h"

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

	uint64 edgeCombIdx;
	//uint32 first;
	//uint32 second;
	float weight;
	uint32 count;
};

class EdgeHashTable: public Buffer1D<EdgeRecord>
{
	__device__ void 
	insertEdge( EdgeRecord & aEdge )
	{
		size_t idx = aEdge.edgeCombIdx % mLength;
		while ( true ) {
			if ( mData[idx].edgeCombIdx == 0 ) {
				if ( atomicCAS( &(mData[idx].edgeCombIdx), uint64(0), aEdge.edgeCombIdx ) == 0 ) {
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

__global__ void 
preallocationOfAdjacencyGraph( Buffer3D< uint32 > aRegionBuffer, EdgeHashTable aEdgeHashMap, int3 blockResolution )
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

	
	FillSharedMemory3D_8x8x8< uint32, cRadius, syStride, szStride >( inData, sidx, aRegionBuffer.mData, aRegionBuffer.mStrides, size, blockOrigin, coordinates, idx );
	
	__syncthreads();

	uint32 current = inData[sidx];
	uint32 second = inData[sidx+1];
	float weight = 1.0f;
	if ( current != second ) {
		aEdgeHashMap.insertEdge( EdgeRecord( current, second, weight ) );
	}
	second = inData[sidx+syStride];
	if ( current != second ) {
		aEdgeHashMap.insertEdge( EdgeRecord( current, second, weight ) );
	}
	second = inData[sidx+szStride];
	if ( current != second ) {
		aEdgeHashMap.insertEdge( EdgeRecord( current, second, weight ) );
	}
}


void
computeRegionAdjacencyGraph( Buffer3D< uint32 > aRegionBuffer )
{
	/*int3 radius = make_int3( 1, 1, 1 );


	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (inputBuffer.mLength + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	dim3 blockSize3D( 8, 8, 8 );
	int3 blockResolution3D = GetBlockResolution( inputBuffer.mSize, blockSize3D, make_int3(0,0,0) );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );


	preallocationOfAdjacencyGraph( aRegionBuffer, EdgeHashTable aEdgeHashMap, int3 blockResolution )*/
}