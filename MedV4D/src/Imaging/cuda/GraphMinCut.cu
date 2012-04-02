#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"



__device__ int pushSuccessful;
__device__ int relabelSuccessful;


struct EdgeList
{
	__device__ __host__ int
	size()const
	{ return 0; }
};

struct VertexList
{
	__device__ __host__ int
	size()const
	{ return 0; }

	__device__ float &
	getExcess( int aIdx )const
	{ return mExcessArray[aIdx]; }

	float *mExcessArray;
};


__device__ void
loadEdgeV1V2L1L2C( EdgeList aEdges, int aEdgeIdx, int &aV1, int &aV2, int &aLabel1, int &aLabel2, float &aResidualCapacity )
{

}

__device__ void
updateResiduals( EdgeList aEdges, int aEdgeIdx, float aPushedFlow )
{

}

__device__ float
tryPullFromVertex( VertexList aVertices, int aVertex, float aResidualCapacity )
{
	float excess = aVertices.getExcess( aVertex );
	float pushedFlow;
	while ( excess > 0.0f ) {
		pushedFlow = min( excess, aResidualCapacity );
		float oldExcess = atomicFloatCAS( &(aVertices.getExcess( aVertex )), excess, excess - pushedFlow );
		if( excess == oldExcess ) {
			return pushedFlow;
		} else {
			excess = oldExcess;
		}
	}
	return 0.0f;
}

__device__ void
pushToVertex( VertexList aVertices, int aVertex, float aPushedFlow )
{
	atomicAdd( &(aVertices.getExcess( aVertex )), aPushedFlow );
}

__global__ void 
pushKernel( EdgeList aEdges, VertexList aVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if ( edgeIdx < aEdges.size() ) {
		int v1, v2;
		int label1, label2;
		float residualCapacity;
		loadEdgeV1V2L1L2C( aEdges, edgeIdx, v1, v2, label1, label2, residualCapacity );

		if ( label1 > label2 && residualCapacity > 0 ) {
			float pushedFlow = tryPullFromVertex( aVertices, v1, residualCapacity );
			if( pushedFlow > 0.0f ) {
				pushToVertex( aVertices, v2, pushedFlow );
				updateResiduals( aEdges, edgeIdx, pushedFlow );
				pushSuccessful = 1;
			}
		}
	}
}
/*
__global__ void
relabelPhase2Kernel( EdgeList aEdges, VertexList aVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if ( edgeIdx < aEdges.size() ) {
		int v1, v2;
		int label1, label2;
		bool v1Enabled;
		loadEdge();

		if ( v1Enabled ) {
			if( label1 < label2 ) {
				trySetNewHeight( aVertices, v1, label2+1 );
			} else {
				disableVertex( aVertices, v1 );
			}
		}
	}
}*/

void
push( EdgeList aEdges, VertexList aVertices )
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (aEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	int pushSuccessful;
	

	size_t pushIterations = 0;
	do {
		++pushIterations;
		cudaMemcpyToSymbol( "pushSuccessful", &(pushSuccessful = 0), sizeof(int), 0, cudaMemcpyHostToDevice );
		pushKernel<<< gridSize1D, blockSize1D >>>( aEdges, aVertices );
		cudaMemcpyFromSymbol( &pushSuccessful, "pushSuccessful", sizeof(int), 0, cudaMemcpyDeviceToHost );

		cudaThreadSynchronize();
		CheckCudaErrorState( TO_STRING( "After push iteration n. " << pushIterations ) );
	} while ( pushSuccessful > 0 )

	D_PRINT( "Push iteration count = " << pushIterations );
}

void
relabel()
{

}

/*size_t aEdgeCount, aEdges, aWeights, aResidualCapacities, 
size_t aVertexCount, aVertices, aExcess, aLabels*/

/*void 
pushRelabelMaxFlow( aEdges, size_t aEdgeCount, aVertices, size_t aVertexCount, size_t aSourceID, size_t aSinkID )
{

}*/