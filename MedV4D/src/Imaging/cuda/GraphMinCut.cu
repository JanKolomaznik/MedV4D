#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"

#include "MedV4D/Imaging/cuda/GraphDefinitions.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

__device__ int pushSuccessful;
__device__ int relabelSuccessful;





__device__ void
loadEdgeV1V2L1L2C( EdgeList &aEdges, VertexList &aVertices, int aEdgeIdx, int &aV1, int &aV2, int &aLabel1, int &aLabel2, float &aResidualCapacity )
{
	EdgeRecord rec = aEdges.getEdge( aEdgeIdx );
	aV1 = rec.first;
	aV2 = rec.first;

	aLabel1 = aVertices.getLabel( aV1 );
	aLabel2 = aVertices.getLabel( aV2 );

	int tmpIdx = aV1 < aV2 ? 0 : 1;

	aResidualCapacity = aEdges.getResiduals( aEdgeIdx ).residuals[ tmpIdx ];
}

__device__ void
updateResiduals( EdgeList &aEdges, int aEdgeIdx, float aPushedFlow, bool aFirst )
{
	EdgeResidualsRecord &residuals = aEdges.getResiduals( aEdgeIdx );
	int tmpIdx = aFirst ? 0 : 1;
	
	residuals.residuals[ tmpIdx ] -= aPushedFlow;
	residuals.residuals[ (tmpIdx+1)%2 ] += aPushedFlow;
}

__device__ float
tryPullFromVertex( VertexList &aVertices, int aVertex, float aResidualCapacity )
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
pushToVertex( VertexList &aVertices, int aVertex, float aPushedFlow )
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
		loadEdgeV1V2L1L2C( aEdges, aVertices, edgeIdx, v1, v2, label1, label2, residualCapacity );

		if ( label1 > label2 && residualCapacity > 0 ) {
			float pushedFlow = tryPullFromVertex( aVertices, v1, residualCapacity );
			if( pushedFlow > 0.0f ) {
				pushToVertex( aVertices, v2, pushedFlow );
				updateResiduals( aEdges, edgeIdx, pushedFlow, v1 < v2 );
				pushSuccessful = 1;
			}
		}
	}
}
//*********************************************************************************************************
__global__ void
relabelPhase1Kernel( VertexList aVertices, bool *aEnabledVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if ( vertexIdx < aVertices.size() && vertexIdx > 0 ) {
		aEnabledVertices[ vertexIdx ] = aVertices.getExcess( vertexIdx ) > 0.0f;
	}
}

__device__ void
loadEdgeV1V2L1L2( EdgeList &aEdges, VertexList &aVertices, int aEdgeIdx, int &aV1, int &aV2, int &aLabel1, int &aLabel2 )
{
	EdgeRecord rec = aEdges.getEdge( aEdgeIdx );
	aV1 = rec.first;
	aV2 = rec.first;

	aLabel1 = aVertices.getLabel( aV1 );
	aLabel2 = aVertices.getLabel( aV2 );
}

__device__ void
trySetNewHeight( int *aLabels, int aVertexIdx, int label )
{
	atomicMax( aLabels + aVertexIdx, label );
}

__global__ void
relabelPhase2Kernel( EdgeList aEdges, VertexList aVertices, bool *aEnabledVertices, int *aLabels, int aSource, int aSink )
{
	
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if ( edgeIdx < aEdges.size() ) {
		int v1, v2;
		int label1, label2;
		loadEdgeV1V2L1L2( aEdges, aVertices, edgeIdx, v1, v2, label1, label2 );

		bool v1Enabled = aEnabledVertices[v1];
		if ( v1Enabled ) {
			if( label1 < label2 || v2 == aSink ) { //TODO check if edge is saturated in case its leading down
				trySetNewHeight( aLabels, v1, label2+1 );
			} else {
				aEnabledVertices[v1] = false;
			}
		}
	}
}

__global__ void
relabelPhase3Kernel( VertexList aVertices, bool *aEnabledVertices, int *aLabels )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if ( vertexIdx < aVertices.size() && vertexIdx > 0 ) {
		if ( aEnabledVertices[ vertexIdx ] ) {
			aVertices.getLabel( vertexIdx ) = aLabels[ vertexIdx ];
			relabelSuccessful = 1;
		}
	}
}
//*********************************************************************************************************
void
initLabels( EdgeList &aEdges, VertexList &aVertices, int aSource, int aSink )
{
	//TODO
}

bool
push( EdgeList &aEdges, VertexList &aVertices, int aSource, int aSink )
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (aEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	int pushSuccessful;
	
	size_t pushIterations = 0;
	do {
		++pushIterations;
		cudaMemcpyToSymbol( "pushSuccessful", &(pushSuccessful = 0), sizeof(int), 0, cudaMemcpyHostToDevice );
		pushKernel<<< gridSize1D, blockSize1D >>>( aEdges, aVertices );

		cudaThreadSynchronize();

		cudaMemcpyFromSymbol( &pushSuccessful, "pushSuccessful", sizeof(int), 0, cudaMemcpyDeviceToHost );
		CheckCudaErrorState( TO_STRING( "After push iteration n. " << pushIterations ) );
	} while ( pushSuccessful > 0 );

	D_PRINT( "Push iteration count = " << pushIterations );
	return pushIterations > 0;
}

bool
relabel( EdgeList &aEdges, VertexList &aVertices, thrust::device_vector< bool > &aEnabledVertices, thrust::device_vector< int > &aLabels, int aSource, int aSink )
{
	dim3 blockSize1D( 512 );
	dim3 vertexGridSize1D( (aVertices.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	dim3 edgeGridSize1D( (aEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	int relabelSuccessful;
	cudaMemcpyToSymbol( "relabelSuccessful", &(relabelSuccessful = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

	relabelPhase1Kernel<<< vertexGridSize1D, blockSize1D >>>( 
					aVertices, 
					thrust::raw_pointer_cast(&aEnabledVertices[0]) 
					);
	cudaThreadSynchronize();
	CheckCudaErrorState( "After relabelPhase1Kernel()" );
	
	relabelPhase2Kernel<<< edgeGridSize1D, blockSize1D >>>( 
					aEdges, 
					aVertices, 
					thrust::raw_pointer_cast(&aEnabledVertices[0]), 
					thrust::raw_pointer_cast(&aLabels[0]),
					aSource,
					aSink
					);
	cudaThreadSynchronize();
	CheckCudaErrorState( "After relabelPhase2Kernel()" );

	
	//Sink and source doesn't change height
	aEnabledVertices[aSource] = false;
	aEnabledVertices[aSink] = false;
	relabelPhase3Kernel<<< vertexGridSize1D, blockSize1D >>>( 
					aVertices, 
					thrust::raw_pointer_cast(&aEnabledVertices[0]), 
					thrust::raw_pointer_cast(&aLabels[0]) 
					);
	cudaThreadSynchronize();
	CheckCudaErrorState( "After relabelPhase3Kernel()" );

	
	cudaMemcpyFromSymbol( &relabelSuccessful, "relabelSuccessful", sizeof(int), 0, cudaMemcpyDeviceToHost );

	return relabelSuccessful > 0;
}

/*size_t aEdgeCount, aEdges, aWeights, aResidualCapacities, 
size_t aVertexCount, aVertices, aExcess, aLabels*/

void 
pushRelabelMaxFlow( EdgeList &aEdges, VertexList &aVertices, int aSourceID, int aSinkID )
{
	thrust::device_vector< int > tmpLabels( aVertices.size() );
	thrust::device_vector< bool > tmpEnabledVertex( aVertices.size() );

	initLabels( aEdges, aVertices, aSourceID, aSinkID );

	bool res = push( aEdges, aVertices, aSourceID, aSinkID );

	res = relabel( aEdges, aVertices, tmpEnabledVertex, tmpLabels, aSourceID, aSinkID );
}