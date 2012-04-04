#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"

#include "MedV4D/Imaging/cuda/detail/GraphMinCut.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

__device__ int pushSuccessful;
__device__ int relabelSuccessful;
__device__ int bfsFrontNotEmpty;



__device__ void
loadEdgeV1V2L1L2C( EdgeList &aEdges, VertexList &aVertices, int aEdgeIdx, int &aV1, int &aV2, int &aLabel1, int &aLabel2, EdgeResidualsRecord &aResidualCapacities )
{
	EdgeRecord rec = aEdges.getEdge( aEdgeIdx );
	aV1 = rec.first;
	aV2 = rec.second;

	aLabel1 = aVertices.getLabel( aV1 );
	aLabel2 = aVertices.getLabel( aV2 );

	aResidualCapacities = aEdges.getResiduals( aEdgeIdx );
}

__device__ void
updateResiduals( EdgeList &aEdges, int aEdgeIdx, float aPushedFlow, int aFrom, int aTo )
{
	EdgeResidualsRecord &residuals = aEdges.getResiduals( aEdgeIdx );
	
	residuals.getResidual( aFrom < aTo ) -= aPushedFlow;
	residuals.getResidual( !(aFrom < aTo) ) += aPushedFlow;
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

inline __device__ void
pushToVertex( VertexList &aVertices, int aVertex, float aPushedFlow )
{
	atomicAdd( &(aVertices.getExcess( aVertex )), aPushedFlow );
}

inline __device__ void
tryToPushFromTo( VertexList &aVertices, int aFrom, int aTo, EdgeList &aEdges, int aEdgeIdx, float residualCapacity )
{
	//printf( "Push successfull\n" );
	if ( residualCapacity > 0 ) {
		float pushedFlow = tryPullFromVertex( aVertices, aFrom, residualCapacity );
		if( pushedFlow > 0.0f ) {
			pushToVertex( aVertices, aTo, pushedFlow );
			updateResiduals( aEdges, aEdgeIdx, pushedFlow, aFrom, aTo );
			pushSuccessful = 1;
			printf( "Push successfull\n" );
		}
	}
}

__global__ void 
pushKernel( EdgeList aEdges, VertexList aVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if ( edgeIdx < aEdges.size() ) {
		int v1, v2;
		int label1, label2;
		EdgeResidualsRecord residualCapacities;
		loadEdgeV1V2L1L2C( aEdges, aVertices, edgeIdx, v1, v2, label1, label2, residualCapacities );
		//printf( "%i -> %i => %i -> %i %f; %f\n", v1, label1, v2, label2, residualCapacities.getResidual( true ), residualCapacities.getResidual( false ) );
		if ( label1 > label2 ) {
			tryToPushFromTo( aVertices, v1, v2, aEdges, edgeIdx, residualCapacities.getResidual( v1 < v2 ) );
		} else if ( label1 < label2 ) {
			tryToPushFromTo( aVertices, v2, v1, aEdges, edgeIdx, residualCapacities.getResidual( v2 < v1 ) );
		}

		
	}

	/*if ( label1 > label2 && residualCapacity > 0 ) {
			float pushedFlow = tryPullFromVertex( aVertices, v1, residualCapacity );
			if( pushedFlow > 0.0f ) {
				pushToVertex( aVertices, v2, pushedFlow );
				updateResiduals( aEdges, edgeIdx, pushedFlow, v1 < v2 );
				pushSuccessful = 1;
			}
		}*/
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
		bool v2Enabled = aEnabledVertices[v2];

		if ( v1Enabled ) {
			if( label1 < label2 || v2 == aSink ) { //TODO check if edge is saturated in case its leading down
				trySetNewHeight( aLabels, v1, label2+1 );
			} else {
				aEnabledVertices[v1] = false;
			}
		}
		if ( v2Enabled ) {
			if( label2 < label1 || v1 == aSink ) { //TODO check if edge is saturated in case its leading down
				trySetNewHeight( aLabels, v2, label1+1 );
			} else {
				aEnabledVertices[v2] = false;
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

struct DummyOperation
{
	__device__ void
	processEdge( const EdgeRecord &rec, int edgeIdx, int aStep, bool aFirst )
	{ /*empty*/ }

	__device__ void
	processVertex( int vertexIdx, int aStep )
	{ /*empty*/ }
};

struct SetLabelOperation: public DummyOperation
{
	SetLabelOperation( int *aLabels ): mLabels( aLabels )
	{}

	__device__ void
	processVertex( int vertexIdx, int aStep )
	{ 	
		CUDA_ASSERT( vertexIdx != 0 );
		mLabels[vertexIdx] = aStep; 
	}
	int *mLabels;
};

template< typename TOperation >
__global__ void
bfsKernel( EdgeList aEdges, VertexList aVertices, bool *aFrontier, bool *aFrontierNew, bool *aVisited, int aStep, TOperation aOperation )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int edgeIdx = blockId * blockDim.x + threadIdx.x;

	if ( edgeIdx < aEdges.size() ) {
		EdgeRecord rec = aEdges.getEdge( edgeIdx );
		if ( aFrontier[ rec.first ] ) {
			aOperation.processEdge( rec, edgeIdx, aStep, true );
			if ( !aVisited[ rec.second ] ) {
				aFrontierNew[ rec.second ] = true;
				bfsFrontNotEmpty = 1;
			}
		}
		if ( aFrontier[ rec.second ] ) {
			aOperation.processEdge( rec, edgeIdx, aStep, false );
			if ( !aVisited[ rec.first ] ) {
				aFrontierNew[ rec.first ] = true;
				bfsFrontNotEmpty = 1;
			}
		}
	}
}

template< typename TOperation >
__global__ void
bfsMarkVisitedKernel( VertexList aVertices, bool *aFrontier, bool *aVisited, int aStep, TOperation aOperation )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if ( vertexIdx < aVertices.size() && vertexIdx > 0 ) {
		if ( aFrontier[vertexIdx] ) {
			aOperation.processVertex( vertexIdx, aStep );
			aVisited[vertexIdx] = true;
		}
	}
}

template< typename TOperation >
int
bfsSearch( EdgeList &aEdges, VertexList &aVertices, int aStart, TOperation aOperation )
{
	dim3 blockSize1D( 512 );
	dim3 vertexGridSize1D( (aVertices.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
	dim3 edgeGridSize1D( (aEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	thrust::device_vector< bool > frontier( aVertices.size() );
	thrust::device_vector< bool > frontier2( aVertices.size() );
	thrust::device_vector< bool > visited( aVertices.size() );

	int bfsFrontNotEmpty;
	
	frontier[ aStart ] = true;
	visited[ aStart ] = true;

	int bfsStepCount = 0;
	do {
		++bfsStepCount;
		cudaMemcpyToSymbol( "bfsFrontNotEmpty", &(bfsFrontNotEmpty = 0), sizeof(int), 0, cudaMemcpyHostToDevice );
		
		thrust::fill( frontier2.begin(), frontier2.end(), false );

		bfsKernel<<< edgeGridSize1D, blockSize1D >>>( 
						aEdges, 
						aVertices, 
						thrust::raw_pointer_cast(&frontier[0]), 
						thrust::raw_pointer_cast(&frontier2[0]),
						thrust::raw_pointer_cast(&visited[0]),
						bfsStepCount,
						aOperation
						);
		cudaThreadSynchronize();
		
		bfsMarkVisitedKernel<<< vertexGridSize1D, blockSize1D >>>( 
						aVertices, 
						thrust::raw_pointer_cast(&frontier2[0]), 
						thrust::raw_pointer_cast(&visited[0]),
						bfsStepCount,
						aOperation
						);

		cudaThreadSynchronize();
		
		frontier.swap( frontier2 );
		cudaMemcpyFromSymbol( &bfsFrontNotEmpty, "bfsFrontNotEmpty", sizeof(int), 0, cudaMemcpyDeviceToHost );
		CheckCudaErrorState( TO_STRING( "After BFS iteration n. " << bfsStepCount ) );
	} while ( bfsFrontNotEmpty > 0 );

	D_PRINT( "Exiting BFS after ste n.: " << bfsStepCount );
	return bfsStepCount;
}

struct RevertLabels: public thrust::unary_function< int, int >
{
	RevertLabels( int aHeight ) :mHeight( aHeight )
	{}

	__host__ __device__ int 
	operator()( const int &aVal )const
	{
		return mHeight - aVal;
	}
	int mHeight;
};

void
initLabels( EdgeList &aEdges, VertexList &aVertices, int aSource, int aSink )
{
	D_COMMAND( M4D::Common::Clock clock; );
	int h = 1 + bfsSearch( aEdges, aVertices, aSource, SetLabelOperation( aVertices.mLabelArray ) );
	thrust::transform( 
			thrust::device_pointer_cast( aVertices.mLabelArray + 1 ), 
			thrust::device_pointer_cast( aVertices.mLabelArray + aVertices.size() ), 
			thrust::device_pointer_cast( aVertices.mLabelArray + 1 ),
			RevertLabels( h )
			);

	D_PRINT( "init labels execution time = " << clock.SecondsPassed() );
	//thrust::copy( thrust::device_pointer_cast( aVertices.mLabelArray + 1 ), thrust::device_pointer_cast( aVertices.mLabelArray + aVertices.size() ), std::ostream_iterator<int>(std::cout, "\n"));
}

bool
push( EdgeList &aEdges, VertexList &aVertices, int aSource, int aSink )
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (aEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	//D_PRINT( "gridSize1D " << gridSize1D.x << "; " << gridSize1D.y << "; " << gridSize1D.z );
	static const float SOURCE_EXCESS = 1000000.0f;
	int pushSuccessful;
	D_COMMAND( M4D::Common::Clock clock; );
	size_t pushIterations = 0;
	do {
		CUDA_CHECK_RESULT( cudaMemcpy( (void*)(aVertices.mExcessArray + aSource), (void*)(&SOURCE_EXCESS), sizeof(float), cudaMemcpyHostToDevice ) );

		++pushIterations;
		CUDA_CHECK_RESULT( cudaMemcpyToSymbol( "pushSuccessful", &(pushSuccessful = 0), sizeof(int), 0, cudaMemcpyHostToDevice ) );
		
		CheckCudaErrorState( TO_STRING( "Before push kernel n. " << pushIterations ) );
		pushKernel<<< gridSize1D, blockSize1D >>>( aEdges, aVertices );
		CheckCudaErrorState( TO_STRING( "After push iteration n. " << pushIterations ) );
		
		CUDA_CHECK_RESULT( cudaThreadSynchronize() );

		CUDA_CHECK_RESULT( cudaMemcpyFromSymbol( &pushSuccessful, "pushSuccessful", sizeof(int), 0, cudaMemcpyDeviceToHost ) );
		
	} while ( pushSuccessful > 0 );

	D_PRINT( "Push iteration count = " << pushIterations << "; took " << clock.SecondsPassed() );
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
	D_COMMAND( M4D::Common::Clock clock; );
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
	D_PRINT( "Relabel result = " << relabelSuccessful << "; took " << clock.SecondsPassed() );
	return relabelSuccessful > 0;
}

/*size_t aEdgeCount, aEdges, aWeights, aResidualCapacities, 
size_t aVertexCount, aVertices, aExcess, aLabels*/

void 
pushRelabelMaxFlow( EdgeList &aEdges, VertexList &aVertices, int aSourceID, int aSinkID )
{
	CheckCudaErrorState( "Before pushRelabelMaxFlow() code" );

	D_PRINT( "Edges size = " << aEdges.size() );
	D_PRINT( "Vertices size = " << aVertices.size() );
	D_PRINT( "Source = " << aSourceID  << "; sink = " << aSinkID );

	D_PRINT( "Entering inner pushRelabelMaxFlow()" );
	thrust::device_vector< int > tmpLabels( aVertices.size() );
	CheckCudaErrorState( "tmpLabels allocation" );
	thrust::device_vector< bool > tmpEnabledVertex( aVertices.size() );
	CheckCudaErrorState( "tmpEnabledVertex allocation" );
	D_PRINT( "tmpEnabledVertex allocated : " << cudaMemoryInfoText() );

	initLabels( aEdges, aVertices, aSourceID, aSinkID );

	bool res = push( aEdges, aVertices, aSourceID, aSinkID );

	res = relabel( aEdges, aVertices, tmpEnabledVertex, tmpLabels, aSourceID, aSinkID );
	D_PRINT( "Leaving inner pushRelabelMaxFlow()" );
}

struct WeightToResiduals: public thrust::unary_function< float, EdgeResidualsRecord >
{
	__host__ __device__ EdgeResidualsRecord
	operator()( const float &aWeight ) {
		return EdgeResidualsRecord( aWeight );
	}
};

void
pushRelabelMaxFlow( size_t aEdgeCount, size_t aVertexCount, thrust::device_vector< EdgeRecord > &aEdges, thrust::device_vector< float > &aWeights, int aSourceID, int aSinkID )
{
	
	thrust::device_vector< EdgeResidualsRecord > residuals( aEdgeCount );
	thrust::device_vector< int >  labels( aVertexCount + 1 );
	thrust::device_vector< float >  excess( aVertexCount + 1 );
	D_PRINT( "After allocation of residual, labels and excess : " << cudaMemoryInfoText() );
	
	thrust::transform( aWeights.begin(), aWeights.begin() + aEdgeCount, residuals.begin(), WeightToResiduals() );
	D_PRINT( "After filling residuals." );

	//thrust::copy( aWeights.begin(), aWeights.begin() + aEdgeCount, std::ostream_iterator<int>(std::cout, "\n"));

	EdgeList edgeList( aEdges, aWeights, residuals, aEdgeCount );
	VertexList vertexList( labels, excess, aVertexCount );
	D_PRINT( "After filling edgeList and vertexList." );

	pushRelabelMaxFlow( edgeList, vertexList, aSourceID, aSinkID );

	D_PRINT( "Leaving outer pushRelabelMaxFlow()" );
}


template< typename TEType >
void
pushRelabelMaxFlow( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput );

