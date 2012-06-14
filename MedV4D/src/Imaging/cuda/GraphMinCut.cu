#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/GraphTools.h"

#include "MedV4D/Imaging/cuda/detail/GraphMinCut.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

__device__ int pushSuccessful;
__device__ int relabelSuccessful;
__device__ int bfsFrontNotEmpty;
__device__ int cutFound;



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
			//printf( "Push successfull from %i to %i (edge %i), flow = %f\n", aFrom, aTo, aEdgeIdx, pushedFlow );
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
#define MAX_LABEL MAX_INT32
__global__ void
relabelPhase1Kernel( VertexList aVertices, bool *aEnabledVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if ( vertexIdx < aVertices.size() && vertexIdx > 0 ) {
		aEnabledVertices[ vertexIdx ] = aVertices.getExcess( vertexIdx ) > 0.0f;
	}
}

/*__device__ void
loadEdgeV1V2L1L2( EdgeList &aEdges, VertexList &aVertices, int aEdgeIdx, int &aV1, int &aV2, int &aLabel1, int &aLabel2 )
{
	EdgeRecord rec = aEdges.getEdge( aEdgeIdx );
	aV1 = rec.first;
	aV2 = rec.first;

	aLabel1 = aVertices.getLabel( aV1 );
	aLabel2 = aVertices.getLabel( aV2 );
}*/

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
		EdgeResidualsRecord residualCapacities;
		loadEdgeV1V2L1L2C( aEdges, aVertices, edgeIdx, v1, v2, label1, label2, residualCapacities );

		bool v1Enabled = aEnabledVertices[v1];
		bool v2Enabled = aEnabledVertices[v2];

		if ( v1Enabled ) { //TODO - check if set to maximum is right
			if( label1 <= label2 || residualCapacities.getResidual( v1 < v2 ) <= 0.0f/* || v2 == aSink*//* || v2 == aSource*/ ) { //TODO check if edge is saturated in case its leading down
				trySetNewHeight( aLabels, v1, label2+1 );
			} else {
				//printf( "%i -> %i, l1 %i l2 %i label1\n", v1, v2, label1, label2 );
				aEnabledVertices[v1] = false;
			}
		}
		if ( v2Enabled ) {
			if( label2 <= label1 || residualCapacities.getResidual( v2 < v1 ) <= 0.0f/* || v1 == aSink*//* || v1 == aSource*/  ) { //TODO check if edge is saturated in case its leading down
				trySetNewHeight( aLabels, v2, label1+1 );
			} else {
				aEnabledVertices[v2] = false;
			}
		}
	}
}

__global__ void
relabelPhase3Kernel( VertexList aVertices, bool *aEnabledVertices, int *aLabels, int aSource, int aSink )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if ( vertexIdx < aVertices.size() && vertexIdx > 0 ) {
		if ( vertexIdx != aSink && aEnabledVertices[ vertexIdx ] ) {
			//printf( "vertexIdx %i orig label %i, label = %i excess = %f\n", vertexIdx, aVertices.getLabel( vertexIdx ), aLabels[ vertexIdx ], aVertices.mExcessArray[ vertexIdx ] );
			int newLabel = aLabels[ vertexIdx ];
			//if (newLabel != MAX_LABEL) 
			{
				aVertices.getLabel( vertexIdx ) = newLabel;
				if ( vertexIdx != aSource ) {
					relabelSuccessful = 1;
					//printf("relabel %i\n", vertexIdx);
				}
			}
		}
	}
}
//*********************************************************************************************************

struct DummyOperation
{
	__device__ bool
	processEdge( const EdgeRecord &rec, int edgeIdx, int aStep, bool aFirst )
	{ return true; }

	__device__ void
	processVertex( int vertexIdx, int aStep )
	{ /*empty*/ }
};

struct SetLabelAsDistance: public DummyOperation
{
	SetLabelAsDistance( int *aLabels ): mLabels( aLabels )
	{}

	__device__ void
	processVertex( int vertexIdx, int aStep )
	{ 	
		CUDA_ASSERT( vertexIdx != 0 );
		mLabels[vertexIdx] = aStep; 
	}
	int *mLabels;
};

struct SetLabelAsDistanceSkipSaturatedEdgesOpposite: public SetLabelAsDistance
{
	SetLabelAsDistanceSkipSaturatedEdgesOpposite( int *aLabels, EdgeResidualsRecord * aResiduals, float *aExcess ): SetLabelAsDistance( aLabels ), mResiduals( aResiduals ), mExcess( aExcess )
	{}
	__device__ bool
	processEdge( const EdgeRecord &rec, int edgeIdx, int aStep, bool aFirst )
	{ 
		bool res = mResiduals[edgeIdx].getResidual( !aFirst ) > 0.0f;
		if( !res ) printf( "%i saturated distance\n", aStep );
		return true;//res;
		//return true; 
	}

	__device__ void
	processVertex( int vertexIdx, int aStep )
	{ 	
		CUDA_ASSERT( vertexIdx != 0 );
		mLabels[vertexIdx] = aStep;
		if ( mExcess[vertexIdx] > 0.0f ) {
			printf( "excess found on distance %i\n", aStep );
			cutFound = 0;
		}
	}

	EdgeResidualsRecord * mResiduals;
	float *mExcess;
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
			bool res = aOperation.processEdge( rec, edgeIdx, aStep, true );
			if ( res && !aVisited[ rec.second ] ) {
				aFrontierNew[ rec.second ] = true;
				bfsFrontNotEmpty = 1;
			}
		}
		if ( aFrontier[ rec.second ] ) {
			bool res = aOperation.processEdge( rec, edgeIdx, aStep, false );
			if ( res && !aVisited[ rec.first ] ) {
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

__global__ void
printNonZeroExcessKernel( VertexList aVertices )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int vertexIdx = blockId * blockDim.x + threadIdx.x;

	if ( vertexIdx < aVertices.size() && vertexIdx > 0 ) {
		if ( aVertices.getExcess(vertexIdx) > 0.0f ) {
			printf( "Vertex %i - excess %f; label %i \n", vertexIdx, aVertices.getExcess(vertexIdx), aVertices.getLabel(vertexIdx) );
		}
	}
}

void
printNonZeroExcess( VertexList &aVertices )
{
	dim3 blockSize1D( 512 );
	dim3 vertexGridSize1D( (aVertices.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );
		
	printNonZeroExcessKernel<<< vertexGridSize1D, blockSize1D >>>( aVertices );
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
	int h = 1 + bfsSearch( aEdges, aVertices, aSource, SetLabelAsDistance( aVertices.mLabelArray ) );
	thrust::transform( 
			thrust::device_pointer_cast( aVertices.mLabelArray + 1 ), 
			thrust::device_pointer_cast( aVertices.mLabelArray + aVertices.size() ), 
			thrust::device_pointer_cast( aVertices.mLabelArray + 1 ),
			RevertLabels( h )
			);

	thrust::device_ptr<int> sourceLabel( aVertices.mLabelArray + aSource );
	thrust::device_ptr<int> sinkLabel( aVertices.mLabelArray + aSink );
	D_PRINT( "SRC label original = " << *sourceLabel );
	D_PRINT( "DST label original = " << *sinkLabel );

	*sinkLabel = 0;
	D_PRINT( "init labels execution time = " << clock.SecondsPassed() );
	//thrust::copy( thrust::device_pointer_cast( aVertices.mLabelArray + 1 ), thrust::device_pointer_cast( aVertices.mLabelArray + aVertices.size() ), std::ostream_iterator<int>(std::cout, "\n"));
}

void
globalRelabel( EdgeList &aEdges, VertexList &aVertices, int aSource, int aSink )
{
	D_COMMAND( M4D::Common::Clock clock; );
	int h = 1 + bfsSearch( aEdges, aVertices, aSink, SetLabelAsDistance( aVertices.mLabelArray ) );
	/*thrust::transform( 
			thrust::device_pointer_cast( aVertices.mLabelArray + 1 ), 
			thrust::device_pointer_cast( aVertices.mLabelArray + aVertices.size() ), 
			thrust::device_pointer_cast( aVertices.mLabelArray + 1 ),
			RevertLabels( h )
			);*/

	thrust::device_ptr<int> sourceLabel( aVertices.mLabelArray + aSource );
	thrust::device_ptr<int> sinkLabel( aVertices.mLabelArray + aSink );
	D_PRINT( "SRC label original = " << *sourceLabel );
	D_PRINT( "DST label original = " << *sinkLabel );

	//*sinkLabel = 0;
	D_PRINT( "globalRelabel execution time = " << clock.SecondsPassed() );
	//thrust::copy( thrust::device_pointer_cast( aVertices.mLabelArray + 1 ), thrust::device_pointer_cast( aVertices.mLabelArray + aVertices.size() ), std::ostream_iterator<int>(std::cout, "\n"));
}

bool
testForCut( EdgeList &aEdges, VertexList &aVertices, int aSource, int aSink )
{
	D_COMMAND( M4D::Common::Clock clock; );
	thrust::device_vector< int > distances( aVertices.size() );
	
	int cutFound = 1;
	CUDA_CHECK_RESULT( cudaMemcpyToSymbol( "cutFound", &(cutFound = 1), sizeof(int), 0, cudaMemcpyHostToDevice ) );

	int h = bfsSearch( aEdges, aVertices, aSink, SetLabelAsDistanceSkipSaturatedEdgesOpposite( distances.data().get(), aEdges.mEdgeResiduals, aVertices.mExcessArray ) );

	CUDA_CHECK_RESULT( cudaMemcpyFromSymbol( &cutFound, "cutFound", sizeof(int), 0, cudaMemcpyDeviceToHost ) );

	D_PRINT( "testForCut execution time = " << clock.SecondsPassed() );
	D_PRINT( "SOURCE dst through unsaturated edges = " << distances[aSource] );
	D_PRINT( "CUT FOUND = " << cutFound );
	return distances[aSource] == 0;
	//thrust::copy( thrust::device_pointer_cast( aVertices.mLabelArray + 1 ), thrust::device_pointer_cast( aVertices.mLabelArray + aVertices.size() ), std::ostream_iterator<int>(std::cout, "\n"));
}

bool
push( EdgeList &aEdges, VertexList &aVertices, int aSource, int aSink )
{
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (aEdges.size() + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	thrust::device_ptr<float > sourceExcess( aVertices.mExcessArray + aSource );
	thrust::device_ptr<float > sinkExcess( aVertices.mExcessArray + aSink );

	//D_PRINT( "gridSize1D " << gridSize1D.x << "; " << gridSize1D.y << "; " << gridSize1D.z );
	static const float SOURCE_EXCESS = 1000000.0f;
	int pushSuccessful;
	D_COMMAND( M4D::Common::Clock clock; );
	size_t pushIterations = 0;
	do {
		//CUDA_CHECK_RESULT( cudaMemcpy( (void*)(aVertices.mExcessArray + aSource), (void*)(&SOURCE_EXCESS), sizeof(float), cudaMemcpyHostToDevice ) );
		*sourceExcess = SOURCE_EXCESS;

		++pushIterations;
		CUDA_CHECK_RESULT( cudaMemcpyToSymbol( "pushSuccessful", &(pushSuccessful = 0), sizeof(int), 0, cudaMemcpyHostToDevice ) );
		
		CheckCudaErrorState( TO_STRING( "Before push kernel n. " << pushIterations ) );
		pushKernel<<< gridSize1D, blockSize1D >>>( aEdges, aVertices );
		CheckCudaErrorState( TO_STRING( "After push iteration n. " << pushIterations ) );
		
		CUDA_CHECK_RESULT( cudaThreadSynchronize() );

		CUDA_CHECK_RESULT( cudaMemcpyFromSymbol( &pushSuccessful, "pushSuccessful", sizeof(int), 0, cudaMemcpyDeviceToHost ) );
		
		D_PRINT( "-----------------------------------" );
		
	} while ( pushSuccessful > 0 );

	D_PRINT( "Push iteration count = " << pushIterations << "; Sink excess = " << *sinkExcess << "; took " << clock.SecondsPassed());
	return pushIterations > 1;
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
	
	//thrust::fill( aLabels.begin(), aLabels.end(), MAX_LABEL );
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
					thrust::raw_pointer_cast(&aLabels[0]),
					aSource,
					aSink 
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

	M4D::Common::Clock clock;

	initLabels( aEdges, aVertices, aSourceID, aSinkID );
	testForCut( aEdges, aVertices, aSourceID, aSinkID );
	bool res = true;
	size_t iteration = 0;
	while( res ) {
		bool pushRes = push( aEdges, aVertices, aSourceID, aSinkID );

		res = relabel( aEdges, aVertices, tmpEnabledVertex, tmpLabels, aSourceID, aSinkID );
		++iteration;
		D_PRINT( "Finished iteration n.: " << iteration << "; Push sucessful = " << pushRes << "; seconds passed: " << clock.secondsPassed() );
		if( iteration % 20 == 0 ) {
			//globalRelabel( aEdges, aVertices, aSourceID, aSinkID );
			//testForCut( aEdges, aVertices, aSourceID, aSinkID );
		}
	}

	LOG( "Push relabel took " << clock.secondsPassed() << " seconds" );
	printNonZeroExcess( aVertices );
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


void
minGraphCut( WeightedEdgeListGraph &aGraph, std::vector< bool > &aComponentSet, int aSourceID, int aSinkID )
{
	thrust::device_vector< EdgeRecord > edges( aGraph.mEdgeCount );
	thrust::device_vector< float > weights( aGraph.mEdgeCount );

	

	thrust::host_vector< EdgeRecord > host_edges(aGraph.mEdgeCount);
	std::copy( aGraph.mEdges.begin(), aGraph.mEdges.end(), reinterpret_cast<WeightedEdgeListGraph::EdgeRecord*>(&host_edges[0]) );
	thrust::copy( host_edges.begin(), host_edges.end(), edges.begin() );
	host_edges.clear();

	thrust::copy( aGraph.mWeights.begin(), aGraph.mWeights.end(), weights.begin() );

	pushRelabelMaxFlow( aGraph.mEdgeCount, aGraph.mVertexCount, edges, weights, aSourceID, aSinkID );
}


