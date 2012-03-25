#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <cuda.h>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/ImageRegion.h"

#include "MedV4D/Imaging/cuda/AdjacencyGraph.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>

__device__ int edgeInsertions;

struct VertexRecord
{
	size_t edgeStart;
};

struct EdgeRecord
{
	__host__ __device__  
	EdgeRecord( uint32 aFirst, uint32 aSecond )
	{
		first = aFirst;
		second = aSecond;
	}
	__host__ __device__  
	EdgeRecord(): edgeCombIdx(0)
	{ }

	union {
		uint64 edgeCombIdx;
		struct {
			uint32 second;
			uint32 first;
		};
	};
};

struct GetSymmetricEdge : public thrust::unary_function< EdgeRecord, EdgeRecord >
{
	__host__ __device__ EdgeRecord operator()(const EdgeRecord &aEdge) const 
	{
		EdgeRecord rec;// = aEdge;
		//rec.edgeCombIdx = ((aEdge.edgeCombIdx & 0xFFFFFFFF) << 32) + (aEdge.edgeCombIdx >> 32);
		rec.second = aEdge.first;
		rec.first = aEdge.second;
		return rec;
	}
	__host__ __device__ thrust::tuple< EdgeRecord, float > operator()(const thrust::tuple< EdgeRecord, float > &aEdge) const 
	{
		thrust::tuple< EdgeRecord, float > rec;// = aEdge;
		//rec.edgeCombIdx = ((aEdge.edgeCombIdx & 0xFFFFFFFF) << 32) + (aEdge.edgeCombIdx >> 32);
		rec.get<0>().second = aEdge.get<0>().first;
		rec.get<0>().first = aEdge.get<0>().second;
		return rec;
	}
};

struct CompareEdge : public thrust::binary_function< EdgeRecord, EdgeRecord, bool>
{
	__host__ __device__ bool operator()( const EdgeRecord &aEdgeA, const EdgeRecord &aEdgeB ) const 
	{
		if ( aEdgeA.edgeCombIdx == 0 ) return false;//aEdgeB.edgeCombIdx == 0;
		if ( aEdgeB.edgeCombIdx == 0 ) return true;
		
		return aEdgeA.edgeCombIdx < aEdgeB.edgeCombIdx;
	}

	__host__ __device__ bool operator()( const thrust::tuple< EdgeRecord, float > &aEdgeA, const thrust::tuple< EdgeRecord, float > &aEdgeB ) const 
	{
		if ( aEdgeA.get<0>().edgeCombIdx == 0 ) return false;//aEdgeB.edgeCombIdx == 0;
		if ( aEdgeB.get<0>().edgeCombIdx == 0 ) return true;
		
		return aEdgeA.get<0>().edgeCombIdx < aEdgeB.get<0>().edgeCombIdx;
	}
};

struct IsValidEdge : public thrust::unary_function< EdgeRecord, EdgeRecord >
{
	__host__ __device__ bool operator()(const EdgeRecord &aEdge) const 
	{
		return aEdge.edgeCombIdx != 0;
	}
};

inline __device__ void
hashEdge( EdgeRecord *aTable, float *aWeights, size_t aSize, const EdgeRecord &aEdge, float aWeight )
{
	size_t idx = (0x2C87*(aEdge.edgeCombIdx % 0xC2FF97889)+0x159732CF) % aSize;
	//size_t counter = 0;
	//if( aEdge.edgeCombIdx == 0) 
		//atomicAdd( &edgeInsertions, 1 );
	bool inserted = false;
	while ( !inserted ) {
		if ( aTable[idx].edgeCombIdx == 0 ) {
			if ( atomicCAS( &(aTable[idx].edgeCombIdx), uint64(0), aEdge.edgeCombIdx ) == 0 ) {
				atomicAdd( aWeights + idx, aWeight );
				inserted = true;
				return;
			}
		} else {
			if ( aTable[idx].edgeCombIdx == aEdge.edgeCombIdx ) {
				//atomicAdd( &(aTable[idx].count), aEdge.count );
				//atomicAdd( &(aTable[idx].weight), aEdge.weight );
				atomicAdd( aWeights + idx, aWeight );
				inserted = true;
				return;
			}
		}
		idx = (idx+1) % aSize;
	}
}


class EdgeHashTable: public Buffer1D<EdgeRecord>
{
public:
	EdgeHashTable( thrust::device_vector< EdgeRecord > &aVect, thrust::device_vector< float > &aWeights )
		: Buffer1D<EdgeRecord>( aVect.size(), thrust::raw_pointer_cast(&aVect[0]) ), 
		mWeights( aWeights.size(), thrust::raw_pointer_cast(&aWeights[0]) )
	{ }

	EdgeHashTable( size_t aLength, EdgeRecord *aData, float *aWeights ): Buffer1D<EdgeRecord>( aLength, aData ), mWeights( aLength, aWeights )
	{ }

	__device__ void 
	insertEdge( const EdgeRecord & aEdge, float aWeight )
	{
		hashEdge( mData, mWeights.mData, mLength, aEdge, aWeight );
	}

	Buffer1D<float> mWeights;
};

__global__ void 
fillVertexNeighbors( const EdgeRecord *aEdges, size_t aEdgeCount, VertexRecord *aVertices, size_t aVertexCount )
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	int idx = blockId * blockDim.x + threadIdx.x;

	if ( idx < aEdgeCount-1 ) {
		uint first = aEdges[idx].first;// >> 32;
		uint second = aEdges[idx+1].first;// >> 32;
		if( first != second ) {
			aVertices[ second ].edgeStart = idx+1;
		}
	}
}


template< typename TEType >
__global__ void 
preallocationOfAdjacencyGraph( Buffer3D< uint32 > aRegionBuffer, Buffer3D< TEType > aGradientBuffer, EdgeHashTable aEdgeHashMap, int3 blockResolution )
{
	const size_t EDGE_HASH_TABLE_SIZE = 3*8*8*8;
	__shared__ uint32 inData[10*10*10];
	__shared__ EdgeRecord edgeHashTable[ EDGE_HASH_TABLE_SIZE ];
	__shared__ float edgeWeightTable[ EDGE_HASH_TABLE_SIZE ];
	size_t hashingIndex = 3*( (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) +  threadIdx.x );
	const int cBlockDim = 8;
	const int cRadius = 1;
	const uint syStride = cBlockDim+2*cRadius;
	const uint szStride = (cBlockDim+2*cRadius) * (cBlockDim+2*cRadius);

	uint sidx = (threadIdx.y+cRadius) * syStride + (threadIdx.z+cRadius) * szStride + threadIdx.x + cRadius;
	int3 size = toInt3( aRegionBuffer.mSize );
	int3 asize = size - make_int3( 1, 1, 1 );
	int3 blockCoordinates = GetBlockCoordinates( blockResolution, __mul24(blockIdx.y, gridDim.x) + blockIdx.x );
	int3 blockOrigin = GetBlockOrigin( blockDim, blockCoordinates );
	
	
	int3 coordinates = blockOrigin + threadIdx;
	bool projected = ProjectionToInterval( coordinates, make_int3(0,0,0), asize );
	int idx = IdxFromCoordStrides( coordinates, aRegionBuffer.mStrides );
	int idx2 = IdxFromCoordStrides( coordinates, aGradientBuffer.mStrides );

	
	FillSharedMemory3D_8x8x8< uint32, cRadius, syStride, szStride >( inData, sidx, aRegionBuffer.mData, aRegionBuffer.mStrides, size, blockOrigin, coordinates, idx );
	
	__syncthreads();
	if ( projected ) {
		return;
	}
	//uint32 current = aRegionBuffer.mData[idx];
	uint32 current = inData[sidx];
	uint32 second = inData[sidx+1];
	//uint32 second = aRegionBuffer.mData[idx+1];
	TEType val = aGradientBuffer.mData[idx2];
	if( current == 0 ) return;
	//assert( current );
	
	if ( current != second ) {
		float weight = max( val, aGradientBuffer.mData[idx2+1] );
		//assert( second );
		if( second ) hashEdge( edgeHashTable, edgeWeightTable, EDGE_HASH_TABLE_SIZE, EdgeRecord( current, second/*, weight*/ ), weight );//aEdgeHashMap.insertEdge( EdgeRecord( current, second, weight ) );
	}
	second = inData[sidx+syStride];
	//second = aRegionBuffer.mData[idx+aRegionBuffer.mStrides.y];
	if ( current != second ) {
		float weight = max( val, aGradientBuffer.mData[idx2+aGradientBuffer.mStrides.y] );
		//assert( second );
		if( second ) hashEdge( edgeHashTable, edgeWeightTable, EDGE_HASH_TABLE_SIZE, EdgeRecord( current, second/*, weight*/ ), weight );//aEdgeHashMap.insertEdge( EdgeRecord( current, second, weight ) );
	}
	second = inData[sidx+szStride];
	//second = aRegionBuffer.mData[idx+aRegionBuffer.mStrides.z];
	if ( current != second ) {
		float weight = max( val, aGradientBuffer.mData[idx2+aGradientBuffer.mStrides.z] );
		//assert( second );
		if( second ) hashEdge( edgeHashTable, edgeWeightTable, EDGE_HASH_TABLE_SIZE, EdgeRecord( current, second/*, weight*/ ), weight );//aEdgeHashMap.insertEdge( EdgeRecord( current, second, weight ) );
	}
	__syncthreads();
	for( size_t i = 0; i < 3; ++i ) {
		if( edgeHashTable[hashingIndex + i].edgeCombIdx != 0 ) {
			aEdgeHashMap.insertEdge( edgeHashTable[hashingIndex + i], edgeWeightTable[hashingIndex + i]  );
		}
	}
	
}

template< typename TEType >
void
fillEdgeList( Buffer3D< uint32 > &aRegionBuffer, Buffer3D< TEType > &aGradientBuffer, thrust::device_vector< EdgeRecord > &aEdges, thrust::device_vector< float > aEdgeWeights, size_t &aEdgeCount )
{
		int3 radius = make_int3( 1, 1, 1 );

	EdgeHashTable edgeHashMap( aEdges.size(), thrust::raw_pointer_cast(&aEdges[0]), thrust::raw_pointer_cast(&aEdgeWeights[0]) );

	dim3 blockSize3D( 8, 8, 8 );
	int3 blockResolution3D = GetBlockResolution( aRegionBuffer.mSize, blockSize3D, make_int3(0,0,0) );
	dim3 gridSize3D( blockResolution3D.x * blockResolution3D.y, blockResolution3D.z, 1 );

	int edgeInsertions;
	cudaMemcpyToSymbol( "edgeInsertions", &(edgeInsertions = 0), sizeof(int), 0, cudaMemcpyHostToDevice );

	M4D::Common::Clock clock;
	CheckCudaErrorState( "Before preallocationOfAdjacencyGraph()" );
	preallocationOfAdjacencyGraph< TEType >
		<<< gridSize3D, blockSize3D >>>( aRegionBuffer, aGradientBuffer, edgeHashMap, blockResolution3D );
	cudaThreadSynchronize();
	
	cudaMemcpyFromSymbol( &edgeInsertions, "edgeInsertions", sizeof(int), 0, cudaMemcpyDeviceToHost );
	D_PRINT( "edge insertions " << edgeInsertions );

	CheckCudaErrorState( "After preallocationOfAdjacencyGraph()" );
	//size_t testedgeCount = thrust::count_if( aEdges.begin(), aEdges.end(), IsValidEdge() );
	//D_PRINT( "debug created " << testedgeCount << " edges" );
	D_PRINT( "Time after preallocationOfAdjacencyGraph() " << clock.SecondsPassed() );

	aEdgeCount = thrust::count_if( aEdges.begin(), aEdges.end(), IsValidEdge() );
	
	//thrust::sort( aEdges.begin(), aEdges.end(), CompareEdge() );
	thrust::sort( 
			thrust::make_zip_iterator( thrust::make_tuple( aEdges.begin(), aEdgeWeights.begin() ) ), 
			thrust::make_zip_iterator( thrust::make_tuple( aEdges.end(), aEdgeWeights.end() ) ), 
			CompareEdge() 
			);

	D_COMMAND( EdgeRecord rec = aEdges[aEdgeCount-1]; ); ASSERT( rec.edgeCombIdx != 0 );
	D_COMMAND( rec = aEdges[aEdgeCount]; ); ASSERT( rec.edgeCombIdx == 0 );
}

template< typename TEType >
void
computeRegionAdjacencyGraph( Buffer3D< uint32 > &aRegionBuffer, Buffer3D< TEType > &aGradientBuffer, thrust::device_vector< EdgeRecord > &aEdges, thrust::device_vector< float > aEdgeWeights, size_t &aEdgeCount, thrust::device_vector< VertexRecord > &aVertices )
{

	ASSERT( aEdges.size() > 2*aEdgeCount );
	
	M4D::Common::Clock clock;

	fillEdgeList( aRegionBuffer, aGradientBuffer, aEdges, aEdgeWeights, aEdgeCount );

	//thrust::transform( aEdges.begin(), aEdges.begin() + aEdgeCount, aEdges.begin() + aEdgeCount, GetSymmetricEdge() );
	thrust::transform( 
			thrust::make_zip_iterator( thrust::make_tuple( aEdges.begin(), aEdgeWeights.begin() ) ), 
			thrust::make_zip_iterator( thrust::make_tuple( aEdges.begin() + aEdgeCount, aEdgeWeights.begin() + aEdgeCount ) ),
			thrust::make_zip_iterator( thrust::make_tuple( aEdges.begin() + aEdgeCount, aEdgeWeights.begin() + aEdgeCount ) ), 
			GetSymmetricEdge() 
			);
	D_PRINT( "Time after graph mirroring " << clock.SecondsPassed() );
	aEdgeCount *= 2;
	//thrust::sort( aEdges.begin(), aEdges.begin() + aEdgeCount, CompareEdge() );
	thrust::sort( 
			thrust::make_zip_iterator( thrust::make_tuple( aEdges.begin(), aEdgeWeights.begin() ) ), 
			thrust::make_zip_iterator( thrust::make_tuple( aEdges.begin() + aEdgeCount, aEdgeWeights.begin() + aEdgeCount ) ),
			CompareEdge() 
			);
	D_PRINT( "Time after edge sorting " << clock.SecondsPassed() );
	D_PRINT( "created " << aEdgeCount << " edges" );

	D_PRINT( "Filling info for vertices" );
	dim3 blockSize1D( 512 );
	dim3 gridSize1D( (aEdgeCount + 64*blockSize1D.x - 1) / (64*blockSize1D.x) , 64 );

	fillVertexNeighbors<<< gridSize1D, blockSize1D >>>( thrust::raw_pointer_cast(&aEdges[0]), aEdgeCount, thrust::raw_pointer_cast(&aVertices[0]), aVertices.size() );
	
	LOG( "computeRegionAdjacencyGraph computations took " << clock.SecondsPassed() );
}

template< typename TEType >
void
createAdjacencyGraphGPU( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput, size_t aRegionCount )
{
	D_PRINT( "Before " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
	Buffer3D< uint32 > labeledRegionsBuffer = CudaBuffer3DFromImageRegionCopy( aLabeledMarkerRegions );
	aRegionCount = *(thrust::max_element( thrust::device_pointer_cast( labeledRegionsBuffer.mData ), thrust::device_pointer_cast( labeledRegionsBuffer.mData+labeledRegionsBuffer.mLength ) ));
	D_PRINT( "Region count " << aRegionCount );

	Buffer3D< TEType > inputBuffer = CudaBuffer3DFromImageRegionCopy( aInput );

	thrust::device_vector< VertexRecord > vertices( aRegionCount+10 );
	thrust::device_vector< EdgeRecord > edges( aRegionCount*25 );
	thrust::device_vector< float > edgeWeights( edges.size() );
	size_t edgeCount = 0;
	D_PRINT( "After allocation in " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
	computeRegionAdjacencyGraph( labeledRegionsBuffer, inputBuffer, edges, edgeWeights, edgeCount, vertices );
	D_PRINT( "After " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
}

template< typename TEType >
void
createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput )
{
	M4D::Common::Clock clock;
	D_PRINT( "Before " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
	Buffer3D< uint32 > labeledRegionsBuffer = CudaBuffer3DFromImageRegionCopy( aLabeledMarkerRegions );
	size_t regionCount = *(thrust::max_element( thrust::device_pointer_cast( labeledRegionsBuffer.mData ), thrust::device_pointer_cast( labeledRegionsBuffer.mData+labeledRegionsBuffer.mLength ) ));
	D_PRINT( "Region count " << regionCount );

	Buffer3D< TEType > inputBuffer = CudaBuffer3DFromImageRegionCopy( aInput );

	thrust::device_vector< EdgeRecord > edges( regionCount*25 );
	thrust::device_vector< float > edgeWeights( edges.size() );

	size_t edgeCount = 0;
	D_PRINT( "After allocation in " << __FUNCTION__ << ": " << cudaMemoryInfoText() );
	fillEdgeList( labeledRegionsBuffer, inputBuffer, edges, edgeWeights, edgeCount );
	D_PRINT( "After " << __FUNCTION__ << ": " << cudaMemoryInfoText() );


	thrust::host_vector< EdgeRecord > host_edges(edgeCount);
	thrust::host_vector< float > host_weights(edgeCount);

	thrust::copy( edges.begin(), edges.begin() + edgeCount, host_edges.begin() );
	thrust::copy( edgeWeights.begin(), edgeWeights.begin() + edgeCount, host_weights.begin() );
	aGraph = WeightedUndirectedGraph( host_edges.begin(), host_edges.end(), host_weights.begin(), regionCount+1 );
	LOG( "createAdjacencyGraph() computations took " << clock.SecondsPassed() );
}

template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int8, 3 > aInput );
template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint8, 3 > aInput );
template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int16, 3 > aInput );
template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint16, 3 > aInput );
template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int32, 3 > aInput );
template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint32, 3 > aInput );
template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int64, 3 > aInput );
template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint64, 3 > aInput );
template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< float, 3 > aInput );
template void createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< double, 3 > aInput );

/*template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int8, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint8, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int16, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint16, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int32, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint32, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< int64, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< uint64, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< float, 3 > aInput, size_t aRegionCount );
template void createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< double, 3 > aInput, size_t aRegionCount );
*/
