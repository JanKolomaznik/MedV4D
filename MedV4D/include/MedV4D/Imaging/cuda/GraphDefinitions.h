#ifndef CUDA_GRAPH_DEFINITIONS_H
#define CUDA_GRAPH_DEFINITIONS_H

#include <cuda.h>
#include "MedV4D/Common/Common.h"
#include <thrust/device_vector.h>

#define CUDA_ASSERT( expression ) assert( expression )

struct EdgeResidualsRecord
{
	__host__ __device__ 
	EdgeResidualsRecord( float aWeight = 0.0f )
	{
		residuals[0] = residuals[1] = aWeight;
	}
	
	__host__ __device__ float &
	getResidual( bool aFirst )
	{
		return aFirst ? residuals[0] : residuals[1];
	}
	float residuals[2];
};


struct VertexRecord
{
	size_t edgeStart;
};

struct EdgeRecord
{
	__host__ __device__  
	EdgeRecord( uint32 aFirst, uint32 aSecond )
	{
		first = min( aFirst, aSecond );
		second = max( aFirst, aSecond );
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

struct EdgeList
{
	EdgeList( thrust::device_vector< EdgeRecord > &aEdges, thrust::device_vector< float > &aWeights, thrust::device_vector< EdgeResidualsRecord > &aResiduals, int aEdgeCount )
		: mEdges( aEdges.data().get() ), mWeights( aWeights.data().get() ), mEdgeResiduals( aResiduals.data().get() ), mSize( aEdgeCount )
	{
		
	}
	__device__ __host__ int
	size()const
	{ return mSize; }
	
	/*__device__ __host__ int
	residualsCount()const
	{ return 0; }*/
	
	__device__ EdgeRecord &
	getEdge( int aIdx )
	{ 
		CUDA_ASSERT( aIdx < size() );
		CUDA_ASSERT( mEdges != NULL );
		return mEdges[aIdx]; 
	}
	
	/*__device__ int &
	getEdgeIndexToOtherStructures( int aIdx ) const
	{
		CUDA_ASSERT( aIdx < size() );
		int tmp = mEdgeIndices[ aIdx ];
		CUDA_ASSERT( tmp < residualsCount() );
		
		
		return tmp;
	}*/
	
	__device__ EdgeResidualsRecord &
	getResiduals( int aIdx )
	{
		CUDA_ASSERT( aIdx < size() );
		CUDA_ASSERT( mEdgeResiduals != NULL );
		return mEdgeResiduals[ aIdx ];
	}
	
	

	EdgeRecord *mEdges;
	float *mWeights;
	EdgeResidualsRecord *mEdgeResiduals;
	int mSize;
	//int *mEdgeIndices;
};

struct VertexList
{
	VertexList( thrust::device_vector< int >  &aLabels, thrust::device_vector< float >  &aExcess, int aVertexCount )
		: mLabelArray( aLabels.data().get() ), mExcessArray( aExcess.data().get() ), mSize( aVertexCount + 1 )
	{
		
	}
	
	__device__ __host__ int
	size()const
	{ return mSize; }

	__device__ float &
	getExcess( int aIdx )
	{ 
		CUDA_ASSERT( aIdx < size() && aIdx > 0 );
		return mExcessArray[aIdx]; 
	}

	__device__ int &
	getLabel( int aIdx )
	{ 
		CUDA_ASSERT( aIdx < size() && aIdx > 0 );
		return mLabelArray[aIdx]; 
	}

	int *mLabelArray;
	float *mExcessArray;
	int mSize;
};

#endif //CUDA_GRAPH_DEFINITIONS_H