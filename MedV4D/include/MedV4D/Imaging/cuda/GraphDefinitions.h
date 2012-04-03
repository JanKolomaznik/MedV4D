#ifndef CUDA_GRAPH_DEFINITIONS_H
#define CUDA_GRAPH_DEFINITIONS_H

#include <cuda.h>
#include "MedV4D/Common/Common.h"

struct EdgeResidualsRecord
{
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
	__device__ __host__ int
	size()const
	{ return 0; }
	
	__device__ EdgeRecord &
	getEdge( int aIdx )
	{ return mEdges[aIdx]; }
	
	__device__ EdgeResidualsRecord &
	getResiduals( int aIdx )
	{
		return mEdgeResiduals[ mEdgeIndices[ aIdx ] ];
	}

	EdgeRecord *mEdges;
	EdgeResidualsRecord *mEdgeResiduals;
	float *mWeights;
	int *mEdgeIndices;
};

struct VertexList
{
	__device__ __host__ int
	size()const
	{ return 0; }

	__device__ float &
	getExcess( int aIdx )
	{ return mExcessArray[aIdx]; }

	__device__ int &
	getLabel( int aIdx )
	{ return mLabelArray[aIdx]; }

	float *mExcessArray;
	int *mLabelArray;
};

#endif //CUDA_GRAPH_DEFINITIONS_H