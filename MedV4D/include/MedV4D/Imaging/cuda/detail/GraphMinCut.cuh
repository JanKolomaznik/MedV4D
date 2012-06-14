#ifndef CUDA_GRAPH_MIN_CUT_CUH
#define CUDA_GRAPH_MIN_CUT_CUH

#include <cuda.h>
#include "MedV4D/Common/Common.h"

#include "MedV4D/Imaging/cuda/GraphDefinitions.h"

#include <thrust/device_vector.h>

void 
pushRelabelMaxFlow( EdgeList &aEdges, VertexList &aVertices, int aSourceID, int aSinkID );

void
pushRelabelMaxFlow( size_t aEdgeCount, size_t aVertexCount, thrust::device_vector< EdgeRecord > &aEdges, thrust::device_vector< float > &aWeights, int aSourceID, int aSinkID );

#endif //CUDA_GRAPH_MIN_CUT_CUH
