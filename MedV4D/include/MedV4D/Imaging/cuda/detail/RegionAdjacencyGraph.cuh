#ifndef CUDA_REGION_ADJACENCY_GRAPH_CUH
#define CUDA_REGION_ADJACENCY_GRAPH_CUH

#include "MedV4D/Imaging/cuda/detail/CUDAFiltersUtils.cuh"
#include <thrust/device_vector.h>
#include "MedV4D/Imaging/cuda/GraphDefinitions.h"

template< typename TEType >
void
fillEdgeList( const Buffer3D< uint32 > &aRegionBuffer, const Buffer3D< TEType > &aGradientBuffer, thrust::device_vector< EdgeRecord > &aEdges, thrust::device_vector< float > &aEdgeWeights, size_t &aEdgeCount );


#endif //CUDA_REGION_ADJACENCY_GRAPH_CUH