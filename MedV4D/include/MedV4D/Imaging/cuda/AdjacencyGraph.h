#ifndef ADJACENCY_GRAPH_H
#define ADJACENCY_GRAPH_H

#include "MedV4D/Common/GraphTools.h"

template< typename TEType >
void
createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput );



#endif //ADJACENCY_GRAPH_H