#ifndef GRAPH_OPERATIONS_H
#define GRAPH_OPERATIONS_H

#include "MedV4D/Common/GraphTools.h"
#include "MedV4D/Imaging/ImageRegion.h"

template< typename TEType >
void
createAdjacencyGraph( WeightedUndirectedGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput );

template< typename TEType >
void
createAdjacencyGraph( WeightedEdgeListGraph &aGraph, M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput );


template< typename TEType >
void
pushRelabelMaxFlow( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput );


void
minGraphCut( WeightedEdgeListGraph &aGraph, std::vector< bool > &aComponentSet, int aSourceID, int aSinkID );

#endif //ADJACENCY_GRAPH_H
