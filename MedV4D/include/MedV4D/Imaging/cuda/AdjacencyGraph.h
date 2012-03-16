#ifndef ADJACENCY_GRAPH_H
#define ADJACENCY_GRAPH_H

template< typename TEType >
void
createAdjacencyGraph( M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, M4D::Imaging::ImageRegion< TEType, 3 > aInput, size_t aRegionCount );


#endif //ADJACENCY_GRAPH_H