#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

template< typename RegionType >
void
Sobel3D( RegionType input, RegionType output, typename RegionType::ElementType threshold );

#endif //EDGE_DETECTION_H