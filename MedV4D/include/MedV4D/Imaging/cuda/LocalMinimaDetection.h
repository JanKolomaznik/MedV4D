#ifndef LOCAL_MINIMA_DETECTION_H
#define LOCAL_MINIMA_DETECTION_H

template< typename InRegionType >
void
LocalMinima3D( InRegionType input,  M4D::Imaging::MaskRegion3D output, typename InRegionType::ElementType aThreshold );

template< typename InRegionType >
void
LocalMinimaRegions3D( InRegionType input,  M4D::Imaging::ImageRegion< uint32, 3 > output, typename InRegionType::ElementType aThreshold );


#endif LOCAL_MINIMA_DETECTION_H