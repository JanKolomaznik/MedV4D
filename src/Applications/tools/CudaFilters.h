#ifndef CUDA_FILTERS_H
#define CUDA_FILTERS_H

#include "Imaging/ImageRegion.h"

template< typename RegionType >
void
Sobel3D( RegionType input, RegionType output, typename RegionType::ElementType threshold );

template< typename InRegionType >
void
LocalMinima3D( InRegionType input,  M4D::Imaging::MaskRegion3D output, typename InRegionType::ElementType aThreshold );

template< typename InRegionType >
void
LocalMinimaRegions3D( InRegionType input,  M4D::Imaging::ImageRegion< uint32, 3 > output, typename InRegionType::ElementType aThreshold );

template< typename InRegionType >
void
RegionBorderDetection3D( InRegionType input,  M4D::Imaging::MaskRegion3D output );

void
ConnectedComponentLabeling3D( M4D::Imaging::MaskRegion3D input, M4D::Imaging::ImageRegion< uint32, 3 > output );

template< typename TEType >
void
WatershedTransformation3D( 
			M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, 
			M4D::Imaging::ImageRegion< TEType, 3 > aInput, 
			M4D::Imaging::ImageRegion< uint32, 3 > aOutput );




#endif /*CUDA_FILTERS_H*/
